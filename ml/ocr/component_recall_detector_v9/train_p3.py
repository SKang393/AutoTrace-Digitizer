# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer P3 logit-scale repair for OCR detector V9."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.component_fusion_detector_v8.train_p1 import _export

from .dataset import build_split, proposal_examples, split_fingerprint
from .model import ComponentRecallNet
from .model_p3 import OUTPUT_LOGIT_SCALE, ScaledComponentRecallNet
from .pipeline_p2 import evaluate_thresholds
from .protocol import REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/ocr/component_recall_detector_v9/training/p3.json")
CANONICAL_OUTPUT = Path("ml/ocr/component_recall_detector_v9/artifacts/P3-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/component_recall_detector_v9/dataset.py"),
    Path("ml/ocr/component_recall_detector_v9/model.py"),
    Path("ml/ocr/component_recall_detector_v9/model_p3.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline_p2.py"),
    Path("ml/ocr/component_recall_detector_v9/protocol.py"),
    Path("ml/ocr/component_recall_detector_v9/train_p3.py"),
    Path("ml/ocr/component_fusion_detector_v8/dataset.py"),
    Path("ml/ocr/component_fusion_detector_v8/pipeline.py"),
    Path("ml/ocr/component_fusion_detector_v8/train_p1.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def validate_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    optimizer_steps = 0
    try:
        p2_result_path = REPO_ROOT / config["p2_result_path"]
        if sha256_file(p2_result_path) != config["p2_result_sha256"]:
            raise RuntimeError("OCR V9 P2 result checksum changed")
        p2_result = json.loads(p2_result_path.read_text(encoding="utf-8"))
        p2_report_path = REPO_ROOT / p2_result["candidate_report_path"]
        checkpoint_path = REPO_ROOT / p2_result["checkpoint_path"]
        if sha256_file(p2_report_path) != p2_result["candidate_report_sha256"]:
            raise RuntimeError("OCR V9 P2 direct report checksum changed")
        if sha256_file(checkpoint_path) != p2_result["checkpoint_sha256"]:
            raise RuntimeError("OCR V9 P2 checkpoint checksum changed")
        if (
            p2_result.get("status") != "failed_parity"
            or p2_result.get("selection_gate_passed") is not True
            or p2_result.get("selection_exact_scene_count") != p2_result.get("selection_scene_count")
            or p2_result.get("selection_false_positives") != 0
            or p2_result.get("selection_false_negatives") != 0
            or p2_result.get("public_gate_archive_opened") is not False
            or p2_result.get("half_scaled_diagnostic_maximum_absolute_error") > config["onnx_parity_tolerance"]
        ):
            raise RuntimeError("OCR V9 P2 evidence does not authorize the P3 parity repair")
        if config["output_logit_scale"] != OUTPUT_LOGIT_SCALE or config["optimizer_steps"] != 0:
            raise RuntimeError("OCR V9 P3 output-scale or zero-optimizer contract changed")
        selection_path = REPO_ROOT / config["selection_manifest_path"]
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("OCR V9 selection manifest checksum changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("OCR V9 public seal checksum changed")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / seal["fixture_archive_path"]) != seal["fixture_archive_sha256"]:
            raise RuntimeError("OCR V9 sealed-public archive changed before P3 validation")
        validation_scenes = build_split("validation")
        if split_fingerprint(validation_scenes) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V9 validation renderer changed after freeze")
        validation_values, _ = proposal_examples(validation_scenes)
        base = ComponentRecallNet(seed=int(config["seed"]))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        base.load_state_dict(checkpoint["state_dict"])
        model = ScaledComponentRecallNet(base.eval()).eval()
        phase = "export"
        onnx_path = output_dir / "graph-text-component-recall-v9-p3.onnx"
        parity_values = torch.from_numpy(validation_values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V9 P3 validation requires CPUExecutionProvider only")
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])

        def onnx_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(
                session.run(None, {"region_proposals": np.ascontiguousarray(values, dtype=np.float32)})[0],
                dtype=np.float32,
            )

        phase = "selection"
        comparisons = evaluate_thresholds(
            validation_scenes, onnx_runner, tuple(float(value) for value in config["selection_thresholds"])
        )
        selected = max(
            comparisons,
            key=lambda item: (
                item["metrics"]["exact_scene_count"],
                -item["metrics"]["false_positives"],
                -item["metrics"]["false_negatives"],
                -item["metrics"]["duplicate_region_count"],
                item["threshold"],
            ),
        )
        metrics = selected["metrics"]
        selection_passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == metrics["false_negatives"] == 0
            and metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-component-recall-training-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if selection_passed and parity_passed else "failed_selection",
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "generalization_label_included": False,
            "predecessor_fixture_bytes_reused": False,
            "prior_validation_pixels_used_for_training": False,
            "training_authorization": authorization.binding,
            "p2_result_path": config["p2_result_path"],
            "p2_result_sha256": config["p2_result_sha256"],
            "source_checkpoint_path": p2_result["checkpoint_path"],
            "source_checkpoint_sha256": p2_result["checkpoint_sha256"],
            "isolated_change": config["isolated_change"],
            "output_logit_scale": OUTPUT_LOGIT_SCALE,
            "optimizer_steps": optimizer_steps,
            "validation_scene_count": len(validation_scenes),
            "selection_execution": "exact exported ONNX via CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "selected_threshold": selected["threshold"],
            "selection_metrics": metrics,
            "selection_gate_passed": selection_passed,
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_passed,
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "sealed_public_archive_opened": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-component-recall-training-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False,
            "phase": phase,
            "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    arguments = parser.parse_args()
    report = validate_candidate(REPO_ROOT / arguments.output)
    print(json.dumps({
        "candidate_id": report["candidate_id"],
        "status": report["status"],
        "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report["selected_threshold"],
        "selection_gate_passed": report["selection_gate_passed"],
        "onnx_parity_passed": report["onnx_parity_passed"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
