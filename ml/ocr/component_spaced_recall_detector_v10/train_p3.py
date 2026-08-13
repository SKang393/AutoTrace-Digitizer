# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V10 P3 evidence-path recovery using exact P2 model bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.component_recall_detector_v9.model import ComponentRecallNet
from ml.ocr.component_recall_detector_v9.model_p3 import ScaledComponentRecallNet

from .dataset import build_split, encode_proposal, proposal_summary, proposals, split_fingerprint
from .pipeline import evaluate_thresholds
from .protocol import REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/component_spaced_recall_detector_v10")
CONFIG_PATH = ROOT / "training/p3.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p3.py",
    Path("ml/ocr/component_recall_detector_v9/dataset.py"),
    Path("ml/ocr/component_recall_detector_v9/model.py"),
    Path("ml/ocr/component_recall_detector_v9/model_p3.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline_p2.py"),
    Path("ml/ocr/component_recall_detector_v9/protocol.py"),
    Path("ml/ocr/component_fusion_detector_v8/dataset.py"),
    Path("ml/ocr/component_fusion_detector_v8/pipeline.py"),
    Path("ml/ocr/component_fusion_detector_v8/protocol.py"),
    Path("ml/ocr/component_fusion_detector_v8/train_p1.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/protocol.py"),
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/production_composition_v1/protocol.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _parity_values(scenes: tuple[object, ...], limit: int = 256) -> np.ndarray:
    values: list[np.ndarray] = []
    for scene in scenes:
        for candidate in proposals(scene.raster):
            values.append(encode_proposal(scene.raster, candidate))
            if len(values) == limit:
                return np.stack(values).astype(np.float32)
    raise RuntimeError("OCR V10 P3 validation split did not provide enough parity proposals")


def evaluate_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V10 P3 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id="P3",
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(
            REPO_ROOT, RUNNER_SOURCE_PATHS
        ):
            raise RuntimeError("OCR V10 P3 runner source changed")
        if config["p2_result_sha256"] != sha256_file(REPO_ROOT / P2_RESULT_PATH):
            raise RuntimeError("OCR V10 P2 failure result changed")
        p2 = json.loads((REPO_ROOT / P2_RESULT_PATH).read_text(encoding="utf-8"))
        if (
            p2.get("status") != "failed_runner"
            or p2.get("failure_phase") != "selection"
            or p2.get("selection_metrics_available_for_approval") is not False
            or p2.get("public_gate_archive_opened") is not False
        ):
            raise RuntimeError("OCR V10 P2 failure does not authorize P3")
        for configured, recorded in (
            ("p2_candidate_report_sha256", "candidate_report_sha256"),
            ("p2_checkpoint_sha256", "checkpoint_sha256"),
            ("p2_onnx_sha256", "onnx_sha256"),
        ):
            if config[configured] != p2[recorded]:
                raise RuntimeError(f"OCR V10 P3 {configured} disagrees with P2 result")
        if sha256_file(REPO_ROOT / p2["candidate_report_path"]) != p2["candidate_report_sha256"]:
            raise RuntimeError("OCR V10 P2 failure report bytes changed")
        if sha256_file(REPO_ROOT / p2["checkpoint_path"]) != p2["checkpoint_sha256"]:
            raise RuntimeError("OCR V10 P2 checkpoint bytes changed")
        if sha256_file(REPO_ROOT / p2["onnx_path"]) != p2["onnx_sha256"]:
            raise RuntimeError("OCR V10 P2 ONNX bytes changed")
        if (
            config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH)
            or config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH)
        ):
            raise RuntimeError("OCR V10 P3 split evidence changed")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        validation = build_split("validation")
        expected_summary = {
            key: selection["validation"][key] for key in proposal_summary(validation)
        }
        if (
            split_fingerprint(validation) != selection["validation"]["split_fingerprint"]
            or proposal_summary(validation) != expected_summary
        ):
            raise RuntimeError("OCR V10 P3 validation generator changed")
        phase = "parity"
        base = ComponentRecallNet(seed=int(config["seed"]))
        checkpoint = torch.load(REPO_ROOT / p2["checkpoint_path"], map_location="cpu", weights_only=True)
        base.load_state_dict(checkpoint["state_dict"])
        model = ScaledComponentRecallNet(base).eval()
        parity_values = _parity_values(validation)
        session = ort.InferenceSession(str(REPO_ROOT / p2["onnx_path"]), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(torch.from_numpy(parity_values)).numpy()
        actual = np.asarray(
            session.run(None, {"region_proposals": parity_values})[0], dtype=np.float32
        )
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])
        input_digest, output_digest, calls = sha256(), sha256(), 0

        def runner(input_values: np.ndarray) -> np.ndarray:
            nonlocal calls
            contiguous = np.ascontiguousarray(input_values, dtype=np.float32)
            input_digest.update(contiguous.tobytes(order="C"))
            output = np.asarray(
                session.run(None, {"region_proposals": contiguous})[0], dtype=np.float32
            )
            output_digest.update(np.ascontiguousarray(output).tobytes(order="C"))
            calls += 1
            return output

        phase = "selection"
        comparisons = evaluate_thresholds(
            validation, runner, tuple(float(value) for value in config["selection_thresholds"])
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
        truth_count = sum(len(scene.truths) for scene in validation)
        passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["true_positives"] == truth_count
            and metrics["false_positives"]
            == metrics["false_negatives"]
            == metrics["duplicate_region_count"]
            == metrics["prohibited_structure_hits"]
            == 0
            and parity_passed
            and calls == len(validation)
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-spaced-component-recall-candidate.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": "P3",
            "status": "selected" if passed else "failed_selection",
            "production_approval": False,
            "release_eligible": False,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "generalization_label_included": False,
            "optimizer_steps": 0,
            "weights_changed": False,
            "isolated_change": config["isolated_change"],
            "p2_result_sha256": config["p2_result_sha256"],
            "onnx_path": p2["onnx_path"],
            "onnx_sha256": p2["onnx_sha256"],
            "checkpoint_sha256": p2["checkpoint_sha256"],
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_passed": parity_passed,
            "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "selected_threshold": selected["threshold"],
            "selection_metrics": metrics,
            "selection_gate_passed": passed,
            "direct_execution": {
                "inference_calls": calls,
                "input_tensor_stream_sha256": input_digest.hexdigest(),
                "output_tensor_stream_sha256": output_digest.hexdigest(),
            },
            "sealed_public_archive_opened": False,
            "public_gate_evaluations": 0,
            "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(
            authorization, status=str(report["status"]), report_sha256=sha256_file(report_path)
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-spaced-component-recall-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": "P3",
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": 0,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False,
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(
            authorization, status="failed_runner", report_sha256=sha256_file(report_path)
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = evaluate_candidate(REPO_ROOT / args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_steps": report["optimizer_steps"],
                "selected_threshold": report["selected_threshold"],
                "selection_metrics": report["selection_metrics"],
                "onnx_parity_maximum_absolute_error": report[
                    "onnx_parity_maximum_absolute_error"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
