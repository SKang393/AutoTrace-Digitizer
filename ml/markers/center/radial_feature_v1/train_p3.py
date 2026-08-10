# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P3 selection for local geometry-consensus center refinement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.center.line_aware_v1.pipeline import extract_proposals
from ml.markers.center.radial_feature_v1.dataset import build_selection_scenes, selection_manifest
from ml.markers.center.radial_feature_v1.model import RadialFeatureModelConfig, RadialFeatureNet
from ml.markers.center.radial_feature_v1.pipeline_p3 import evaluate_scenes
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-radial-feature-v1"
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/markers/center/radial_feature_v1/training/p3.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/radial_feature_v1/dataset.py"),
    Path("ml/markers/center/radial_feature_v1/model.py"),
    Path("ml/markers/center/radial_feature_v1/pipeline_p3.py"),
    Path("ml/markers/center/radial_feature_v1/train_p3.py"),
    Path("ml/markers/center/line_aware_v1/dataset.py"),
    Path("ml/markers/center/line_aware_v1/model.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def select_candidate(output_dir: Path) -> dict[str, object]:
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
    try:
        selection_path = REPO_ROOT / config["selection_manifest_path"]
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection manifest checksum does not match preregistration")
        if _canonical_hash(selection_manifest()) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection renderer no longer reproduces the frozen manifest")
        sealed_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(sealed_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Sealed public test seal checksum does not match preregistration")
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        archive = REPO_ROOT / sealed["fixture_archive_path"]
        if sha256_file(archive) != sealed["fixture_archive_sha256"]:
            raise RuntimeError("Sealed public archive changed before selection")

        p1_report_path = REPO_ROOT / config["p1_training_report_path"]
        p1_checkpoint_path = REPO_ROOT / config["p1_checkpoint_path"]
        p1_onnx_path = REPO_ROOT / config["p1_onnx_path"]
        for path, expected in (
            (p1_report_path, config["p1_training_report_sha256"]),
            (p1_checkpoint_path, config["p1_checkpoint_sha256"]),
            (p1_onnx_path, config["p1_onnx_sha256"]),
        ):
            if sha256_file(path) != expected:
                raise RuntimeError(f"P1 reuse input checksum mismatch: {path}")
        p1_report = json.loads(p1_report_path.read_text(encoding="utf-8"))
        if (
            p1_report.get("candidate_id") != "P1"
            or p1_report.get("status") != "failed_selection"
            or p1_report.get("sealed_public_archive_opened") is not False
        ):
            raise RuntimeError("P1 reuse report does not describe the consumed selection failure")

        phase = "selection"
        session = ort.InferenceSession(str(p1_onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        def run(value: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: value.astype(np.float32, copy=False)})[0]

        validation_scenes = build_selection_scenes("validation")
        threshold_results = [
            {"threshold": float(threshold), "metrics": evaluate_scenes(validation_scenes, run, threshold=float(threshold))}
            for threshold in config["selection_thresholds"]
        ]
        selected = max(threshold_results, key=lambda item: (
            item["metrics"]["exact_scene_count"],
            -item["metrics"]["false_positives"],
            item["metrics"]["f1"],
            item["threshold"],
        ))
        metrics = selected["metrics"]
        selection_passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == 0
            and metrics["false_negatives"] == 0
            and metrics["duplicate_count"] == 0
            and metrics["prohibited_structure_hits"] == 0
        )

        phase = "parity"
        checkpoint = torch.load(p1_checkpoint_path, map_location="cpu", weights_only=False)
        model = RadialFeatureNet(RadialFeatureModelConfig(seed=20261001)).eval()
        model.load_state_dict(checkpoint["state_dict"])
        parity_patches = extract_proposals(validation_scenes[0].tensor).patches[:128]
        parity_input = parity_patches.numpy().astype(np.float32, copy=False)
        with torch.inference_mode():
            expected = model(parity_patches).numpy()
        actual = run(parity_input)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])

        checkpoint_path = output_dir / "marker-center-radial-feature-p3.pt"
        onnx_path = output_dir / "marker-center-radial-feature-p3.onnx"
        shutil.copyfile(p1_checkpoint_path, checkpoint_path)
        shutil.copyfile(p1_onnx_path, onnx_path)
        if sha256_file(checkpoint_path) != config["p1_checkpoint_sha256"]:
            raise RuntimeError("P3 checkpoint copy changed P1 bytes")
        if sha256_file(onnx_path) != config["p1_onnx_sha256"]:
            raise RuntimeError("P3 ONNX copy changed P1 bytes")

        report: dict[str, object] = {
            "schema": "graphreader.marker-center-radial-feature-training-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if selection_passed and parity_passed else "failed_selection",
            "release_eligible": False,
            "production_approval": False,
            "public_gate_evaluations": 0,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "training_authorization": authorization.binding,
            "optimizer_steps": 0,
            "weights_changed": False,
            "reused_candidate_id": "P1",
            "postprocess_revision": "radial-local-consensus-refinement-v1",
            "refinement_offsets_pixels": [-1.0, 0.0, 1.0],
            "threshold_comparisons": threshold_results,
            "selected_threshold": selected["threshold"],
            "selection_gate_passed": selection_passed,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_passed,
            "tensor_contract": model.export_contract(),
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "sealed_public_archive_opened": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-radial-feature-training-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "release_eligible": False,
            "production_approval": False,
            "public_gate_evaluations": 0,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "sealed_public_archive_opened": False,
            "phase": phase,
            "optimizer_steps": 0,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ml/markers/center/artifacts/radial-feature-v1/P3-run"))
    args = parser.parse_args()
    result = select_candidate(REPO_ROOT / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
