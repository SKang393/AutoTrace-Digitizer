# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer P3 calibration of the exact P2 payload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml.markers.center.line_aware_v1.dataset import build_selection_scenes, selection_manifest
from ml.markers.center.line_aware_v1.model_p2 import LineAwareP2ModelConfig, LineAwarePatchNetP2
from ml.markers.center.line_aware_v1.pipeline import evaluate_scenes, extract_proposals
from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-line-aware-v1"
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/markers/center/line_aware_v1/training/p3.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/line_aware_v1/dataset.py"),
    Path("ml/markers/center/line_aware_v1/model.py"),
    Path("ml/markers/center/line_aware_v1/model_p2.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/center/line_aware_v1/train_p3.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _canonical_hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def calibrate_candidate(output_dir: Path) -> dict[str, object]:
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
            raise RuntimeError("Selection manifest checksum does not match P3 preregistration")
        if _canonical_hash(selection_manifest()) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection renderer no longer reproduces the frozen manifest")
        sealed_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(sealed_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Sealed public test seal checksum does not match P3 preregistration")
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        fixture_archive = REPO_ROOT / sealed["fixture_archive_path"]
        if sha256_file(fixture_archive) != sealed["fixture_archive_sha256"]:
            raise RuntimeError("Sealed public fixture archive changed before P3 calibration")

        phase = "source_candidate_verification"
        source_report_path = REPO_ROOT / config["source_candidate_report_path"]
        checkpoint_path = REPO_ROOT / config["source_checkpoint_path"]
        onnx_path = REPO_ROOT / config["source_onnx_path"]
        expected_hashes = {
            source_report_path: config["source_candidate_report_sha256"],
            checkpoint_path: config["source_checkpoint_sha256"],
            onnx_path: config["source_onnx_sha256"],
        }
        for path, expected in expected_hashes.items():
            if sha256_file(path) != expected:
                raise RuntimeError(f"P3 source candidate checksum mismatch: {_relative(path)}")
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if (
            source_report.get("candidate_id") != "P2"
            or source_report.get("status") != "failed_selection"
            or source_report.get("public_gate_evaluations") != 0
            or source_report.get("sealed_public_archive_opened") is not False
            or source_report.get("onnx_parity_passed") is not True
            or source_report.get("checkpoint_sha256") != config["source_checkpoint_sha256"]
            or source_report.get("onnx_sha256") != config["source_onnx_sha256"]
        ):
            raise RuntimeError("P3 requires the exact parity-passing, selection-failed P2 report")
        onnx.checker.check_model(onnx.load(onnx_path))

        phase = "zero_optimizer_selection"
        validation_scenes = build_selection_scenes("validation")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("P3 calibration requires CPUExecutionProvider only")

        def runner(value: np.ndarray) -> np.ndarray:
            return session.run(None, {"candidate_patches": value})[0]

        threshold = float(config["selection_threshold"])
        metrics = evaluate_scenes(validation_scenes, runner, threshold=threshold)
        gate_passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == 0
            and metrics["false_negatives"] == 0
            and metrics["duplicate_count"] == 0
            and metrics["prohibited_structure_hits"] == 0
        )

        phase = "onnx_parity_recheck"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = LineAwarePatchNetP2(LineAwareP2ModelConfig(seed=int(config["source_seed"])))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        parity_patches = extract_proposals(validation_scenes[0].tensor).patches[:128]
        parity_input = parity_patches.numpy().astype(np.float32, copy=False)
        with torch.inference_mode():
            expected = model(parity_patches).numpy()
        actual = runner(parity_input)
        maximum_absolute_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_absolute_error <= float(config["onnx_parity_tolerance"])

        report: dict[str, object] = {
            "schema": "graphreader.marker-center-line-aware-training-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if gate_passed and parity_passed else "failed_selection",
            "release_eligible": False,
            "production_approval": False,
            "public_gate_evaluations": 0,
            "private_or_article_images": False,
            "synthetic_only": True,
            "chandler_included": False,
            "training_authorization": authorization.binding,
            "optimizer_steps": 0,
            "weights_changed": False,
            "source_candidate_id": "P2",
            "source_candidate_report_path": _relative(source_report_path),
            "source_candidate_report_sha256": sha256_file(source_report_path),
            "checkpoint_path": _relative(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": _relative(onnx_path),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_parity_maximum_absolute_error": maximum_absolute_error,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_passed,
            "threshold_comparisons": [{"threshold": threshold, "metrics": metrics}],
            "selected_threshold": threshold,
            "selection_gate_passed": gate_passed,
            "tensor_contract": model.export_contract(),
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "sealed_public_archive_opened": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(
            authorization,
            status=str(report["status"]),
            report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-line-aware-training-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "release_eligible": False,
            "production_approval": False,
            "public_gate_evaluations": 0,
            "optimizer_steps": 0,
            "weights_changed": False,
            "private_or_article_images": False,
            "synthetic_only": True,
            "chandler_included": False,
            "sealed_public_archive_opened": False,
            "phase": phase,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(
            authorization,
            status="failed_runner",
            report_sha256=sha256_file(report_path),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/markers/center/artifacts/line-aware-v1/P3-run"),
    )
    args = parser.parse_args()
    result = calibrate_candidate(REPO_ROOT / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
