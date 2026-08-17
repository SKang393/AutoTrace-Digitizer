# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P3 precision-weighted reflection training for marker-center V10."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import ml.markers.center.decoupled_heads_v10.train_p2 as p2
from ml.markers.center.decoupled_heads_v10.dataset import TRAIN_SCENE_COUNT
from ml.markers.center.decoupled_heads_v10.protocol import (
    ONNX_PARITY_TOLERANCE,
    REVISION,
    TASK,
    THRESHOLDS,
)
from ml.markers.center.decoupled_heads_v10.train_p1 import (
    ARTIFACT_BCE_FRACTION,
    ARTIFACT_LOSS_WEIGHT,
    BATCH_SIZE,
    CENTER_LOSS_WEIGHT,
    EPOCHS,
    HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
    LEARNING_RATE,
    MARKER_CLEAR_LOSS_WEIGHT,
    POSITIVE_MARGIN_LOSS_WEIGHT,
    _artifact_loss,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/decoupled_heads_v10"
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/markers/center/decoupled_heads_v10/training/p3.json")
P2_RESULT_PATH = Path("ml/markers/center/decoupled_heads_v10/P2_RESULT.json")
P2_RESULT_SHA256 = "d7e00a3890eb54bd55e5783eb748bee6b7812d5aee29718171e3121367c6f724"
ARTIFACT_FALSE_POSITIVE_WEIGHT = 0.95
ARTIFACT_FALSE_NEGATIVE_WEIGHT = 0.05
RUNNER_SOURCE_PATHS = (
    *p2.RUNNER_SOURCE_PATHS,
    Path("ml/markers/center/decoupled_heads_v10/P2_RESULT.json"),
    Path("ml/markers/center/decoupled_heads_v10/train_p3.py"),
)


def _precision_artifact_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Retain P2's loss form while changing only its Tversky precision balance."""

    probability = prediction.clamp(1e-6, 1.0 - 1e-6)
    truth = target.clamp(0.0, 1.0)
    axes = (1, 2, 3)
    true_positive = (probability * truth).sum(dim=axes)
    false_positive = (probability * (1.0 - truth)).sum(dim=axes)
    false_negative = ((1.0 - probability) * truth).sum(dim=axes)
    tversky = (true_positive + 1.0) / (
        true_positive
        + ARTIFACT_FALSE_POSITIVE_WEIGHT * false_positive
        + ARTIFACT_FALSE_NEGATIVE_WEIGHT * false_negative
        + 1.0
    )
    return ARTIFACT_BCE_FRACTION * _artifact_loss(probability, truth) + (
        1.0 - ARTIFACT_BCE_FRACTION
    ) * (1.0 - tversky.mean())


def _verify_config_and_inputs(
    config: dict[str, object],
) -> tuple[dict[str, object], Path, Path]:
    expected = {
        "artifact_bce_fraction": ARTIFACT_BCE_FRACTION,
        "artifact_false_negative_weight": ARTIFACT_FALSE_NEGATIVE_WEIGHT,
        "artifact_false_positive_weight": ARTIFACT_FALSE_POSITIVE_WEIGHT,
        "artifact_loss_weight": ARTIFACT_LOSS_WEIGHT,
        "augmentation_epoch_rule": "schedule[epoch_index_modulo_4]",
        "augmentation_interpolation": "none_exact_tensor_reflection",
        "augmentation_schedule": list(p2.REFLECTION_SCHEDULE),
        "batch_size": BATCH_SIZE,
        "center_loss_weight": CENTER_LOSS_WEIGHT,
        "epochs": EPOCHS,
        "expected_optimizer_steps": EPOCHS * (TRAIN_SCENE_COUNT // BATCH_SIZE),
        "hard_negative_margin_loss_weight": HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "marker_clear_loss_weight": MARKER_CLEAR_LOSS_WEIGHT,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "p2_checkpoint_reused": False,
        "p2_result_sha256": P2_RESULT_SHA256,
        "positive_margin_loss_weight": POSITIVE_MARGIN_LOSS_WEIGHT,
        "selection_thresholds": list(THRESHOLDS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V10 P3 configuration changed: {key}")
    p2_result_path = REPO_ROOT / P2_RESULT_PATH
    if sha256_file(p2_result_path) != P2_RESULT_SHA256:
        raise RuntimeError("Marker-center V10 P2 result changed")
    p2_result = json.loads(p2_result_path.read_text(encoding="utf-8"))
    if (
        p2_result.get("status") != "failed_selection_consumed"
        or p2_result.get("selection_exact_scene_count") != 121
        or p2_result.get("selection_false_positives") != 3
        or p2_result.get("selection_false_negatives") != 28
        or p2_result.get("selection_marker_artifact_hits") != 1
        or p2_result.get("public_gate_archive_opened") is not False
        or p2_result.get("case_detail_or_pixels_inspected") is not False
    ):
        raise RuntimeError("Marker-center V10 P3 requires the consumed aggregate P2 failure")
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    if sha256_file(selection_path) != config.get("selection_manifest_sha256"):
        raise RuntimeError("Marker-center V10 selection manifest changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    train_path = REPO_ROOT / str(selection["train"]["archive_path"])
    validation_path = REPO_ROOT / str(selection["validation"]["archive_path"])
    if sha256_file(train_path) != selection["train"]["archive_sha256"]:
        raise RuntimeError("Frozen marker-center V10 training archive changed")
    if sha256_file(validation_path) != selection["validation"]["archive_sha256"]:
        raise RuntimeError("Frozen marker-center V10 validation archive changed")
    return selection, train_path, validation_path


def _execute_candidate(
    output_dir: Path,
    authorization: TrainingAuthorization,
    progress: dict[str, object],
) -> dict[str, object]:
    p2.CANDIDATE_ID = CANDIDATE_ID
    p2.CONFIG_PATH = CONFIG_PATH
    p2._specificity_artifact_loss = _precision_artifact_loss
    p2._verify_config_and_inputs = _verify_config_and_inputs
    report = p2._execute_candidate(output_dir, authorization, progress)
    old_checkpoint = output_dir / "marker-center-decoupled-heads-v10-p2.pt"
    old_onnx = output_dir / "marker-center-decoupled-heads-v10-p2.onnx"
    checkpoint = output_dir / "marker-center-decoupled-heads-v10-p3.pt"
    onnx = output_dir / "marker-center-decoupled-heads-v10-p3.onnx"
    old_checkpoint.rename(checkpoint)
    old_onnx.rename(onnx)
    report["checkpoint_path"] = checkpoint.relative_to(REPO_ROOT).as_posix()
    report["checkpoint_sha256"] = sha256_file(checkpoint)
    report["onnx_path"] = onnx.relative_to(REPO_ROOT).as_posix()
    report["onnx_sha256"] = sha256_file(onnx)
    report["p2_checkpoint_reused"] = False
    report["p2_result_sha256"] = P2_RESULT_SHA256
    report["artifact_false_positive_weight"] = ARTIFACT_FALSE_POSITIVE_WEIGHT
    report["artifact_false_negative_weight"] = ARTIFACT_FALSE_NEGATIVE_WEIGHT
    report.pop("p1_checkpoint_reused", None)
    report.pop("p1_result_sha256", None)
    return report


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Marker-center V10 P3 output exists: {output_dir}")
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
    progress: dict[str, object] = {
        "started": p2.time.perf_counter(),
        "phase": "initialization",
        "optimizer_steps": 0,
    }
    try:
        report = _execute_candidate(output_dir, authorization, progress)
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-decoupled-heads-failure.v10",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": progress["phase"],
            "optimizer_steps": progress["optimizer_steps"],
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "production_approval": False,
            "release_eligible": False,
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(
            authorization, status="failed_runner", report_sha256=sha256_file(report_path)
        )
        raise
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(
        authorization, status=str(report["status"]), report_sha256=sha256_file(report_path)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "ml/markers/center/artifacts/decoupled-heads-v10/P3-run",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
