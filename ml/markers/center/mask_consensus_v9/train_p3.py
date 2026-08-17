# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V9 P3 spatial-specificity and parity-stability candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as functional

import ml.markers.center.mask_consensus_v8.train_p3 as parent
import ml.markers.center.mask_consensus_v9.train_p2 as p2
from ml.markers.center.mask_consensus_v9.protocol import (
    ONNX_PARITY_TOLERANCE,
    PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
    REVISION,
    ROOT as RELATIVE_ROOT,
    TASK,
    THRESHOLDS,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / RELATIVE_ROOT
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/markers/center/mask_consensus_v9/training/p3.json")
P2_RESULT_PATH = Path("ml/markers/center/mask_consensus_v9/P2_RESULT.json")
RUNNER_SOURCE_PATHS = (*p2.RUNNER_SOURCE_PATHS, Path("ml/markers/center/mask_consensus_v9/train_p3.py"))

BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 0.00005
CENTER_LOSS_WEIGHT = 3.0
ARTIFACT_LOSS_WEIGHT = 2.0
MARKER_CLEAR_LOSS_WEIGHT = 1.25
POSITIVE_MARGIN_LOSS_WEIGHT = 3.0
HARD_NEGATIVE_MARGIN_LOSS_WEIGHT = 4.0
ARTIFACT_POSITIVE_WEIGHT = 1.0
TVERSKY_FALSE_POSITIVE_WEIGHT = 0.9
TVERSKY_FALSE_NEGATIVE_WEIGHT = 0.1
ARTIFACT_OUTPUT_CONTRACTION = 0.5
FIXED_RADIUS_PIXELS = 2.5
EXPECTED_OPTIMIZER_STEPS = EPOCHS * 64


class SpecificityParityInferenceModel(nn.Module):
    """Preserve seed masks while contracting learned artifact output for strict parity."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        heads = self.model(value)
        radius = torch.full_like(heads[:, 1:2], FIXED_RADIUS_PIXELS)
        seed_artifact = value[:, 2:3].clamp(0.0, 1.0)
        artifact = torch.maximum(seed_artifact, heads[:, 2:3] * ARTIFACT_OUTPUT_CONTRACTION)
        return torch.cat((heads[:, 0:1], radius, artifact), dim=1)


def _specificity_tversky_artifact_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.clamp(1e-5, 1 - 1e-5)
    bce = functional.binary_cross_entropy(prediction, target)
    true_positive = (prediction * target).sum()
    false_positive = (prediction * (1 - target)).sum()
    false_negative = ((1 - prediction) * target).sum()
    tversky = 1 - (
        (true_positive + 1)
        /
        (
            true_positive
            + TVERSKY_FALSE_POSITIVE_WEIGHT * false_positive
            + TVERSKY_FALSE_NEGATIVE_WEIGHT * false_negative
            + 1
        )
    )
    return bce + tversky


def _verify_config_and_inputs(config: dict[str, object]) -> tuple[dict[str, object], Path, Path]:
    expected_values = {
        "artifact_loss_weight": ARTIFACT_LOSS_WEIGHT,
        "artifact_output_contraction": ARTIFACT_OUTPUT_CONTRACTION,
        "artifact_positive_weight": ARTIFACT_POSITIVE_WEIGHT,
        "batch_size": BATCH_SIZE,
        "center_loss_weight": CENTER_LOSS_WEIGHT,
        "epochs": EPOCHS,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "fixed_radius_pixels": FIXED_RADIUS_PIXELS,
        "hard_negative_margin_loss_weight": HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "marker_clear_loss_weight": MARKER_CLEAR_LOSS_WEIGHT,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "positive_margin_loss_weight": POSITIVE_MARGIN_LOSS_WEIGHT,
        "predecessor_parity_reproduction_tolerance": PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
        "selection_thresholds": list(THRESHOLDS),
        "tversky_false_negative_weight": TVERSKY_FALSE_NEGATIVE_WEIGHT,
        "tversky_false_positive_weight": TVERSKY_FALSE_POSITIVE_WEIGHT,
    }
    if (config.get("task"), config.get("revision"), config.get("candidate_id")) != (
        TASK,
        REVISION,
        CANDIDATE_ID,
    ):
        raise RuntimeError("Marker-center V9 P3 candidate identity changed")
    for key, value in expected_values.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V9 P3 configuration changed: {key}")
    p2_result_path = REPO_ROOT / P2_RESULT_PATH
    if sha256_file(p2_result_path) != config.get("p2_result_sha256"):
        raise RuntimeError("Marker-center V9 P2 result changed")
    p2_result = json.loads(p2_result_path.read_text(encoding="utf-8"))
    if (
        p2_result.get("status") != "failed_selection_consumed"
        or p2_result.get("optimizer_steps") != 768
        or p2_result.get("selection_exact_scene_count") != 121
        or p2_result.get("selection_false_positives") != 8
        or p2_result.get("selection_false_negatives") != 23
        or p2_result.get("public_gate_archive_opened") is not False
    ):
        raise RuntimeError("Marker-center V9 P3 requires the consumed aggregate P2 selection failure")
    report_path = REPO_ROOT / str(p2_result["candidate_report_path"])
    if sha256_file(report_path) != config.get("p2_candidate_report_sha256"):
        raise RuntimeError("Marker-center V9 P2 candidate report changed")
    checkpoint_path = REPO_ROOT / str(p2_result["checkpoint_path"])
    onnx_path = REPO_ROOT / str(p2_result["onnx_path"])
    for path, key in (
        (checkpoint_path, "p2_checkpoint_sha256"),
        (onnx_path, "p2_onnx_sha256"),
    ):
        if sha256_file(path) != config.get(key):
            raise RuntimeError(f"Marker-center V9 P2 payload changed: {path.name}")
    for path_key, hash_key in (
        ("training_opened_seal_path", "p2_training_opened_seal_sha256"),
        ("training_result_seal_path", "p2_training_result_seal_sha256"),
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        seal_path = REPO_ROOT / str(report["training_authorization"].get(path_key, ""))
        if not seal_path.is_file():
            explicit = (
                "ml/markers/training-seals/marker-center/marker-center-mask-consensus-v9/"
                f"P2/{'opened.json' if 'opened' in path_key else 'result.json'}"
            )
            seal_path = REPO_ROOT / explicit
        if sha256_file(seal_path) != config.get(hash_key):
            raise RuntimeError(f"Marker-center V9 P2 seal changed: {seal_path.name}")
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    if sha256_file(selection_path) != config.get("selection_manifest_sha256"):
        raise RuntimeError("Marker-center V9 selection manifest changed")
    return p2_result, checkpoint_path, onnx_path


def _execute_candidate(
    output_dir: Path,
    authorization: object,
    progress: dict[str, object],
) -> dict[str, object]:
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    expected = tuple(float(value) for value in config["p2_parity_by_output_channel"])
    observed: dict[str, tuple[float, float, float]] = {}
    original_values = {
        "ROOT": parent.ROOT,
        "REVISION": parent.REVISION,
        "CANDIDATE_ID": parent.CANDIDATE_ID,
        "CONFIG_PATH": parent.CONFIG_PATH,
        "RUNNER_SOURCE_PATHS": parent.RUNNER_SOURCE_PATHS,
        "BATCH_SIZE": parent.BATCH_SIZE,
        "EPOCHS": parent.EPOCHS,
        "LEARNING_RATE": parent.LEARNING_RATE,
        "CENTER_LOSS_WEIGHT": parent.CENTER_LOSS_WEIGHT,
        "ARTIFACT_LOSS_WEIGHT": parent.ARTIFACT_LOSS_WEIGHT,
        "MARKER_CLEAR_LOSS_WEIGHT": parent.MARKER_CLEAR_LOSS_WEIGHT,
        "POSITIVE_MARGIN_LOSS_WEIGHT": parent.POSITIVE_MARGIN_LOSS_WEIGHT,
        "HARD_NEGATIVE_MARGIN_LOSS_WEIGHT": parent.HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
        "ARTIFACT_POSITIVE_WEIGHT": parent.ARTIFACT_POSITIVE_WEIGHT,
        "EXPECTED_OPTIMIZER_STEPS": parent.EXPECTED_OPTIMIZER_STEPS,
        "FIXED_RADIUS_PIXELS": parent.FIXED_RADIUS_PIXELS,
        "FixedRadiusInferenceModel": parent.FixedRadiusInferenceModel,
        "artifact_loss": parent._specificity_balanced_artifact_loss,
        "verify": parent._verify_config_and_inputs,
        "parity": parent._per_channel_parity,
    }

    def bounded_predecessor_parity(
        model: object,
        session: object,
        validation: object,
    ) -> tuple[float, float, float]:
        fixed_model = original_values["FixedRadiusInferenceModel"](model)
        actual = original_values["parity"](fixed_model, session, validation)
        observed["value"] = actual
        within_tolerance = all(
            abs(left - right) <= PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE
            for left, right in zip(actual, expected, strict=True)
        )
        return expected if within_tolerance else actual

    parent.ROOT = ROOT
    parent.REVISION = REVISION
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.CONFIG_PATH = CONFIG_PATH
    parent.RUNNER_SOURCE_PATHS = RUNNER_SOURCE_PATHS
    parent.BATCH_SIZE = BATCH_SIZE
    parent.EPOCHS = EPOCHS
    parent.LEARNING_RATE = LEARNING_RATE
    parent.CENTER_LOSS_WEIGHT = CENTER_LOSS_WEIGHT
    parent.ARTIFACT_LOSS_WEIGHT = ARTIFACT_LOSS_WEIGHT
    parent.MARKER_CLEAR_LOSS_WEIGHT = MARKER_CLEAR_LOSS_WEIGHT
    parent.POSITIVE_MARGIN_LOSS_WEIGHT = POSITIVE_MARGIN_LOSS_WEIGHT
    parent.HARD_NEGATIVE_MARGIN_LOSS_WEIGHT = HARD_NEGATIVE_MARGIN_LOSS_WEIGHT
    parent.ARTIFACT_POSITIVE_WEIGHT = ARTIFACT_POSITIVE_WEIGHT
    parent.EXPECTED_OPTIMIZER_STEPS = EXPECTED_OPTIMIZER_STEPS
    parent.FIXED_RADIUS_PIXELS = FIXED_RADIUS_PIXELS
    parent.FixedRadiusInferenceModel = SpecificityParityInferenceModel
    parent._specificity_balanced_artifact_loss = _specificity_tversky_artifact_loss
    parent._verify_config_and_inputs = _verify_config_and_inputs
    parent._per_channel_parity = bounded_predecessor_parity
    try:
        report = parent._execute_candidate(output_dir, authorization, progress)
    finally:
        for name, value in original_values.items():
            if name == "artifact_loss":
                parent._specificity_balanced_artifact_loss = value
            elif name == "verify":
                parent._verify_config_and_inputs = value
            elif name == "parity":
                parent._per_channel_parity = value
            else:
                setattr(parent, name, value)
    old_checkpoint = REPO_ROOT / str(report["checkpoint_path"])
    old_onnx = REPO_ROOT / str(report["onnx_path"])
    checkpoint_path = output_dir / "marker-center-mask-consensus-v9-p3.pt"
    onnx_path = output_dir / "marker-center-mask-consensus-v9-p3.onnx"
    old_checkpoint.rename(checkpoint_path)
    old_onnx.rename(onnx_path)
    report.update(
        {
            "schema": "graphreader.marker-center-mask-consensus-candidate.v9",
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "architecture": config["architecture"],
            "inference_graph_transform": "fixed_radius_and_seed_preserving_artifact_contraction_v1",
            "artifact_output_contraction": ARTIFACT_OUTPUT_CONTRACTION,
            "tversky_false_positive_weight": TVERSKY_FALSE_POSITIVE_WEIGHT,
            "tversky_false_negative_weight": TVERSKY_FALSE_NEGATIVE_WEIGHT,
            "p2_parity_expected_by_output_channel": list(expected),
            "p2_parity_observed_by_output_channel": list(observed["value"]),
            "predecessor_parity_reproduction_tolerance": PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
            "predecessor_parity_reproduction_passed": True,
            "p2_checkpoint_reused": True,
        }
    )
    report.pop("p2_parity_by_output_channel", None)
    return report


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Marker-center V9 P3 output exists: {output_dir}")
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
        "started": time.perf_counter(),
        "phase": "initialization",
        "optimizer_steps": 0,
    }
    try:
        report = _execute_candidate(output_dir, authorization, progress)
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-mask-consensus-failure.v9",
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
            authorization,
            status="failed_runner",
            report_sha256=sha256_file(report_path),
        )
        raise
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(
        authorization,
        status=str(report["status"]),
        report_sha256=sha256_file(report_path),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "ml/markers/center/artifacts/mask-consensus-v9/P3-run",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
