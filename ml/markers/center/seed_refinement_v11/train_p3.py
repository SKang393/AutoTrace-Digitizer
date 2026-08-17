# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P3 training and visible selection for marker-center V11."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as functional

from ml.markers.center.dense_contract_v5.train_p1 import (
    _center_focal_loss,
    _configure_determinism,
    _export,
    _passing_window,
)
from ml.markers.center.dense_contract_v5.train_p3 import _fuse_inference_model
from ml.markers.center.decoupled_heads_v10.train_p2 import (
    REFLECTION_SCHEDULE,
    _reflect_tensor,
    _reflected_spatial_margin_losses,
)
from ml.markers.center.seed_refinement_v11.dataset import TRAIN_SCENE_COUNT, read_archive
from ml.markers.center.seed_refinement_v11.model import create_model, save_checkpoint
from ml.markers.center.seed_refinement_v11.protocol import (
    ARTIFACT_THRESHOLD,
    MINIMUM_ARTIFACT_PRECISION,
    MINIMUM_ARTIFACT_RECALL,
    MINIMUM_PASSING_THRESHOLD_COUNT,
    ONNX_PARITY_TOLERANCE,
    REVISION,
    TASK,
    THRESHOLDS,
)
from ml.markers.center.seed_refinement_v11.train_p1 import (
    ARTIFACT_BCE_FRACTION,
    ARTIFACT_FALSE_NEGATIVE_WEIGHT,
    ARTIFACT_FALSE_POSITIVE_WEIGHT,
    ARTIFACT_LOSS_WEIGHT,
    BATCH_SIZE,
    CENTER_LOSS_WEIGHT,
    EPOCHS,
    FIXED_RADIUS_PIXELS,
    HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
    LEARNING_RATE,
    MARKER_CLEAR_LOSS_WEIGHT,
    MAXIMUM_LOGIT_CORRECTION,
    POSITIVE_MARGIN_LOSS_WEIGHT,
    _evaluate,
    _onnx_output,
    _specificity_artifact_loss,
    _torch_output,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/seed_refinement_v11"
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/markers/center/seed_refinement_v11/training/p3.json")
P2_RESULT_PATH = Path("ml/markers/center/seed_refinement_v11/P2_RESULT.json")
P2_RESULT_SHA256 = "563197cffbf5f86c6dd866e327ccd816967229f4dde5b3de7888b239e64d64a3"
UNSUPPORTED_SEED_ADDITION_LOSS_WEIGHT = 3.0
FALSE_SEED_RETENTION_LOSS_WEIGHT = 6.0
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/markers/center/model.py"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/markers/center/postprocess.py"),
    Path("ml/markers/center/dense_contract_v5/dataset.py"),
    Path("ml/markers/center/dense_contract_v5/train_p1.py"),
    Path("ml/markers/center/dense_contract_v5/train_p3.py"),
    Path("ml/markers/center/feasible_dense_v6/dataset.py"),
    Path("ml/markers/center/mask_consensus_v8/dataset.py"),
    Path("ml/markers/center/decoupled_heads_v10/AGGREGATE_FEASIBILITY.json"),
    Path("ml/markers/center/decoupled_heads_v10/train_p2.py"),
    Path("ml/markers/center/seed_refinement_v11/dataset.py"),
    Path("ml/markers/center/seed_refinement_v11/model.py"),
    Path("ml/markers/center/seed_refinement_v11/protocol.py"),
    Path("ml/markers/center/seed_refinement_v11/train_p1.py"),
    Path("ml/markers/center/seed_refinement_v11/P1_RESULT.json"),
    P2_RESULT_PATH,
    Path("ml/markers/center/seed_refinement_v11/train_p2.py"),
    Path("ml/markers/center/seed_refinement_v11/train_p3.py"),
)


def _unsupported_seed_addition_loss(
    artifact_prediction: torch.Tensor,
    seed_artifact: torch.Tensor,
    artifact_target: torch.Tensor,
) -> torch.Tensor:
    """Penalize only predicted additions that both seed and truth reject."""

    unsupported = (seed_artifact < ARTIFACT_THRESHOLD) & (artifact_target < 0.5)
    if not bool(unsupported.any()):
        return artifact_prediction.sum() * 0.0
    probability = artifact_prediction[unsupported].clamp(1e-6, 1.0 - 1e-6)
    return functional.binary_cross_entropy(probability, torch.zeros_like(probability))


def _false_seed_retention_loss(
    artifact_prediction: torch.Tensor,
    seed_artifact: torch.Tensor,
    artifact_target: torch.Tensor,
) -> torch.Tensor:
    """Penalize retained seed pixels that artifact truth rejects."""

    unsupported = (seed_artifact >= ARTIFACT_THRESHOLD) & (artifact_target < 0.5)
    if not bool(unsupported.any()):
        return artifact_prediction.sum() * 0.0
    probability = artifact_prediction[unsupported].clamp(1e-6, 1.0 - 1e-6)
    return functional.binary_cross_entropy(probability, torch.zeros_like(probability))


def _verify_config_and_inputs(
    config: dict[str, object],
) -> tuple[dict[str, object], Path, Path]:
    expected = {
        "artifact_bce_fraction": ARTIFACT_BCE_FRACTION,
        "artifact_false_negative_weight": ARTIFACT_FALSE_NEGATIVE_WEIGHT,
        "artifact_false_positive_weight": ARTIFACT_FALSE_POSITIVE_WEIGHT,
        "artifact_loss_weight": ARTIFACT_LOSS_WEIGHT,
        "artifact_threshold": ARTIFACT_THRESHOLD,
        "augmentation_epoch_rule": "schedule[epoch_index_modulo_4]",
        "augmentation_interpolation": "none_exact_tensor_reflection",
        "augmentation_schedule": list(REFLECTION_SCHEDULE),
        "batch_size": BATCH_SIZE,
        "center_loss_weight": CENTER_LOSS_WEIGHT,
        "epochs": EPOCHS,
        "expected_optimizer_steps": EPOCHS * (TRAIN_SCENE_COUNT // BATCH_SIZE),
        "fixed_radius_pixels": FIXED_RADIUS_PIXELS,
        "hard_negative_margin_loss_weight": HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "marker_clear_loss_weight": MARKER_CLEAR_LOSS_WEIGHT,
        "maximum_logit_correction": MAXIMUM_LOGIT_CORRECTION,
        "minimum_artifact_precision": MINIMUM_ARTIFACT_PRECISION,
        "minimum_artifact_recall": MINIMUM_ARTIFACT_RECALL,
        "minimum_consecutive_passing_thresholds": MINIMUM_PASSING_THRESHOLD_COUNT,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "p2_checkpoint_reused": False,
        "p2_result_sha256": P2_RESULT_SHA256,
        "positive_margin_loss_weight": POSITIVE_MARGIN_LOSS_WEIGHT,
        "runtime_pass_count": 1,
        "runtime_postprocess_profile": "nonmonotonic_seed_refinement_v1",
        "selection_thresholds": list(THRESHOLDS),
        "unsupported_seed_addition_definition": "binary_cross_entropy(refined_artifact[seed_artifact<0.35 and artifact_truth<0.5], zero)",
        "unsupported_seed_addition_loss_weight": UNSUPPORTED_SEED_ADDITION_LOSS_WEIGHT,
        "false_seed_retention_definition": "binary_cross_entropy(refined_artifact[seed_artifact>=0.35 and artifact_truth<0.5], zero)",
        "false_seed_retention_loss_weight": FALSE_SEED_RETENTION_LOSS_WEIGHT,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V11 P3 configuration changed: {key}")
    p2_result_path = REPO_ROOT / P2_RESULT_PATH
    if sha256_file(p2_result_path) != P2_RESULT_SHA256:
        raise RuntimeError("Marker-center V11 P2 aggregate result changed")
    p2_result = json.loads(p2_result_path.read_text(encoding="utf-8"))
    if (
        p2_result.get("status") != "failed_selection_consumed"
        or p2_result.get("selection_exact_scene_count") != 122
        or p2_result.get("selection_false_positives") != 0
        or p2_result.get("selection_false_negatives") != 29
        or p2_result.get("artifact_precision") != 0.782828150056521
        or p2_result.get("artifact_recall") != 0.9603460905861702
        or p2_result.get("seed_added_pixels") != 244692
        or p2_result.get("seed_removed_pixels") != 1253
        or p2_result.get("public_gate_archive_opened") is not False
        or p2_result.get("case_detail_or_pixels_inspected") is not False
    ):
        raise RuntimeError("Marker-center V11 P3 requires the consumed aggregate P2 failure")
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    if sha256_file(selection_path) != config.get("selection_manifest_sha256"):
        raise RuntimeError("Marker-center V11 selection manifest changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    train_path = REPO_ROOT / str(selection["train"]["archive_path"])
    validation_path = REPO_ROOT / str(selection["validation"]["archive_path"])
    if sha256_file(train_path) != selection["train"]["archive_sha256"]:
        raise RuntimeError("Frozen marker-center V11 training archive changed")
    if sha256_file(validation_path) != selection["validation"]["archive_sha256"]:
        raise RuntimeError("Frozen marker-center V11 validation archive changed")
    return selection, train_path, validation_path


def _execute_candidate(
    output_dir: Path,
    authorization: TrainingAuthorization,
    progress: dict[str, object],
) -> dict[str, object]:
    started = float(progress["started"])
    progress["phase"] = "input_validation"
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    selection, train_path, validation_path = _verify_config_and_inputs(config)
    train = read_archive(train_path)
    validation = read_archive(validation_path)
    _configure_determinism(int(config["seed"]))
    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    indices = np.arange(train["inputs"].shape[0])
    optimizer_steps = 0
    loss_checkpoints: list[dict[str, float | int]] = []
    reflection_epoch_counts = {name: 0 for name in REFLECTION_SCHEDULE}
    progress["phase"] = "training"
    model.train()
    for epoch in range(EPOCHS):
        transform_index = epoch % len(REFLECTION_SCHEDULE)
        reflection_epoch_counts[REFLECTION_SCHEDULE[transform_index]] += 1
        epoch_indices = np.roll(indices, (epoch * 29) % len(indices))
        for start in range(0, len(epoch_indices), BATCH_SIZE):
            batch_indices = epoch_indices[start : start + BATCH_SIZE]
            inputs = _reflect_tensor(
                torch.from_numpy(train["inputs"][batch_indices]), transform_index
            )
            center_target = _reflect_tensor(
                torch.from_numpy(train["center_targets"][batch_indices]), transform_index
            )
            artifact_target = _reflect_tensor(
                torch.from_numpy(train["artifact_targets"][batch_indices]), transform_index
            )
            optimizer.zero_grad(set_to_none=True)
            heads = model(inputs)
            center_loss = _center_focal_loss(heads[:, 0:1], center_target)
            artifact_loss = _specificity_artifact_loss(heads[:, 2:3], artifact_target)
            unsupported_addition_loss = _unsupported_seed_addition_loss(
                heads[:, 2:3], inputs[:, 2:3], artifact_target
            )
            false_seed_retention_loss = _false_seed_retention_loss(
                heads[:, 2:3], inputs[:, 2:3], artifact_target
            )
            marker_pixels = center_target >= 0.1
            marker_clear_loss = functional.binary_cross_entropy(
                heads[:, 2:3][marker_pixels],
                torch.zeros_like(heads[:, 2:3][marker_pixels]),
            )
            positive_margin_loss, hard_negative_margin_loss = _reflected_spatial_margin_losses(
                heads[:, 0:1], train, batch_indices, transform_index
            )
            loss = (
                CENTER_LOSS_WEIGHT * center_loss
                + ARTIFACT_LOSS_WEIGHT * artifact_loss
                + MARKER_CLEAR_LOSS_WEIGHT * marker_clear_loss
                + POSITIVE_MARGIN_LOSS_WEIGHT * positive_margin_loss
                + HARD_NEGATIVE_MARGIN_LOSS_WEIGHT * hard_negative_margin_loss
                + UNSUPPORTED_SEED_ADDITION_LOSS_WEIGHT * unsupported_addition_loss
                + FALSE_SEED_RETENTION_LOSS_WEIGHT * false_seed_retention_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            progress["optimizer_steps"] = optimizer_steps
        if epoch in (0, EPOCHS // 2, EPOCHS - 1):
            loss_checkpoints.append(
                {
                    "epoch": epoch + 1,
                    "total": float(loss.detach()),
                    "center": float(center_loss.detach()),
                    "artifact": float(artifact_loss.detach()),
                    "unsupported_seed_addition": float(unsupported_addition_loss.detach()),
                    "false_seed_retention": float(false_seed_retention_loss.detach()),
                    "marker_clear": float(marker_clear_loss.detach()),
                    "positive_margin": float(positive_margin_loss.detach()),
                    "hard_negative_margin": float(hard_negative_margin_loss.detach()),
                }
            )
    if optimizer_steps != int(config["expected_optimizer_steps"]):
        raise RuntimeError("Marker-center V11 P3 optimizer step count changed")
    if set(reflection_epoch_counts.values()) != {EPOCHS // len(REFLECTION_SCHEDULE)}:
        raise RuntimeError("Marker-center V11 reflection schedule did not complete evenly")
    model.eval()
    progress["phase"] = "export"
    checkpoint_path = output_dir / "marker-center-seed-refinement-v11-p3.pt"
    onnx_path = output_dir / "marker-center-seed-refinement-v11-p3.onnx"
    inference_model = _fuse_inference_model(model)
    sample = torch.from_numpy(validation["inputs"][0:1])
    _export(inference_model, sample, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Marker-center V11 P3 selection requires CPUExecutionProvider only")
    progress["phase"] = "selection"
    onnx_outputs: list[np.ndarray] = []
    parity = 0.0
    inference_semantic_error = 0.0
    input_stream = hashlib.sha256()
    output_stream = hashlib.sha256()
    for index in range(validation["inputs"].shape[0]):
        value = validation["inputs"][index : index + 1]
        raw_output = _torch_output(model, value)
        frozen_output = _torch_output(inference_model, value)
        onnx_value = _onnx_output(session, value)
        inference_semantic_error = max(
            inference_semantic_error,
            float(np.max(np.abs(raw_output - frozen_output))),
        )
        parity = max(parity, float(np.max(np.abs(frozen_output - onnx_value))))
        onnx_outputs.append(onnx_value)
        input_stream.update(value.tobytes(order="C"))
        output_stream.update(onnx_value.tobytes(order="C"))
    comparisons = [_evaluate(validation, onnx_outputs, threshold) for threshold in THRESHOLDS]
    window = _passing_window(comparisons)
    selected = max(
        comparisons,
        key=lambda item: (
            bool(item["passed"]),
            int(item["exact_scene_count"]),
            float(item["artifact_precision"]),
            float(item["artifact_recall"]),
            -abs(float(item["threshold"]) - 0.45),
        ),
    )
    selection_passed = (
        len(window) >= MINIMUM_PASSING_THRESHOLD_COUNT
        and parity <= ONNX_PARITY_TOLERANCE
    )
    save_checkpoint(
        checkpoint_path,
        model,
        selected_threshold=float(selected["threshold"]),
        dataset_manifest_sha256=selection["train"]["archive_sha256"],
        training_revision=REVISION,
    )
    return {
        "schema": "graphreader.marker-center-seed-refinement-candidate.v11",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "selection_passed" if selection_passed else "failed_selection",
        "selection_gate_passed": selection_passed,
        "isolated_change": config["isolated_change"],
        "private_data": False,
        "chandler_used": False,
        "synthetic_only": True,
        "provider": "CPUExecutionProvider",
        "runtime_pass_count": 1,
        "runtime_postprocess_profile": "nonmonotonic_seed_refinement_v1",
        "fixed_radius_pixels": FIXED_RADIUS_PIXELS,
        "maximum_logit_correction": MAXIMUM_LOGIT_CORRECTION,
        "unsupported_seed_addition_loss_weight": UNSUPPORTED_SEED_ADDITION_LOSS_WEIGHT,
        "false_seed_retention_loss_weight": FALSE_SEED_RETENTION_LOSS_WEIGHT,
        "optimizer_steps": optimizer_steps,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "inference_graph_transform": "deepcopy_eval_no_batchnorm_v1",
        "checkpoint_to_inference_graph_maximum_absolute_error": inference_semantic_error,
        "onnx_parity_maximum_absolute_error": parity,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "onnx_parity_passed": parity <= ONNX_PARITY_TOLERANCE,
        "selected_threshold": selected["threshold"],
        "selection_metrics": selected,
        "passing_threshold_window": window,
        "threshold_aggregates": comparisons,
        "direct_execution_inference_calls": validation["inputs"].shape[0],
        "input_tensor_stream_sha256": input_stream.hexdigest(),
        "output_tensor_stream_sha256": output_stream.hexdigest(),
        "loss_checkpoints": loss_checkpoints,
        "reflection_epoch_counts": reflection_epoch_counts,
        "training_authorization": authorization.binding,
        "public_gate_archive_opened": False,
        "public_gate_evaluations": 0,
        "manifest_created": False,
        "model_store_promoted": False,
        "packaging_discovery": False,
        "private_validation": False,
        "production_approval": False,
        "release_eligible": False,
    }


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Marker-center V11 P3 output exists: {output_dir}")
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
            "schema": "graphreader.marker-center-seed-refinement-failure.v11",
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
        default=REPO_ROOT / "ml/markers/center/artifacts/seed-refinement-v11/P3-run",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
