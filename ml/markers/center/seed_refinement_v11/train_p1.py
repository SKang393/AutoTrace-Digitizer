# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 training and visible selection for marker-center V11."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as functional

from ml.markers.center.dense_contract_v5.train_p1 import (
    _artifact_loss,
    _center_focal_loss,
    _configure_determinism,
    _export,
    _passing_window,
)
from ml.markers.center.dense_contract_v5.train_p3 import (
    _fuse_inference_model,
)
from ml.markers.center.decoupled_heads_v10.train_p2 import (
    REFLECTION_SCHEDULE,
    _reflect_tensor,
    _reflected_spatial_margin_losses,
)
from ml.markers.center.metrics import aggregate_scene_metrics, center_metrics
from ml.markers.center.postprocess import detect_heads
from ml.markers.center.seed_refinement_v11.dataset import (
    HARD_NEGATIVE_TOLERANCE,
    KIND_TO_INDEX,
    MATCH_TOLERANCE,
    PROHIBITED_KINDS,
    TRAIN_SCENE_COUNT,
    read_archive,
)
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
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/seed_refinement_v11"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/seed_refinement_v11/training/p1.json")
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
)
BATCH_SIZE = 8
EPOCHS = 28
LEARNING_RATE = 0.0005
CENTER_LOSS_WEIGHT = 3.0
ARTIFACT_LOSS_WEIGHT = 1.5
ARTIFACT_BCE_FRACTION = 0.35
ARTIFACT_FALSE_POSITIVE_WEIGHT = 0.95
ARTIFACT_FALSE_NEGATIVE_WEIGHT = 0.05
MARKER_CLEAR_LOSS_WEIGHT = 1.5
POSITIVE_MARGIN_LOSS_WEIGHT = 2.0
HARD_NEGATIVE_MARGIN_LOSS_WEIGHT = 3.0
FIXED_RADIUS_PIXELS = 2.5
MAXIMUM_LOGIT_CORRECTION = 6.0


def _specificity_artifact_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


@torch.inference_mode()
def _torch_output(model: torch.nn.Module, value: np.ndarray) -> np.ndarray:
    return model(torch.from_numpy(np.asarray(value, dtype=np.float32))).cpu().numpy().astype(
        np.float32
    )


def _onnx_output(session: ort.InferenceSession, value: np.ndarray) -> np.ndarray:
    return np.asarray(
        session.run(["marker_heads"], {"image_and_masks": value})[0],
        dtype=np.float32,
    )


def _centers(archive: dict[str, np.ndarray], index: int) -> tuple[tuple[float, float], ...]:
    count = int(archive["center_counts"][index])
    return tuple(
        (float(row[0]), float(row[1]))
        for row in archive["centers"][index, :count]
    )


def _hard_negatives(
    archive: dict[str, np.ndarray], index: int
) -> tuple[tuple[str, float, float], ...]:
    count = int(archive["hard_counts"][index])
    reverse = {value: key for key, value in KIND_TO_INDEX.items()}
    return tuple(
        (
            reverse[int(archive["hard_kinds"][index, ordinal])],
            float(archive["hard_points"][index, ordinal, 0]),
            float(archive["hard_points"][index, ordinal, 1]),
        )
        for ordinal in range(count)
    )


def _evaluate(
    archive: dict[str, np.ndarray],
    outputs: list[np.ndarray],
    threshold: float,
) -> dict[str, object]:
    scene_metrics = []
    exact_scene_count = 0
    prohibited = {kind: 0 for kind in PROHIBITED_KINDS}
    artifact_intersection = 0
    artifact_predicted = 0
    artifact_truth = 0
    marker_artifact_hits = 0
    seed_removed_pixels = 0
    seed_added_pixels = 0
    for index, output in enumerate(outputs):
        value = archive["inputs"][index : index + 1]
        refined_artifact = output[0, 2]
        detections = detect_heads(
            output,
            text_mask=value[0, 1],
            artifact_mask=refined_artifact,
            center_threshold=threshold,
            artifact_threshold=ARTIFACT_THRESHOLD,
        )
        metric = center_metrics(detections, _centers(archive, index), MATCH_TOLERANCE)
        scene_metrics.append(metric)
        if metric.false_positives == 0 and metric.false_negatives == 0 and metric.duplicate_count == 0:
            exact_scene_count += 1
        for kind, x, y in _hard_negatives(archive, index):
            prohibited[kind] += sum(
                math.hypot(item.x - x, item.y - y) <= HARD_NEGATIVE_TOLERANCE
                for item in detections
            )
        predicted_mask = refined_artifact >= ARTIFACT_THRESHOLD
        seed_mask = value[0, 2] >= ARTIFACT_THRESHOLD
        truth_mask = archive["artifact_targets"][index, 0] >= 0.5
        artifact_intersection += int(np.logical_and(predicted_mask, truth_mask).sum())
        artifact_predicted += int(predicted_mask.sum())
        artifact_truth += int(truth_mask.sum())
        seed_removed_pixels += int(np.logical_and(seed_mask, ~predicted_mask).sum())
        seed_added_pixels += int(np.logical_and(~seed_mask, predicted_mask).sum())
        for x, y in _centers(archive, index):
            marker_artifact_hits += int(refined_artifact[round(y), round(x)] >= ARTIFACT_THRESHOLD)
    aggregate = aggregate_scene_metrics(scene_metrics, MATCH_TOLERANCE)
    artifact_precision = artifact_intersection / artifact_predicted if artifact_predicted else 0.0
    artifact_recall = artifact_intersection / artifact_truth if artifact_truth else 1.0
    passed = (
        exact_scene_count == len(outputs)
        and aggregate.false_positives == 0
        and aggregate.false_negatives == 0
        and aggregate.duplicate_count == 0
        and not any(prohibited.values())
        and artifact_precision >= MINIMUM_ARTIFACT_PRECISION
        and artifact_recall >= MINIMUM_ARTIFACT_RECALL
        and marker_artifact_hits == 0
    )
    return {
        "threshold": threshold,
        "passed": passed,
        "scene_count": len(outputs),
        "exact_scene_count": exact_scene_count,
        "true_positives": aggregate.true_positives,
        "false_positives": aggregate.false_positives,
        "false_negatives": aggregate.false_negatives,
        "duplicate_count": aggregate.duplicate_count,
        "prohibited_structure_hits": prohibited,
        "artifact_precision": artifact_precision,
        "artifact_recall": artifact_recall,
        "marker_artifact_hits": marker_artifact_hits,
        "seed_removed_pixels": seed_removed_pixels,
        "seed_added_pixels": seed_added_pixels,
    }


def _verify_config_and_inputs(config: dict[str, object]) -> tuple[dict[str, object], Path, Path]:
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
        "positive_margin_loss_weight": POSITIVE_MARGIN_LOSS_WEIGHT,
        "runtime_pass_count": 1,
        "runtime_postprocess_profile": "nonmonotonic_seed_refinement_v1",
        "selection_thresholds": list(THRESHOLDS),
        "trigger_result_sha256": TRIGGER_RESULT_SHA256,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V11 P1 configuration changed: {key}")
    trigger_path = REPO_ROOT / TRIGGER_RESULT_PATH
    if sha256_file(trigger_path) != TRIGGER_RESULT_SHA256:
        raise RuntimeError("Marker-center V10 aggregate feasibility proof changed")
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    if (
        trigger.get("status") != "structurally_infeasible_exhausted"
        or trigger.get("monotonic_seed_union_can_pass_precision") is not False
        or trigger.get("maximum_precision_under_monotonic_seed_union")
        != 0.796366958474284
        or trigger.get("case_detail_or_pixels_inspected") is not False
        or trigger.get("public_gate_archive_opened") is not False
    ):
        raise RuntimeError("Marker-center V11 requires the terminal aggregate V10 feasibility proof")
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
                    "marker_clear": float(marker_clear_loss.detach()),
                    "positive_margin": float(positive_margin_loss.detach()),
                    "hard_negative_margin": float(hard_negative_margin_loss.detach()),
                }
            )
    if optimizer_steps != int(config["expected_optimizer_steps"]):
        raise RuntimeError("Marker-center V11 P1 optimizer step count changed")
    if set(reflection_epoch_counts.values()) != {EPOCHS // len(REFLECTION_SCHEDULE)}:
        raise RuntimeError("Marker-center V11 reflection schedule did not complete evenly")
    model.eval()
    progress["phase"] = "export"
    checkpoint_path = output_dir / "marker-center-seed-refinement-v11-p1.pt"
    onnx_path = output_dir / "marker-center-seed-refinement-v11-p1.onnx"
    inference_model = _fuse_inference_model(model)
    sample = torch.from_numpy(validation["inputs"][0:1])
    _export(inference_model, sample, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Marker-center V11 P1 selection requires CPUExecutionProvider only")
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
        raise RuntimeError(f"Marker-center V11 P1 output exists: {output_dir}")
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
        default=REPO_ROOT / "ml/markers/center/artifacts/seed-refinement-v11/P1-run",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
