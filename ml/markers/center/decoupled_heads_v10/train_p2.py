# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P2 reflection-augmented training and visible selection for V10."""

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
    HARD_NEGATIVE_TOLERANCE,
    MATCH_TOLERANCE,
    _center_focal_loss,
    _configure_determinism,
    _evaluate,
    _export,
    _onnx_outputs,
    _passing_window,
    _torch_outputs,
)
from ml.markers.center.dense_contract_v5.train_p3 import (
    _fuse_inference_model,
    _maximum_output_difference,
)
from ml.markers.center.decoupled_heads_v10.dataset import TRAIN_SCENE_COUNT, read_archive
from ml.markers.center.decoupled_heads_v10.model import create_model, save_checkpoint
from ml.markers.center.decoupled_heads_v10.protocol import (
    ONNX_PARITY_TOLERANCE,
    REVISION,
    TASK,
    THRESHOLDS,
)
from ml.markers.center.decoupled_heads_v10.train_p1 import (
    ARTIFACT_BCE_FRACTION,
    ARTIFACT_FALSE_NEGATIVE_WEIGHT,
    ARTIFACT_FALSE_POSITIVE_WEIGHT,
    ARTIFACT_LOSS_WEIGHT,
    BATCH_SIZE,
    CENTER_LOSS_WEIGHT,
    EPOCHS,
    HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
    LEARNING_RATE,
    MARKER_CLEAR_LOSS_WEIGHT,
    POSITIVE_MARGIN_LOSS_WEIGHT,
    _specificity_artifact_loss,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/decoupled_heads_v10"
CANDIDATE_ID = "P2"
CONFIG_PATH = Path("ml/markers/center/decoupled_heads_v10/training/p2.json")
P1_RESULT_PATH = Path("ml/markers/center/decoupled_heads_v10/P1_RESULT.json")
P1_RESULT_SHA256 = "72e26731a7398f87bb6a700f4acf920907b8e49562ef588e442d8bc55ec92ad4"
REFLECTION_SCHEDULE = (
    "none",
    "horizontal_reflection",
    "vertical_reflection",
    "horizontal_vertical_reflection",
)
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
    Path("ml/markers/center/decoupled_heads_v10/dataset.py"),
    Path("ml/markers/center/decoupled_heads_v10/model.py"),
    Path("ml/markers/center/decoupled_heads_v10/protocol.py"),
    Path("ml/markers/center/decoupled_heads_v10/train_p1.py"),
    Path("ml/markers/center/decoupled_heads_v10/train_p2.py"),
)


def _reflect_tensor(value: torch.Tensor, transform_index: int) -> torch.Tensor:
    if transform_index not in range(len(REFLECTION_SCHEDULE)):
        raise ValueError("Unknown V10 P2 reflection index")
    dimensions: list[int] = []
    if transform_index & 1:
        dimensions.append(-1)
    if transform_index & 2:
        dimensions.append(-2)
    return torch.flip(value, dimensions) if dimensions else value


def _reflect_point(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    transform_index: int,
) -> tuple[float, float]:
    if transform_index & 1:
        x = (width - 1) - x
    if transform_index & 2:
        y = (height - 1) - y
    return x, y


def _disk_maximum(
    plane: torch.Tensor,
    *,
    x: float,
    y: float,
    radius: float,
) -> torch.Tensor:
    height, width = plane.shape
    left = max(0, int(math.floor(x - radius)))
    right = min(width - 1, int(math.ceil(x + radius)))
    top = max(0, int(math.floor(y - radius)))
    bottom = min(height - 1, int(math.ceil(y + radius)))
    ys = torch.arange(top, bottom + 1, device=plane.device, dtype=plane.dtype)
    xs = torch.arange(left, right + 1, device=plane.device, dtype=plane.dtype)
    mask = (ys[:, None] - y).square() + (xs[None, :] - x).square() <= radius * radius
    return plane[top : bottom + 1, left : right + 1][mask].max()


def _reflected_spatial_margin_losses(
    center_prediction: torch.Tensor,
    archive: dict[str, np.ndarray],
    batch_indices: np.ndarray,
    transform_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_values: list[torch.Tensor] = []
    negative_values: list[torch.Tensor] = []
    height = int(center_prediction.shape[-2])
    width = int(center_prediction.shape[-1])
    for batch_ordinal, scene_index in enumerate(batch_indices):
        plane = center_prediction[batch_ordinal, 0]
        center_count = int(archive["center_counts"][scene_index])
        for point in archive["centers"][scene_index, :center_count]:
            x, y = _reflect_point(
                float(point[0]),
                float(point[1]),
                width=width,
                height=height,
                transform_index=transform_index,
            )
            positive_values.append(
                _disk_maximum(plane, x=x, y=y, radius=MATCH_TOLERANCE)
            )
        hard_count = int(archive["hard_counts"][scene_index])
        for point in archive["hard_points"][scene_index, :hard_count]:
            x, y = _reflect_point(
                float(point[0]),
                float(point[1]),
                width=width,
                height=height,
                transform_index=transform_index,
            )
            negative_values.append(
                _disk_maximum(plane, x=x, y=y, radius=HARD_NEGATIVE_TOLERANCE)
            )
    zero = center_prediction.sum() * 0.0
    if positive_values:
        positives = torch.stack(positive_values).clamp(1e-5, 1.0 - 1e-5)
        positive_loss = functional.binary_cross_entropy(positives, torch.ones_like(positives))
    else:
        positive_loss = zero
    if negative_values:
        negatives = torch.stack(negative_values).clamp(1e-5, 1.0 - 1e-5)
        negative_loss = functional.binary_cross_entropy(negatives, torch.zeros_like(negatives))
    else:
        negative_loss = zero
    return positive_loss, negative_loss


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
        "augmentation_schedule": list(REFLECTION_SCHEDULE),
        "batch_size": BATCH_SIZE,
        "center_loss_weight": CENTER_LOSS_WEIGHT,
        "epochs": EPOCHS,
        "expected_optimizer_steps": EPOCHS * (TRAIN_SCENE_COUNT // BATCH_SIZE),
        "hard_negative_margin_loss_weight": HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "marker_clear_loss_weight": MARKER_CLEAR_LOSS_WEIGHT,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "p1_checkpoint_reused": False,
        "p1_result_sha256": P1_RESULT_SHA256,
        "positive_margin_loss_weight": POSITIVE_MARGIN_LOSS_WEIGHT,
        "selection_thresholds": list(THRESHOLDS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V10 P2 configuration changed: {key}")
    p1_result_path = REPO_ROOT / P1_RESULT_PATH
    if sha256_file(p1_result_path) != P1_RESULT_SHA256:
        raise RuntimeError("Marker-center V10 P1 result changed")
    p1_result = json.loads(p1_result_path.read_text(encoding="utf-8"))
    if (
        p1_result.get("status") != "failed_selection_consumed"
        or p1_result.get("selection_exact_scene_count") != 120
        or p1_result.get("selection_false_positives") != 11
        or p1_result.get("selection_false_negatives") != 23
        or p1_result.get("public_gate_archive_opened") is not False
        or p1_result.get("case_detail_or_pixels_inspected") is not False
    ):
        raise RuntimeError("Marker-center V10 P2 requires the consumed aggregate P1 failure")
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
    loss_checkpoints: list[dict[str, float | int | str]] = []
    reflection_epoch_counts = {name: 0 for name in REFLECTION_SCHEDULE}
    progress["phase"] = "training"
    model.train()
    for epoch in range(EPOCHS):
        transform_index = epoch % len(REFLECTION_SCHEDULE)
        transform_name = REFLECTION_SCHEDULE[transform_index]
        reflection_epoch_counts[transform_name] += 1
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
                    "reflection": transform_name,
                    "total": float(loss.detach()),
                    "center": float(center_loss.detach()),
                    "artifact": float(artifact_loss.detach()),
                    "marker_clear": float(marker_clear_loss.detach()),
                    "positive_margin": float(positive_margin_loss.detach()),
                    "hard_negative_margin": float(hard_negative_margin_loss.detach()),
                }
            )
    if optimizer_steps != int(config["expected_optimizer_steps"]):
        raise RuntimeError("Marker-center V10 P2 optimizer step count changed")
    if set(reflection_epoch_counts.values()) != {EPOCHS // len(REFLECTION_SCHEDULE)}:
        raise RuntimeError("Marker-center V10 P2 reflection schedule changed")
    model.eval()
    progress["phase"] = "export"
    checkpoint_path = output_dir / "marker-center-decoupled-heads-v10-p2.pt"
    onnx_path = output_dir / "marker-center-decoupled-heads-v10-p2.onnx"
    inference_model = _fuse_inference_model(model)
    sample = torch.from_numpy(validation["inputs"][0:1])
    _export(inference_model, sample, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Marker-center V10 P2 selection requires CPUExecutionProvider only")
    progress["phase"] = "selection"
    onnx_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    parity = 0.0
    inference_semantic_error = 0.0
    input_stream = hashlib.sha256()
    output_stream = hashlib.sha256()
    for index in range(validation["inputs"].shape[0]):
        value = validation["inputs"][index : index + 1]
        raw_outputs = _torch_outputs(model, value)
        frozen_outputs = _torch_outputs(inference_model, value)
        onnx_values = _onnx_outputs(session, value)
        inference_semantic_error = max(
            inference_semantic_error, _maximum_output_difference(raw_outputs, frozen_outputs)
        )
        parity = max(parity, _maximum_output_difference(frozen_outputs, onnx_values))
        first, second_input, second = onnx_values
        onnx_outputs.append((first, second))
        input_stream.update(value.tobytes(order="C"))
        input_stream.update(second_input.tobytes(order="C"))
        output_stream.update(first.tobytes(order="C"))
        output_stream.update(second.tobytes(order="C"))
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
    selection_passed = len(window) >= 3 and parity <= ONNX_PARITY_TOLERANCE
    save_checkpoint(
        checkpoint_path,
        model,
        selected_threshold=float(selected["threshold"]),
        dataset_manifest_sha256=selection["train"]["archive_sha256"],
        training_revision=REVISION,
    )
    return {
        "schema": "graphreader.marker-center-decoupled-heads-candidate.v10",
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
        "optimizer_steps": optimizer_steps,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "reflection_epoch_counts": reflection_epoch_counts,
        "p1_checkpoint_reused": False,
        "p1_result_sha256": P1_RESULT_SHA256,
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
        "direct_execution_inference_calls": validation["inputs"].shape[0] * 2,
        "input_tensor_stream_sha256": input_stream.hexdigest(),
        "output_tensor_stream_sha256": output_stream.hexdigest(),
        "loss_checkpoints": loss_checkpoints,
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
        raise RuntimeError(f"Marker-center V10 P2 output exists: {output_dir}")
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
        default=REPO_ROOT / "ml/markers/center/artifacts/decoupled-heads-v10/P2-run",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
