# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use final P3 selection-aligned fine-tuning for marker-center V8."""

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
from torch import nn
import torch.nn.functional as functional

from ml.markers.center.dense_contract_v5.train_p1 import (
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
    _spatial_margin_losses,
)
from ml.markers.center.mask_consensus_v8.dataset import TRAIN_SCENE_COUNT, read_archive
from ml.markers.center.mask_consensus_v8.model import load_checkpoint, save_checkpoint
from ml.markers.center.mask_consensus_v8.protocol import (
    ONNX_PARITY_TOLERANCE,
    REVISION,
    TASK,
    THRESHOLDS,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/mask_consensus_v8"
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/markers/center/mask_consensus_v8/training/p3.json")
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
    Path("ml/markers/center/mask_consensus_v8/model.py"),
    Path("ml/markers/center/mask_consensus_v8/protocol.py"),
    Path("ml/markers/center/mask_consensus_v8/train_p3.py"),
)
BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 0.0001
CENTER_LOSS_WEIGHT = 3.0
ARTIFACT_LOSS_WEIGHT = 1.5
MARKER_CLEAR_LOSS_WEIGHT = 1.25
POSITIVE_MARGIN_LOSS_WEIGHT = 1.5
HARD_NEGATIVE_MARGIN_LOSS_WEIGHT = 2.0
ARTIFACT_POSITIVE_WEIGHT = 1.0
FIXED_RADIUS_PIXELS = 2.5
EXPECTED_OPTIMIZER_STEPS = EPOCHS * (TRAIN_SCENE_COUNT // BATCH_SIZE)


class FixedRadiusInferenceModel(nn.Module):
    """Retain learned center/artifact heads and emit a parity-stable NMS support radius."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        heads = self.model(value)
        radius = torch.full_like(heads[:, 1:2], FIXED_RADIUS_PIXELS)
        return torch.cat((heads[:, 0:1], radius, heads[:, 2:3]), dim=1)


def _specificity_balanced_artifact_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.clamp(1e-5, 1 - 1e-5)
    bce = -(
        ARTIFACT_POSITIVE_WEIGHT * target * prediction.log()
        + (1 - target) * (1 - prediction).log()
    ).mean()
    intersection = (prediction * target).sum()
    dice = 1 - ((2 * intersection + 1) / (prediction.sum() + target.sum() + 1))
    return bce + dice


def _photometric_batch(
    inputs: torch.Tensor,
    *,
    epoch: int,
    batch_ordinal: int,
) -> torch.Tensor:
    """Apply deterministic ink-only capture variation without moving labels or masks."""

    result = inputs.clone()
    phase = (epoch * 11 + batch_ordinal * 7) % 9
    contrast = 0.84 + 0.04 * phase
    offset = -0.04 + 0.01 * ((epoch * 5 + batch_ordinal * 3) % 9)
    result[:, 0:1] = ((result[:, 0:1] - 0.5) * contrast + 0.5 + offset).clamp(0.0, 1.0)
    return result


def _per_channel_parity(
    model: nn.Module,
    session: ort.InferenceSession,
    validation: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    maximum = np.zeros(3, dtype=np.float64)
    for index in range(validation["inputs"].shape[0]):
        value = validation["inputs"][index : index + 1]
        torch_values = _torch_outputs(model, value)
        onnx_values = _onnx_outputs(session, value)
        for torch_value, onnx_value in zip(
            (torch_values[0], torch_values[2]),
            (onnx_values[0], onnx_values[2]),
            strict=True,
        ):
            maximum = np.maximum(
                maximum,
                np.max(np.abs(torch_value - onnx_value), axis=(0, 2, 3)),
            )
    return tuple(float(value) for value in maximum)


def _verify_config_and_inputs(config: dict[str, object]) -> tuple[dict[str, object], Path, Path]:
    expected = {
        "artifact_loss_weight": ARTIFACT_LOSS_WEIGHT,
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
        "selection_thresholds": list(THRESHOLDS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V8 P3 configuration changed: {key}")
    p2_result_path = ROOT / "P2_RESULT.json"
    if sha256_file(p2_result_path) != config["p2_result_sha256"]:
        raise RuntimeError("Marker-center V8 P2 result changed")
    p2_result = json.loads(p2_result_path.read_text(encoding="utf-8"))
    if p2_result.get("status") != "failed_selection_consumed":
        raise RuntimeError("Marker-center V8 P2 is not a consumed selection failure")
    checkpoint_path = REPO_ROOT / str(p2_result["checkpoint_path"])
    onnx_path = REPO_ROOT / str(p2_result["onnx_path"])
    for path, key in (
        (checkpoint_path, "p2_checkpoint_sha256"),
        (onnx_path, "p2_onnx_sha256"),
    ):
        if sha256_file(path) != config[key]:
            raise RuntimeError(f"Marker-center V8 P2 evidence changed: {path.name}")
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    if sha256_file(selection_path) != config["selection_manifest_sha256"]:
        raise RuntimeError("Marker-center V8 selection manifest changed")
    return p2_result, checkpoint_path, onnx_path


def _execute_candidate(
    output_dir: Path,
    authorization: TrainingAuthorization,
    progress: dict[str, object],
) -> dict[str, object]:
    started = float(progress["started"])
    progress["phase"] = "input_validation"
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    p2_result, checkpoint_source, onnx_source = _verify_config_and_inputs(config)
    selection = json.loads((ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    train_path = REPO_ROOT / selection["train"]["archive_path"]
    validation_path = REPO_ROOT / selection["validation"]["archive_path"]
    for path, expected_sha256 in (
        (train_path, selection["train"]["archive_sha256"]),
        (validation_path, selection["validation"]["archive_sha256"]),
    ):
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Frozen marker-center V8 archive changed: {path.name}")
    train = read_archive(train_path)
    validation = read_archive(validation_path)

    progress["phase"] = "p2_parity_localization"
    p2_model, _ = load_checkpoint(checkpoint_source)
    p2_inference = _fuse_inference_model(p2_model)
    p2_session = ort.InferenceSession(str(onnx_source), providers=["CPUExecutionProvider"])
    if p2_session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Marker-center V8 P3 diagnostics require CPUExecutionProvider only")
    p2_parity = _per_channel_parity(p2_inference, p2_session, validation)
    expected_p2_parity = tuple(float(value) for value in config["p2_parity_by_output_channel"])
    if p2_parity != expected_p2_parity:
        raise RuntimeError("Marker-center V8 P2 per-channel parity evidence changed")

    progress["phase"] = "training"
    _configure_determinism(int(config["seed"]))
    model, _ = load_checkpoint(checkpoint_source)
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("radius_head.")
    ]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-5)
    indices = np.arange(train["inputs"].shape[0])
    optimizer_steps = 0
    loss_checkpoints: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_indices = np.roll(indices, (epoch * 29) % len(indices))
        for batch_ordinal, start in enumerate(range(0, len(epoch_indices), BATCH_SIZE)):
            batch_indices = epoch_indices[start : start + BATCH_SIZE]
            inputs = _photometric_batch(
                torch.from_numpy(train["inputs"][batch_indices]),
                epoch=epoch,
                batch_ordinal=batch_ordinal,
            )
            center_target = torch.from_numpy(train["center_targets"][batch_indices])
            artifact_target = torch.from_numpy(train["artifact_targets"][batch_indices])
            optimizer.zero_grad(set_to_none=True)
            heads = model(inputs)
            center_loss = _center_focal_loss(heads[:, 0:1], center_target)
            artifact_loss = _specificity_balanced_artifact_loss(heads[:, 2:3], artifact_target)
            marker_pixels = center_target >= 0.1
            marker_clear_loss = functional.binary_cross_entropy(
                heads[:, 2:3][marker_pixels],
                torch.zeros_like(heads[:, 2:3][marker_pixels]),
            )
            positive_margin_loss, hard_negative_margin_loss = _spatial_margin_losses(
                heads[:, 0:1],
                train,
                batch_indices,
            )
            loss = (
                CENTER_LOSS_WEIGHT * center_loss
                + ARTIFACT_LOSS_WEIGHT * artifact_loss
                + MARKER_CLEAR_LOSS_WEIGHT * marker_clear_loss
                + POSITIVE_MARGIN_LOSS_WEIGHT * positive_margin_loss
                + HARD_NEGATIVE_MARGIN_LOSS_WEIGHT * hard_negative_margin_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
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
    if optimizer_steps != EXPECTED_OPTIMIZER_STEPS:
        raise RuntimeError("Marker-center V8 P3 optimizer step count changed")

    progress["phase"] = "export"
    model.eval()
    checkpoint_model = FixedRadiusInferenceModel(model).eval()
    inference_model = FixedRadiusInferenceModel(_fuse_inference_model(model)).eval()
    checkpoint_path = output_dir / "marker-center-mask-consensus-v8-p3.pt"
    onnx_path = output_dir / "marker-center-mask-consensus-v8-p3.onnx"
    sample = torch.from_numpy(validation["inputs"][0:1])
    _export(inference_model, sample, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Marker-center V8 P3 selection requires CPUExecutionProvider only")

    progress["phase"] = "selection"
    onnx_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    parity = 0.0
    inference_semantic_error = 0.0
    input_stream = hashlib.sha256()
    output_stream = hashlib.sha256()
    for index in range(validation["inputs"].shape[0]):
        value = validation["inputs"][index : index + 1]
        checkpoint_values = _torch_outputs(checkpoint_model, value)
        inference_values = _torch_outputs(inference_model, value)
        onnx_values = _onnx_outputs(session, value)
        inference_semantic_error = max(
            inference_semantic_error,
            _maximum_output_difference(checkpoint_values, inference_values),
        )
        parity = max(parity, _maximum_output_difference(inference_values, onnx_values))
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
        "schema": "graphreader.marker-center-mask-consensus-candidate.v8",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "selection_passed" if selection_passed else "failed_selection",
        "selection_gate_passed": selection_passed,
        "isolated_change": config["isolated_change"],
        "aggregate_design_basis": config["aggregate_design_basis"],
        "private_data": False,
        "chandler_used": False,
        "synthetic_only": True,
        "provider": "CPUExecutionProvider",
        "optimizer_steps": optimizer_steps,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "inference_graph_transform": "fixed_2_5_pixel_nms_support_radius_v1",
        "checkpoint_to_inference_graph_maximum_absolute_error": inference_semantic_error,
        "onnx_parity_maximum_absolute_error": parity,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "onnx_parity_passed": parity <= ONNX_PARITY_TOLERANCE,
        "p2_parity_by_output_channel": list(p2_parity),
        "selected_threshold": selected["threshold"],
        "selection_metrics": selected,
        "passing_threshold_window": window,
        "threshold_aggregates": comparisons,
        "direct_execution_inference_calls": validation["inputs"].shape[0] * 4,
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
        raise RuntimeError(f"Marker-center V8 P3 output exists: {output_dir}")
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
            "schema": "graphreader.marker-center-mask-consensus-failure.v8",
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/P3")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
