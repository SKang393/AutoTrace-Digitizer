# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-use P2 hard-negative and center-recall repair for dense-contract V5."""

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

from ml.markers.center.dense_contract_v5.dataset import read_archive
from ml.markers.center.dense_contract_v5.model import create_model
from ml.markers.center.dense_contract_v5.train_p1 import (
    ARTIFACT_THRESHOLD,
    BATCH_SIZE,
    PARITY_TOLERANCE,
    REPO_ROOT,
    REVISION,
    TASK,
    THRESHOLDS,
    _artifact_loss,
    _center_focal_loss,
    _configure_determinism,
    _evaluate,
    _export,
    _onnx_outputs,
    _passing_window,
    _second_input,
    _torch_outputs,
)
from ml.markers.center.model import save_checkpoint
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


ROOT = REPO_ROOT / "ml/markers/center/dense_contract_v5"
CANDIDATE_ID = "P2"
CONFIG_PATH = Path("ml/markers/center/dense_contract_v5/training/p2.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/markers/center/model.py"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/markers/center/postprocess.py"),
    Path("ml/markers/center/dense_contract_v5/dataset.py"),
    Path("ml/markers/center/dense_contract_v5/model.py"),
    Path("ml/markers/center/dense_contract_v5/train_p1.py"),
    Path("ml/markers/center/dense_contract_v5/train_p2.py"),
)
EPOCHS = 72
LEARNING_RATE = 0.0015
CENTER_LOSS_WEIGHT = 2.0
HARD_NEGATIVE_LOSS_WEIGHT = 0.8


def _point_loss(
    center_prediction: torch.Tensor,
    archive: dict[str, np.ndarray],
    batch_indices: np.ndarray,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for batch_ordinal, scene_index in enumerate(batch_indices):
        hard_count = int(archive["hard_counts"][scene_index])
        for point in archive["hard_points"][scene_index, :hard_count]:
            x = max(0, min(center_prediction.shape[3] - 1, round(float(point[0]))))
            y = max(0, min(center_prediction.shape[2] - 1, round(float(point[1]))))
            values.append(center_prediction[batch_ordinal, 0, y, x])
    if not values:
        return center_prediction.sum() * 0.0
    predictions = torch.stack(values).clamp(1e-5, 1 - 1e-5)
    return functional.binary_cross_entropy(predictions, torch.zeros_like(predictions))


def _execute_candidate(
    output_dir: Path,
    authorization: TrainingAuthorization,
    progress: dict[str, object],
) -> dict[str, object]:
    started = float(progress["started"])
    progress["phase"] = "input_validation"
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    expected_configuration = {
        "batch_size": BATCH_SIZE,
        "center_loss_weight": CENTER_LOSS_WEIGHT,
        "epochs": EPOCHS,
        "expected_optimizer_steps": EPOCHS * (96 // BATCH_SIZE),
        "hard_negative_loss_weight": HARD_NEGATIVE_LOSS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "onnx_parity_tolerance": PARITY_TOLERANCE,
        "selection_thresholds": list(THRESHOLDS),
    }
    for key, expected in expected_configuration.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Dense-contract V5 P2 configuration changed: {key}")
    selection = json.loads((ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256_file(ROOT / "SELECTION_MANIFEST.json") != config["selection_manifest_sha256"]:
        raise RuntimeError("Dense-contract V5 P2 selection manifest changed")
    train_path = REPO_ROOT / selection["train"]["archive_path"]
    validation_path = REPO_ROOT / selection["validation"]["archive_path"]
    if sha256_file(train_path) != selection["train"]["archive_sha256"]:
        raise RuntimeError("Frozen training archive changed")
    if sha256_file(validation_path) != selection["validation"]["archive_sha256"]:
        raise RuntimeError("Frozen validation archive changed")
    train = read_archive(train_path)
    validation = read_archive(validation_path)
    _configure_determinism(int(config["seed"]))
    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    indices = np.arange(train["inputs"].shape[0])
    loss_checkpoints: list[dict[str, float | int]] = []
    optimizer_steps = 0
    progress["phase"] = "training"
    model.train()
    for epoch in range(EPOCHS):
        epoch_indices = np.roll(indices, (epoch * 7) % len(indices))
        for start in range(0, len(epoch_indices), BATCH_SIZE):
            batch_indices = epoch_indices[start : start + BATCH_SIZE]
            inputs = torch.from_numpy(train["inputs"][batch_indices])
            center_target = torch.from_numpy(train["center_targets"][batch_indices])
            radius_target = torch.from_numpy(train["radius_targets"][batch_indices])
            artifact_target = torch.from_numpy(train["artifact_targets"][batch_indices])
            optimizer.zero_grad(set_to_none=True)
            heads = model(inputs)
            center_loss = _center_focal_loss(heads[:, 0:1], center_target)
            radius_mask = center_target >= 0.25
            radius_loss = functional.smooth_l1_loss(heads[:, 1:2][radius_mask], radius_target[radius_mask])
            artifact_loss = _artifact_loss(heads[:, 2:3], artifact_target)
            marker_pixels = center_target >= 0.1
            marker_clear_loss = functional.binary_cross_entropy(
                heads[:, 2:3][marker_pixels],
                torch.zeros_like(heads[:, 2:3][marker_pixels]),
            )
            hard_negative_loss = _point_loss(heads[:, 0:1], train, batch_indices)
            loss = (
                (CENTER_LOSS_WEIGHT * center_loss)
                + (0.15 * radius_loss)
                + (0.65 * artifact_loss)
                + marker_clear_loss
                + (HARD_NEGATIVE_LOSS_WEIGHT * hard_negative_loss)
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
                    "radius": float(radius_loss.detach()),
                    "artifact": float(artifact_loss.detach()),
                    "marker_clear": float(marker_clear_loss.detach()),
                    "hard_negative": float(hard_negative_loss.detach()),
                }
            )
    model.eval()
    if optimizer_steps != int(config["expected_optimizer_steps"]):
        raise RuntimeError("Dense-contract V5 P2 optimizer step count changed")
    progress["phase"] = "export"
    checkpoint_path = output_dir / "marker-center-dense-contract-v5-p2.pt"
    onnx_path = output_dir / "marker-center-dense-contract-v5-p2.onnx"
    sample = torch.from_numpy(validation["inputs"][0:1])
    _export(model, sample, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Dense-contract V5 P2 selection requires CPUExecutionProvider only")
    progress["phase"] = "selection"
    onnx_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    parity = 0.0
    input_stream = hashlib.sha256()
    output_stream = hashlib.sha256()
    for index in range(validation["inputs"].shape[0]):
        value = validation["inputs"][index : index + 1]
        torch_first, torch_second_input, torch_second = _torch_outputs(model, value)
        onnx_first, onnx_second_input, onnx_second = _onnx_outputs(session, value)
        parity = max(
            parity,
            float(np.max(np.abs(torch_first - onnx_first))),
            float(np.max(np.abs(torch_second_input - onnx_second_input))),
            float(np.max(np.abs(torch_second - onnx_second))),
        )
        onnx_outputs.append((onnx_first, onnx_second))
        input_stream.update(value.tobytes(order="C"))
        input_stream.update(onnx_second_input.tobytes(order="C"))
        output_stream.update(onnx_first.tobytes(order="C"))
        output_stream.update(onnx_second.tobytes(order="C"))
    comparisons = [_evaluate(validation, onnx_outputs, threshold) for threshold in THRESHOLDS]
    window = _passing_window(comparisons)
    selected = max(
        comparisons,
        key=lambda item: (
            bool(item["passed"]),
            int(item["exact_scene_count"]),
            float(item["artifact_recall"]),
            float(item["artifact_precision"]),
            -abs(float(item["threshold"]) - 0.45),
        ),
    )
    selection_passed = len(window) >= 3 and parity <= PARITY_TOLERANCE
    save_checkpoint(
        checkpoint_path,
        model,
        selected_threshold=float(selected["threshold"]),
        dataset_manifest_sha256=selection["train"]["archive_sha256"],
        training_revision=REVISION,
    )
    return {
        "schema": "graphreader.marker-center-dense-contract-candidate.v5",
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
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_parity_maximum_absolute_error": parity,
        "onnx_parity_tolerance": PARITY_TOLERANCE,
        "onnx_parity_passed": parity <= PARITY_TOLERANCE,
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
        "production_approval": False,
        "release_eligible": False,
    }


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Dense-contract V5 P2 output exists: {output_dir}")
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
    progress: dict[str, object] = {"started": time.perf_counter(), "phase": "initialization", "optimizer_steps": 0}
    try:
        report = _execute_candidate(output_dir, authorization, progress)
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-dense-contract-failure.v5",
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
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.output.resolve())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
