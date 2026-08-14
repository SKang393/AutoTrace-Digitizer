# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-use P1 training and visible selection for dense-contract V5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as functional

from ml.markers.center.dense_contract_v5.dataset import KIND_TO_INDEX, PROHIBITED_KINDS, read_archive
from ml.markers.center.dense_contract_v5.model import create_model
from ml.markers.center.metrics import aggregate_scene_metrics, center_metrics
from ml.markers.center.model import save_checkpoint
from ml.markers.center.postprocess import detect_heads
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/dense_contract_v5"
TASK = "marker-center"
REVISION = "marker-center-dense-contract-v5"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/dense_contract_v5/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/markers/center/model.py"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/markers/center/postprocess.py"),
    Path("ml/markers/center/dense_contract_v5/dataset.py"),
    Path("ml/markers/center/dense_contract_v5/model.py"),
    Path("ml/markers/center/dense_contract_v5/train_p1.py"),
)
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
ARTIFACT_THRESHOLD = 0.35
PARITY_TOLERANCE = 1e-5
MATCH_TOLERANCE = 5.0
HARD_NEGATIVE_TOLERANCE = 6.0
EPOCHS = 48
BATCH_SIZE = 8
LEARNING_RATE = 0.002


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _center_focal_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(1e-5, 1 - 1e-5)
    positive = target.eq(1)
    negative_weight = (1 - target).pow(4)
    positive_loss = -(prediction.log()) * (1 - prediction).pow(2) * positive
    negative_loss = -((1 - prediction).log()) * prediction.pow(2) * negative_weight * ~positive
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp(min=1)


def _artifact_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(1e-5, 1 - 1e-5)
    positive_weight = 3.0
    bce = -(
        positive_weight * target * prediction.log()
        + (1 - target) * (1 - prediction).log()
    ).mean()
    intersection = (prediction * target).sum()
    dice = 1 - ((2 * intersection + 1) / (prediction.sum() + target.sum() + 1))
    return bce + dice


def _centers(archive: dict[str, np.ndarray], index: int) -> tuple[tuple[float, float], ...]:
    count = int(archive["center_counts"][index])
    return tuple(
        (float(row[0]), float(row[1]))
        for row in archive["centers"][index, :count]
    )


def _hard_negatives(archive: dict[str, np.ndarray], index: int) -> tuple[tuple[str, float, float], ...]:
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


def _second_input(first_input: np.ndarray, first_output: np.ndarray) -> np.ndarray:
    result = np.asarray(first_input, dtype=np.float32).copy()
    result[:, 2] = np.maximum(result[:, 2], first_output[:, 2])
    return result


@torch.inference_mode()
def _torch_outputs(model: torch.nn.Module, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_input = torch.from_numpy(np.asarray(value, dtype=np.float32))
    first_output = model(first_input).cpu().numpy().astype(np.float32)
    second_input = _second_input(value, first_output)
    second_output = model(torch.from_numpy(second_input)).cpu().numpy().astype(np.float32)
    return first_output, second_input, second_output


def _onnx_outputs(session: ort.InferenceSession, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_output = np.asarray(
        session.run(["marker_heads"], {"image_and_masks": value})[0],
        dtype=np.float32,
    )
    second_input = _second_input(value, first_output)
    second_output = np.asarray(
        session.run(["marker_heads"], {"image_and_masks": second_input})[0],
        dtype=np.float32,
    )
    return first_output, second_input, second_output


def _evaluate(
    archive: dict[str, np.ndarray],
    outputs: list[tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, object]:
    scene_metrics = []
    exact_scene_count = 0
    prohibited = {kind: 0 for kind in PROHIBITED_KINDS}
    artifact_intersection = 0
    artifact_predicted = 0
    artifact_truth = 0
    marker_artifact_hits = 0
    for index, (first_output, second_output) in enumerate(outputs):
        first_input = archive["inputs"][index : index + 1]
        combined_artifact = np.maximum(first_input[0, 2], first_output[0, 2])
        detections = detect_heads(
            second_output,
            text_mask=first_input[0, 1],
            artifact_mask=combined_artifact,
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
        predicted_mask = first_output[0, 2] >= ARTIFACT_THRESHOLD
        truth_mask = archive["artifact_targets"][index, 0] >= 0.5
        artifact_intersection += int(np.logical_and(predicted_mask, truth_mask).sum())
        artifact_predicted += int(predicted_mask.sum())
        artifact_truth += int(truth_mask.sum())
        for x, y in _centers(archive, index):
            marker_artifact_hits += int(first_output[0, 2, round(y), round(x)] >= ARTIFACT_THRESHOLD)
    aggregate = aggregate_scene_metrics(scene_metrics, MATCH_TOLERANCE)
    artifact_precision = artifact_intersection / artifact_predicted if artifact_predicted else 0.0
    artifact_recall = artifact_intersection / artifact_truth if artifact_truth else 1.0
    passed = (
        exact_scene_count == len(outputs)
        and aggregate.false_positives == 0
        and aggregate.false_negatives == 0
        and aggregate.duplicate_count == 0
        and not any(prohibited.values())
        and artifact_precision >= 0.90
        and artifact_recall >= 0.95
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
    }


def _passing_window(comparisons: list[dict[str, object]]) -> list[float]:
    best: list[float] = []
    current: list[float] = []
    for item in comparisons:
        if item["passed"]:
            current.append(float(item["threshold"]))
            if len(current) > len(best):
                best = current.copy()
        else:
            current = []
    return best


def _export(model: torch.nn.Module, sample: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model,
        (sample,),
        path,
        input_names=["image_and_masks"],
        output_names=["marker_heads"],
        dynamic_axes={
            "image_and_masks": {0: "batch", 2: "height", 3: "width"},
            "marker_heads": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def _execute_candidate(
    output_dir: Path,
    authorization: TrainingAuthorization,
    progress: dict[str, object],
) -> dict[str, object]:
    started = float(progress["started"])
    progress["phase"] = "input_validation"
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
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
            radius_loss = functional.smooth_l1_loss(
                heads[:, 1:2][radius_mask],
                radius_target[radius_mask],
            )
            artifact_loss = _artifact_loss(heads[:, 2:3], artifact_target)
            marker_pixels = center_target >= 0.1
            marker_clear_loss = functional.binary_cross_entropy(
                heads[:, 2:3][marker_pixels],
                torch.zeros_like(heads[:, 2:3][marker_pixels]),
            )
            loss = center_loss + (0.15 * radius_loss) + (0.65 * artifact_loss) + marker_clear_loss
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
                }
            )
    model.eval()
    progress["phase"] = "export"
    checkpoint_path = output_dir / "marker-center-dense-contract-v5-p1.pt"
    onnx_path = output_dir / "marker-center-dense-contract-v5-p1.onnx"
    sample = torch.from_numpy(validation["inputs"][0:1])
    _export(model, sample, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    progress["phase"] = "selection"
    torch_outputs: list[tuple[np.ndarray, np.ndarray]] = []
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
        torch_outputs.append((torch_first, torch_second))
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
    report = {
        "schema": "graphreader.marker-center-dense-contract-candidate.v5",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "selection_passed" if selection_passed else "failed_selection",
        "selection_gate_passed": selection_passed,
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
    return report


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Dense-contract V5 output exists: {output_dir}")
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
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.output.resolve())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
