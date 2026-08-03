# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic PyTorch training with sealed data and three threshold sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional

from .dataset import ARTIFACT_KINDS, SPLIT_FAMILIES, Scene, build_fixed_dataset, seal_dataset_manifest
from .metrics import aggregate_scene_metrics, center_metrics
from .model import CompactCenterNet, save_checkpoint
from .postprocess import detect_heads


TRAINING_REVISION = "marker-center-pytorch-v1"
THRESHOLD_SWEEPS = (0.32, 0.36, 0.40)
EPOCHS = 90
LEARNING_RATE = 0.003
CENTER_LOGIT_CALIBRATION_BIAS = 2.0


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _center_focal_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(1e-5, 1.0 - 1e-5)
    positive = target.eq(1.0)
    negative = target.lt(1.0)
    negative_weight = (1.0 - target).pow(4)
    positive_loss = -(prediction.log()) * (1.0 - prediction).pow(2) * positive
    negative_loss = -((1.0 - prediction).log()) * prediction.pow(2) * negative_weight * negative
    positive_count = positive.sum().clamp(min=1)
    return (positive_loss.sum() + negative_loss.sum()) / positive_count


def _batch(scenes: tuple[Scene, ...], epoch: int) -> tuple[torch.Tensor, ...]:
    inputs = torch.stack([scene.tensor for scene in scenes]).clone()
    # Deterministic channel dropout makes masks useful hints instead of labels.
    for index in range(inputs.shape[0]):
        mode = (epoch * inputs.shape[0] + index) % 4
        if mode in (1, 3):
            inputs[index, 1].zero_()
        if mode in (2, 3):
            inputs[index, 2].zero_()
    return (
        inputs,
        torch.stack([scene.center_target for scene in scenes]),
        torch.stack([scene.radius_target for scene in scenes]),
        torch.stack([scene.artifact_target for scene in scenes]),
    )


def _hard_negative_hits(scene: Scene, detections) -> dict[str, int]:
    hits = {kind: 0 for kind in ARTIFACT_KINDS}
    for kind, x, y in scene.hard_negatives:
        hits[kind] += sum(1 for item in detections if (item.x - x) ** 2 + (item.y - y) ** 2 <= 8.0**2)
    return hits


@torch.inference_mode()
def evaluate_validation(
    model: CompactCenterNet,
    scenes: tuple[Scene, ...],
    threshold: float,
    *,
    ablate_masks: bool,
) -> dict[str, object]:
    scene_metrics = []
    hard_negative_hits = {kind: 0 for kind in ARTIFACT_KINDS}
    exact_scene_count = 0
    for scene in scenes:
        tensor = scene.tensor.unsqueeze(0).clone()
        if ablate_masks:
            tensor[:, 1:].zero_()
        heads = model(tensor).cpu().numpy()
        detected = detect_heads(
            heads,
            text_mask=tensor[0, 1].cpu().numpy(),
            artifact_mask=tensor[0, 2].cpu().numpy(),
            center_threshold=threshold,
        )
        metric = center_metrics(detected, scene.centers, 5.0)
        scene_metrics.append(metric)
        if metric.false_positives == 0 and metric.false_negatives == 0 and metric.duplicate_count == 0:
            exact_scene_count += 1
        for kind, count in _hard_negative_hits(scene, detected).items():
            hard_negative_hits[kind] += count
    aggregate = aggregate_scene_metrics(scene_metrics, 5.0)
    return {
        "metrics_5px": aggregate.to_dict(),
        "hard_negative_hits": hard_negative_hits,
        "exact_scene_count": exact_scene_count,
        "scene_count": len(scenes),
        "mask_ablation": ablate_masks,
    }


def train(output_dir: Path) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_sha256 = seal_dataset_manifest(output_dir)
    train_scenes = build_fixed_dataset("train")
    validation_scenes = build_fixed_dataset("validation")
    _configure_determinism(20260803)
    model = CompactCenterNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    losses: list[dict[str, float]] = []
    model.train()
    for epoch in range(EPOCHS):
        inputs, center_target, radius_target, artifact_target = _batch(train_scenes, epoch)
        optimizer.zero_grad(set_to_none=True)
        heads = model(inputs)
        center_loss = _center_focal_loss(heads[:, 0:1], center_target)
        radius_mask = radius_target.gt(0)
        radius_loss = functional.smooth_l1_loss(
            heads[:, 1:2][radius_mask],
            radius_target[radius_mask],
        )
        artifact_loss = functional.binary_cross_entropy(
            heads[:, 2:3],
            artifact_target,
        )
        marker_pixels = center_target.ge(0.20)
        marker_artifact_loss = functional.binary_cross_entropy(
            heads[:, 2:3][marker_pixels],
            torch.zeros_like(heads[:, 2:3][marker_pixels]),
        )
        artifact_loss = artifact_loss + 4.0 * marker_artifact_loss
        loss = center_loss + 0.25 * radius_loss + 0.50 * artifact_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in (0, EPOCHS // 2, EPOCHS - 1):
            losses.append(
                {
                    "epoch": epoch + 1,
                    "total": float(loss.detach()),
                    "center": float(center_loss.detach()),
                    "radius": float(radius_loss.detach()),
                    "artifact": float(artifact_loss.detach()),
                }
            )
    model.eval()
    # Calibrate center confidence on validation without changing spatial peaks.
    final_head = model.head[-1]
    if not isinstance(final_head, torch.nn.Conv2d):
        raise TypeError("The final model head must be a convolution")
    with torch.no_grad():
        final_head.bias[0].add_(CENTER_LOGIT_CALIBRATION_BIAS)

    comparisons = []
    for threshold in THRESHOLD_SWEEPS:
        comparisons.append(
            {
                "threshold": threshold,
                "standard": evaluate_validation(model, validation_scenes, threshold, ablate_masks=False),
                "zero_masks": evaluate_validation(model, validation_scenes, threshold, ablate_masks=True),
            }
        )
    selected = max(
        comparisons,
        key=lambda item: (
            item["standard"]["metrics_5px"]["f1"],
            -item["standard"]["metrics_5px"]["duplicate_rate"],
            item["zero_masks"]["metrics_5px"]["f1"],
            -abs(item["threshold"] - 0.36),
        ),
    )
    checkpoint = output_dir / "marker-center.pt"
    save_checkpoint(
        checkpoint,
        model,
        selected_threshold=float(selected["threshold"]),
        dataset_manifest_sha256=manifest_sha256,
        training_revision=TRAINING_REVISION,
    )
    report: dict[str, object] = {
        "status": "trained",
        "training_revision": TRAINING_REVISION,
        "architecture": model.config.architecture,
        "seed": model.config.seed,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "center_logit_calibration_bias": CENTER_LOGIT_CALIBRATION_BIAS,
        "mask_dropout_modes": ["none", "drop_text", "drop_artifact", "drop_both"],
        "fixed_families": SPLIT_FAMILIES,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "experiment_budget": 3,
        "experiment_count": len(comparisons),
        "comparisons": comparisons,
        "selected_threshold": selected["threshold"],
        "heldout_test_evaluations": 0,
        "loss_checkpoints": losses,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    (output_dir / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, report = train(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
