# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Three-candidate, validation-only production repair for marker centers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as functional

from .dataset import Scene, build_fixed_dataset, seal_dataset_manifest
from .model import CompactCenterNet, ModelConfig, save_checkpoint
from .train import _center_focal_loss, _configure_determinism, evaluate_validation
from ml.markers.training_budget import require_training_budget


TRAINING_REVISION = "marker-center-production-repair-v1"
THRESHOLDS = (0.28, 0.32, 0.36, 0.40)
EXPERIMENTS = (
    {"id": "P1", "epochs": 110, "learning_rate": 0.0025, "weight_decay": 0.00001, "robustness": "contrast"},
    {"id": "P2", "epochs": 130, "learning_rate": 0.0020, "weight_decay": 0.00001, "robustness": "resample"},
    {"id": "P3", "epochs": 150, "learning_rate": 0.0015, "weight_decay": 0.000005, "robustness": "mixed"},
)
CANDIDATE_SEEDS = (20260804, 20260805, 20260806)


def _photometric_ink(ink: torch.Tensor, epoch: int, mode: str) -> torch.Tensor:
    result = ink
    if mode in ("contrast", "mixed"):
        scale = 0.88 + 0.04 * (epoch % 7)
        result = torch.clamp(result * scale, 0.0, 1.0)
    if mode in ("resample", "mixed") and epoch % 2:
        reduced = functional.interpolate(result, size=(112, 112), mode="bilinear", align_corners=False)
        result = functional.interpolate(reduced, size=(128, 128), mode="bilinear", align_corners=False)
    return result


def _batch(scenes: tuple[Scene, ...], epoch: int, mode: str) -> tuple[torch.Tensor, ...]:
    inputs = torch.stack([scene.tensor for scene in scenes]).clone()
    inputs[:, 0:1] = _photometric_ink(inputs[:, 0:1], epoch, mode)
    return (
        inputs,
        torch.stack([scene.center_target for scene in scenes]),
        torch.stack([scene.radius_target for scene in scenes]),
        torch.stack([scene.artifact_target for scene in scenes]),
    )


def _robust_validation(scenes: tuple[Scene, ...], mode: str) -> tuple[Scene, ...]:
    result = []
    for index, scene in enumerate(scenes):
        tensor = scene.tensor.clone()
        tensor[0:1] = _photometric_ink(tensor[0:1].unsqueeze(0), index + 1, mode).squeeze(0)
        result.append(
            Scene(
                scene_id=f"{scene.scene_id}-robust-{mode}",
                split=scene.split,
                family=f"{scene.family}_robust_{mode}",
                degradation=f"selection_only_{mode}",
                seed=scene.seed,
                tensor=tensor,
                center_target=scene.center_target,
                radius_target=scene.radius_target,
                artifact_target=scene.artifact_target,
                centers=scene.centers,
                radii=scene.radii,
                hard_negatives=scene.hard_negatives,
            )
        )
    return tuple(result)


def _score(row: dict[str, object]) -> tuple[float, ...]:
    standard = row["standard"]
    robust = row["robust"]
    return (
        float(robust["exact_scene_count"]),
        float(robust["metrics_5px"]["f1"]),
        float(standard["exact_scene_count"]),
        float(standard["metrics_5px"]["f1"]),
        -float(robust["metrics_5px"]["duplicate_rate"]),
        -float(sum(robust["hard_negative_hits"].values())),
    )


def train_candidates(output_dir: Path) -> tuple[Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    require_training_budget(repo_root, task="marker-center", revision=TRAINING_REVISION)
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_sha256 = seal_dataset_manifest(output_dir)
    train_scenes = build_fixed_dataset("train")
    validation_scenes = build_fixed_dataset("validation")
    comparisons = []
    selected: tuple[dict[str, object], Path] | None = None
    for index, spec in enumerate(EXPERIMENTS):
        candidate_seed = CANDIDATE_SEEDS[index]
        _configure_determinism(candidate_seed)
        model = CompactCenterNet(ModelConfig(seed=candidate_seed))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        loss_checkpoints = []
        model.train()
        for epoch in range(int(spec["epochs"])):
            inputs, center_target, radius_target, artifact_target = _batch(train_scenes, epoch, str(spec["robustness"]))
            optimizer.zero_grad(set_to_none=True)
            heads = model(inputs)
            center_loss = _center_focal_loss(heads[:, 0:1], center_target)
            radius_mask = radius_target.gt(0)
            radius_loss = functional.smooth_l1_loss(heads[:, 1:2][radius_mask], radius_target[radius_mask])
            artifact_loss = functional.binary_cross_entropy(heads[:, 2:3], artifact_target)
            marker_pixels = center_target.ge(0.20)
            artifact_loss = artifact_loss + 4.0 * functional.binary_cross_entropy(
                heads[:, 2:3][marker_pixels],
                torch.zeros_like(heads[:, 2:3][marker_pixels]),
            )
            loss = center_loss + 0.25 * radius_loss + 0.50 * artifact_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if epoch in (0, int(spec["epochs"]) // 2, int(spec["epochs"]) - 1):
                loss_checkpoints.append(
                    {
                        "epoch": epoch + 1,
                        "total": float(loss.detach()),
                        "center": float(center_loss.detach()),
                        "radius": float(radius_loss.detach()),
                        "artifact": float(artifact_loss.detach()),
                    }
                )
        model.eval()
        final_head = model.head[-1]
        if not isinstance(final_head, torch.nn.Conv2d):
            raise TypeError("The final model head must be a convolution")
        with torch.no_grad():
            final_head.bias[0].add_(2.0)
        robustness_validation = _robust_validation(validation_scenes, str(spec["robustness"]))
        threshold_rows = []
        for threshold in THRESHOLDS:
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "standard": evaluate_validation(model, validation_scenes, threshold, ablate_masks=False),
                    "robust": evaluate_validation(model, robustness_validation, threshold, ablate_masks=False),
                    "zero_masks": evaluate_validation(model, validation_scenes, threshold, ablate_masks=True),
                }
            )
        selected_threshold = max(threshold_rows, key=_score)
        candidate_dir = output_dir / str(spec["id"])
        checkpoint = candidate_dir / "marker-center.pt"
        save_checkpoint(
            checkpoint,
            model,
            selected_threshold=float(selected_threshold["threshold"]),
            dataset_manifest_sha256=manifest_sha256,
            training_revision=TRAINING_REVISION,
        )
        standard = selected_threshold["standard"]
        robust = selected_threshold["robust"]
        gate_results = {
            "standard_exact": standard["exact_scene_count"] == standard["scene_count"],
            "robust_exact": robust["exact_scene_count"] == robust["scene_count"],
            "zero_duplicates": standard["metrics_5px"]["duplicate_rate"] == 0 and robust["metrics_5px"]["duplicate_rate"] == 0,
            "zero_hard_negative_hits": not any(standard["hard_negative_hits"].values()) and not any(robust["hard_negative_hits"].values()),
        }
        row: dict[str, object] = {
            **spec,
            "global_seed": candidate_seed,
            "model_initialization_seed": model.config.seed,
            "mask_channel_augmentation": "none",
            "selected_threshold": selected_threshold["threshold"],
            "standard": standard,
            "robust": robust,
            "zero_masks": selected_threshold["zero_masks"],
            "threshold_comparisons": threshold_rows,
            "gate_results": gate_results,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "loss_checkpoints": loss_checkpoints,
        }
        comparisons.append(row)
        if selected is None or _score(row) > _score(selected[0]):
            selected = (row, checkpoint)
    if selected is None:
        raise RuntimeError("No candidate was trained")
    selected_row, selected_checkpoint = selected
    report: dict[str, object] = {
        "status": "selected" if all(selected_row["gate_results"].values()) else "fail",
        "release_eligible": False,
        "training_revision": TRAINING_REVISION,
        "experiment_budget": 3,
        "experiment_count": len(comparisons),
        "selection_data": "existing procedural validation plus deterministic photometric validation variants",
        "public_gate_evaluations": 0,
        "private_data": False,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "comparisons": comparisons,
        "selected_candidate": selected_row["id"],
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_row["checkpoint_sha256"],
        "selected_gate_results": selected_row["gate_results"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    report_path = output_dir / "production-training-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selected_checkpoint, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, report = train_candidates(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
