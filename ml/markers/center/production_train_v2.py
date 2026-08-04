# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single preregistered marker-center repair with mask-consensus hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as functional

from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_fixed_dataset, seal_dataset_manifest
from .export import export_onnx
from .model import CompactCenterNet, ModelConfig, save_checkpoint
from .production_train import _batch, _robust_validation
from .train import _center_focal_loss, _configure_determinism, evaluate_validation


TASK = "marker-center"
TRAINING_REVISION = "marker-center-production-repair-v2"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/training/production-repair-v2-p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/production_train_v2.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/markers/center/production_train.py"),
    Path("ml/markers/center/dataset.py"),
    Path("ml/markers/center/model.py"),
    Path("ml/markers/center/train.py"),
    Path("ml/markers/center/export.py"),
)
THRESHOLDS = (0.28, 0.32, 0.36, 0.40)


def _selection_score(row: dict[str, object]) -> tuple[float, ...]:
    standard = row["standard"]
    robust = row["robust"]
    return (
        float(standard["exact_scene_count"]),
        float(robust["exact_scene_count"]),
        float(standard["metrics_5px"]["f1"]),
        float(robust["metrics_5px"]["f1"]),
        -float(standard["metrics_5px"]["duplicate_rate"] + robust["metrics_5px"]["duplicate_rate"]),
        -float(sum(standard["hard_negative_hits"].values()) + sum(robust["hard_negative_hits"].values())),
    )


def train_candidate(output_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    authorization = acquire_training_candidate(
        repo_root,
        task=TASK,
        revision=TRAINING_REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    started = time.perf_counter()
    config = json.loads((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path, manifest_sha256 = seal_dataset_manifest(output_dir)
    if manifest_sha256 != config["selection_dataset_manifest_sha256"]:
        raise RuntimeError("Selection dataset manifest differs from preregistration")
    seed = int(config["seed"])
    _configure_determinism(seed)
    model = CompactCenterNet(ModelConfig(seed=seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    training_scenes = build_fixed_dataset("train")
    validation_scenes = build_fixed_dataset("validation")
    loss_checkpoints = []
    model.train()
    for epoch in range(int(config["epochs"])):
        inputs, center_target, radius_target, artifact_target = _batch(training_scenes, epoch, "mixed")
        optimizer.zero_grad(set_to_none=True)
        heads = model(inputs)
        center_loss = _center_focal_loss(heads[:, 0:1], center_target)
        radius_mask = radius_target.gt(0)
        radius_loss = functional.smooth_l1_loss(heads[:, 1:2][radius_mask], radius_target[radius_mask])
        artifact_loss = functional.binary_cross_entropy(heads[:, 2:3], artifact_target)
        marker_pixels = center_target.ge(0.20)
        marker_artifact_loss = functional.binary_cross_entropy(
            heads[:, 2:3][marker_pixels],
            torch.zeros_like(heads[:, 2:3][marker_pixels]),
        )
        consensus = torch.maximum(inputs[:, 1:2], inputs[:, 2:3])
        hard_negative_pixels = consensus.ge(float(config["changes"]["mask_consensus_threshold"]))
        if not torch.any(hard_negative_pixels):
            raise RuntimeError("Preregistered mask-consensus hard-negative set is empty")
        hard_negative_artifact_loss = functional.binary_cross_entropy(
            heads[:, 2:3][hard_negative_pixels],
            torch.ones_like(heads[:, 2:3][hard_negative_pixels]),
        )
        hard_negative_center_loss = functional.binary_cross_entropy(
            heads[:, 0:1][hard_negative_pixels],
            torch.zeros_like(heads[:, 0:1][hard_negative_pixels]),
        )
        loss = (
            center_loss
            + 0.25 * radius_loss
            + 0.50 * artifact_loss
            + 2.0 * marker_artifact_loss
            + float(config["changes"]["hard_negative_artifact_weight"]) * hard_negative_artifact_loss
            + float(config["changes"]["hard_negative_center_suppression_weight"]) * hard_negative_center_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in (0, int(config["epochs"]) // 2, int(config["epochs"]) - 1):
            loss_checkpoints.append(
                {
                    "epoch": epoch + 1,
                    "total": float(loss.detach()),
                    "center": float(center_loss.detach()),
                    "radius": float(radius_loss.detach()),
                    "artifact": float(artifact_loss.detach()),
                    "hard_negative_artifact": float(hard_negative_artifact_loss.detach()),
                    "hard_negative_center": float(hard_negative_center_loss.detach()),
                }
            )
    model.eval()
    final_head = model.head[-1]
    if not isinstance(final_head, torch.nn.Conv2d):
        raise TypeError("The final model head must be a convolution")
    with torch.no_grad():
        final_head.bias[0].add_(2.0)
    robust_validation = _robust_validation(validation_scenes, "mixed")
    threshold_rows = [
        {
            "threshold": threshold,
            "standard": evaluate_validation(model, validation_scenes, threshold, ablate_masks=False),
            "robust": evaluate_validation(model, robust_validation, threshold, ablate_masks=False),
            "zero_masks": evaluate_validation(model, validation_scenes, threshold, ablate_masks=True),
        }
        for threshold in THRESHOLDS
    ]
    selected = max(threshold_rows, key=_selection_score)
    candidate_dir = output_dir / CANDIDATE_ID
    checkpoint = candidate_dir / "marker-center.pt"
    save_checkpoint(
        checkpoint,
        model,
        selected_threshold=float(selected["threshold"]),
        dataset_manifest_sha256=manifest_sha256,
        training_revision=TRAINING_REVISION,
    )
    onnx_path = candidate_dir / "marker-center.onnx"
    parity_path = candidate_dir / "onnx-parity.json"
    parity = export_onnx(checkpoint, onnx_path, parity_path)
    standard = selected["standard"]
    robust = selected["robust"]
    gates = {
        "standard_exact": standard["exact_scene_count"] == standard["scene_count"],
        "robust_exact": robust["exact_scene_count"] == robust["scene_count"],
        "zero_duplicates": standard["metrics_5px"]["duplicate_rate"] == 0 and robust["metrics_5px"]["duplicate_rate"] == 0,
        "zero_hard_negative_hits": not any(standard["hard_negative_hits"].values()) and not any(robust["hard_negative_hits"].values()),
        "onnx_parity": parity["status"] == "pass",
    }
    report: dict[str, object] = {
        "status": "selected" if all(gates.values()) else "fail",
        "release_eligible": False,
        "task": TASK,
        "training_revision": TRAINING_REVISION,
        "candidate_id": CANDIDATE_ID,
        "experiment_budget": 3,
        "experiment_ordinal": 1,
        "public_gate_evaluations": 0,
        "private_data": False,
        "candidate_config_sha256": hashlib.sha256((repo_root / CONFIG_PATH).read_bytes()).hexdigest(),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "training_seal_binding": authorization.binding,
        "seed": seed,
        "mask_consensus": config["changes"],
        "selected_threshold": selected["threshold"],
        "standard": standard,
        "robust": robust,
        "zero_masks": selected["zero_masks"],
        "threshold_comparisons": threshold_rows,
        "gate_results": gates,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx": str(onnx_path),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "onnx_parity": parity,
        "loss_checkpoints": loss_checkpoints,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    report_path = output_dir / "production-training-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    complete_training_candidate(
        authorization,
        status=str(report["status"]),
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    )
    return checkpoint, onnx_path, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, _, report = train_candidate(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
