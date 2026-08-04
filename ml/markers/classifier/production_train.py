# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Three-candidate, validation-only production repair for marker classification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from ml.markers.training_budget import require_training_budget
from .dataset import SHAPE_NAMES, build_fixed_dataset, seal_dataset_manifest
from .metrics import classification_metrics
from .model import ClassifierConfig, CompactMarkerClassifier, save_checkpoint
from .train import (
    LOCAL_FILL_MACRO_F1_GATE,
    LOCAL_SHAPE_MACRO_F1_GATE,
    _stack,
    _train_one_epoch,
    collect_outputs,
    configure_determinism,
    fit_temperature,
    summarize_outputs,
)


TRAINING_REVISION = "marker-classifier-production-repair-v1"
EXPERIMENTS = (
    {"id": "P1", "epochs": 60, "learning_rate": 0.0025, "weight_decay": 0.0001},
    {"id": "P2", "epochs": 80, "learning_rate": 0.0020, "weight_decay": 0.0001},
    {"id": "P3", "epochs": 100, "learning_rate": 0.0015, "weight_decay": 0.00005},
)
CANDIDATE_SEEDS = (20260804, 20260805, 20260806)


def _minority_min_f1(outputs: dict[str, torch.Tensor], shape_temperature: float) -> float:
    marker_mask = outputs["artifact_targets"].lt(0.5)
    probabilities = torch.softmax(outputs["shape_logits"][marker_mask] / shape_temperature, dim=1).numpy()
    targets = outputs["shape_targets"][marker_mask].numpy()
    detail = classification_metrics(probabilities, targets, len(SHAPE_NAMES))
    return min(detail.per_class_f1[SHAPE_NAMES.index(name)] for name in ("star", "asterisk", "cross"))


def train_candidates(output_dir: Path) -> tuple[Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[3]
    require_training_budget(repo_root, task="marker-classifier", revision=TRAINING_REVISION)
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_sha256 = seal_dataset_manifest(output_dir)
    training_samples = build_fixed_dataset("train")
    validation_samples = build_fixed_dataset("validation")
    tensors = _stack(training_samples)
    comparisons = []
    selected: tuple[dict[str, object], Path] | None = None
    for index, spec in enumerate(EXPERIMENTS):
        candidate_seed = CANDIDATE_SEEDS[index]
        configure_determinism(candidate_seed)
        model = CompactMarkerClassifier(ClassifierConfig(seed=candidate_seed))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        loss_checkpoints = []
        for epoch in range(int(spec["epochs"])):
            losses = _train_one_epoch(model, optimizer, tensors, epoch, seed=candidate_seed)
            if epoch in (0, int(spec["epochs"]) // 2, int(spec["epochs"]) - 1):
                loss_checkpoints.append({"epoch": epoch + 1, **losses})
        outputs = collect_outputs(model, validation_samples)
        marker_mask = outputs["artifact_targets"].lt(0.5)
        shape_temperature = fit_temperature(outputs["shape_logits"][marker_mask], outputs["shape_targets"][marker_mask])
        fill_temperature = fit_temperature(outputs["fill_logits"][marker_mask], outputs["fill_targets"][marker_mask])
        metrics = summarize_outputs(outputs, shape_temperature, fill_temperature)
        minority_min_f1 = _minority_min_f1(outputs, shape_temperature)
        gate_results = {
            "shape_macro_f1": metrics["shape"]["macro_f1"] >= LOCAL_SHAPE_MACRO_F1_GATE,
            "fill_macro_f1": metrics["fill"]["macro_f1"] >= LOCAL_FILL_MACRO_F1_GATE,
            "artifact_f1": metrics["artifact"]["f1"] == 1.0,
            "minority_min_f1": minority_min_f1 >= 0.90,
        }
        candidate_dir = output_dir / str(spec["id"])
        checkpoint = candidate_dir / "marker-classifier.pt"
        save_checkpoint(
            checkpoint,
            model,
            dataset_manifest_sha256=manifest_sha256,
            shape_temperature=shape_temperature,
            fill_temperature=fill_temperature,
            training_revision=TRAINING_REVISION,
        )
        row: dict[str, object] = {
            **spec,
            "global_seed": candidate_seed,
            "model_initialization_seed": model.config.seed,
            "batch_order_seed": candidate_seed,
            "validation_metrics": metrics,
            "minority_min_f1": minority_min_f1,
            "gate_results": gate_results,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "shape_temperature": shape_temperature,
            "fill_temperature": fill_temperature,
            "loss_checkpoints": loss_checkpoints,
        }
        comparisons.append(row)
        if selected is None or (
            metrics["shape"]["macro_f1"],
            metrics["fill"]["macro_f1"],
            metrics["artifact"]["f1"],
        ) > (
            selected[0]["validation_metrics"]["shape"]["macro_f1"],
            selected[0]["validation_metrics"]["fill"]["macro_f1"],
            selected[0]["validation_metrics"]["artifact"]["f1"],
        ):
            selected = (row, checkpoint)
    if selected is None:
        raise RuntimeError("No candidate was trained")
    selected_row, selected_checkpoint = selected
    selected_pass = all(selected_row["gate_results"].values())
    report: dict[str, object] = {
        "status": "selected" if selected_pass else "fail",
        "release_eligible": False,
        "training_revision": TRAINING_REVISION,
        "experiment_budget": 3,
        "experiment_count": len(comparisons),
        "selection_data": "existing procedural validation split only",
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
