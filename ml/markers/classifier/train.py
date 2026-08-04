# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic selection-only training for the marker patch classifier."""

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

from .dataset import FILL_NAMES, SHAPE_NAMES, PatchSample, SPLIT_FAMILIES, SPLIT_TEMPLATES, build_fixed_dataset, seal_dataset_manifest
from .metrics import binary_metrics, classification_metrics, embedding_retrieval_accuracy, fit_temperature, supervised_embedding_loss
from .model import ClassifierConfig, CompactMarkerClassifier, save_checkpoint


TRAINING_REVISION = "marker-classifier-pytorch-v2"
SEED = 20260803
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.003
EXPERIMENT_BUDGET = 3
LOCAL_SHAPE_MACRO_F1_GATE = 0.90
LOCAL_FILL_MACRO_F1_GATE = 0.90


def configure_determinism(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _stack(samples: tuple[PatchSample, ...]) -> tuple[torch.Tensor, ...]:
    return (
        torch.stack([sample.tensor for sample in samples]),
        torch.tensor([sample.shape_index for sample in samples], dtype=torch.long),
        torch.tensor([sample.fill_index for sample in samples], dtype=torch.long),
        torch.tensor([sample.artifact for sample in samples], dtype=torch.float32),
    )


def _marker_identity(shape: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    return shape * len(FILL_NAMES) + fill


def _augment_batch(inputs: torch.Tensor, epoch: int, order: torch.Tensor) -> torch.Tensor:
    padded = functional.pad(inputs, (3, 3, 3, 3), value=0.0)
    augmented = torch.empty_like(inputs)
    for batch_index, source_index in enumerate(order.tolist()):
        dx = (source_index + epoch * 2) % 7
        dy = (source_index * 3 + epoch) % 7
        augmented[batch_index] = padded[batch_index, :, dy : dy + 32, dx : dx + 32]
    contrast = 0.92 + 0.04 * (epoch % 5)
    return torch.clamp(augmented * contrast, 0.0, 1.0)


def _train_one_epoch(
    model: CompactMarkerClassifier,
    optimizer: torch.optim.Optimizer,
    tensors: tuple[torch.Tensor, ...],
    epoch: int,
    *,
    seed: int = SEED,
) -> dict[str, float]:
    inputs, shapes, fills, artifacts = tensors
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    order = torch.randperm(inputs.shape[0], generator=generator)
    totals = {"total": 0.0, "shape": 0.0, "fill": 0.0, "artifact": 0.0, "embedding": 0.0}
    batches = 0
    model.train()
    for start in range(0, len(order), BATCH_SIZE):
        index = order[start : start + BATCH_SIZE]
        batch_inputs = _augment_batch(inputs[index], epoch, index)
        batch_shape = shapes[index]
        batch_fill = fills[index]
        batch_artifact = artifacts[index]
        marker_mask = batch_artifact.lt(0.5)
        optimizer.zero_grad(set_to_none=True)
        shape_logits, fill_logits, artifact_logit, embedding = model(batch_inputs)
        shape_loss = functional.cross_entropy(shape_logits[marker_mask], batch_shape[marker_mask])
        fill_loss = functional.cross_entropy(fill_logits[marker_mask], batch_fill[marker_mask])
        artifact_loss = functional.binary_cross_entropy_with_logits(artifact_logit[:, 0], batch_artifact)
        embedding_loss = supervised_embedding_loss(
            embedding[marker_mask],
            _marker_identity(batch_shape[marker_mask], batch_fill[marker_mask]),
        )
        loss = shape_loss + 0.75 * fill_loss + 0.65 * artifact_loss + 0.08 * embedding_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for name, value in (
            ("total", loss), ("shape", shape_loss), ("fill", fill_loss),
            ("artifact", artifact_loss), ("embedding", embedding_loss),
        ):
            totals[name] += float(value.detach())
        batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


@torch.inference_mode()
def collect_outputs(model: CompactMarkerClassifier, samples: tuple[PatchSample, ...]) -> dict[str, torch.Tensor]:
    model.eval()
    tensors = _stack(samples)
    chunks: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, len(samples), BATCH_SIZE):
        chunks.append(tuple(value.cpu() for value in model(tensors[0][start : start + BATCH_SIZE])))
    return {
        "shape_logits": torch.cat([chunk[0] for chunk in chunks]),
        "fill_logits": torch.cat([chunk[1] for chunk in chunks]),
        "artifact_logits": torch.cat([chunk[2] for chunk in chunks])[:, 0],
        "embedding": torch.cat([chunk[3] for chunk in chunks]),
        "shape_targets": tensors[1],
        "fill_targets": tensors[2],
        "artifact_targets": tensors[3],
    }


def summarize_outputs(outputs: dict[str, torch.Tensor], shape_temperature: float, fill_temperature: float) -> dict[str, object]:
    marker_mask = outputs["artifact_targets"].lt(0.5)
    shape_probabilities = functional.softmax(outputs["shape_logits"][marker_mask] / shape_temperature, dim=1).numpy()
    fill_probabilities = functional.softmax(outputs["fill_logits"][marker_mask] / fill_temperature, dim=1).numpy()
    shape_targets = outputs["shape_targets"][marker_mask].numpy()
    fill_targets = outputs["fill_targets"][marker_mask].numpy()
    artifact_probabilities = torch.sigmoid(outputs["artifact_logits"]).numpy()
    identity = shape_targets * len(FILL_NAMES) + fill_targets
    shape = classification_metrics(shape_probabilities, shape_targets, len(SHAPE_NAMES))
    fill = classification_metrics(fill_probabilities, fill_targets, len(FILL_NAMES))
    return {
        "shape": shape.to_dict(),
        "fill": fill.to_dict(),
        "artifact": binary_metrics(artifact_probabilities, outputs["artifact_targets"].numpy()),
        "embedding_top1_retrieval_accuracy": embedding_retrieval_accuracy(outputs["embedding"][marker_mask].numpy(), identity),
        "marker_count": int(marker_mask.sum()),
        "artifact_count": int((~marker_mask).sum()),
    }


def train(output_dir: Path) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_sha256 = seal_dataset_manifest(output_dir)
    training_samples = build_fixed_dataset("train")
    validation_samples = build_fixed_dataset("validation")
    configure_determinism()
    model = CompactMarkerClassifier(ClassifierConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    training_tensors = _stack(training_samples)
    checkpoints = []
    for epoch in range(EPOCHS):
        losses = _train_one_epoch(model, optimizer, training_tensors, epoch)
        if epoch in (0, EPOCHS // 2, EPOCHS - 1):
            checkpoints.append({"epoch": epoch + 1, **losses})

    validation_outputs = collect_outputs(model, validation_samples)
    marker_mask = validation_outputs["artifact_targets"].lt(0.5)
    shape_temperature = fit_temperature(validation_outputs["shape_logits"][marker_mask], validation_outputs["shape_targets"][marker_mask])
    fill_temperature = fit_temperature(validation_outputs["fill_logits"][marker_mask], validation_outputs["fill_targets"][marker_mask])
    validation_metrics = summarize_outputs(validation_outputs, shape_temperature, fill_temperature)
    checkpoint = output_dir / "marker-classifier.pt"
    save_checkpoint(
        checkpoint,
        model,
        dataset_manifest_sha256=manifest_sha256,
        shape_temperature=shape_temperature,
        fill_temperature=fill_temperature,
        training_revision=TRAINING_REVISION,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    report: dict[str, object] = {
        "status": "selected",
        "training_revision": TRAINING_REVISION,
        "architecture": model.config.architecture,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "experiment_budget": EXPERIMENT_BUDGET,
        "experiment_count": 3,
        "experiment_selection_basis": "E3 compact spatial CNN selected after E1 and E2 failed validation; held-out remained sealed",
        "experiment_comparison": [
            {
                "id": "E1",
                "architecture": "compact-depthwise-global-pooling-patch-classifier-v1",
                "validation_shape_macro_f1": 0.1631174517203929,
                "validation_fill_macro_f1": 0.6105112042820607,
                "decision": "rejected because global pooling discarded shape geometry",
            },
            {
                "id": "E2",
                "architecture": "compact-depthwise-spatial-patch-classifier-v2",
                "validation_shape_macro_f1": 0.5015802147945635,
                "validation_fill_macro_f1": 0.6830065359477123,
                "decision": "rejected because translation-sensitive projection did not generalize",
            },
            {
                "id": "E3",
                "architecture": model.config.architecture,
                "validation_shape_macro_f1": validation_metrics["shape"]["macro_f1"],
                "validation_fill_macro_f1": validation_metrics["fill"]["macro_f1"],
                "decision": "selected using validation only; experiment budget exhausted",
            },
        ],
        "training_augmentation": "deterministic integer translation and contrast schedule",
        "local_acceptance_gates": {
            "shape_macro_f1": LOCAL_SHAPE_MACRO_F1_GATE,
            "fill_macro_f1": LOCAL_FILL_MACRO_F1_GATE,
            "authority": "session-local preregistration; not maintainer-agreed",
        },
        "fixed_families": SPLIT_FAMILIES,
        "fixed_templates": SPLIT_TEMPLATES,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha256,
        "validation_metrics": validation_metrics,
        "shape_temperature": shape_temperature,
        "fill_temperature": fill_temperature,
        "loss_checkpoints": checkpoints,
        "heldout_test_evaluations": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "elapsed_ms": elapsed_ms,
    }
    (output_dir / "training-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
