# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh V9 scenes using the reviewed V8 renderer with disjoint identities."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import ml.markers.center.mask_consensus_v8.dataset as parent
from ml.markers.center.feasible_dense_v6.dataset import DenseScene


WIDTH = parent.WIDTH
HEIGHT = parent.HEIGHT
TRAIN_SCENE_COUNT = 512
VALIDATION_SCENE_COUNT = 128
PUBLIC_SCENE_COUNT = 160
PUBLIC_DATASET_SEED = 9173
SPLITS = {
    "train": {
        "count": TRAIN_SCENE_COUNT,
        "seed_offset": 10_117_173,
        "renderer_family": "mask-consensus-layout-bank-v9-train",
        "degradation_family": "mask-consensus-capture-bank-v9-train",
    },
    "validation": {
        "count": VALIDATION_SCENE_COUNT,
        "seed_offset": 10_917_173,
        "renderer_family": "mask-consensus-heldout-layout-v9-validation",
        "degradation_family": "mask-consensus-heldout-capture-v9-validation",
    },
    "sealed_public": {
        "count": PUBLIC_SCENE_COUNT,
        "seed_offset": 11_717_173,
        "renderer_family": "mask-consensus-hidden-layout-v9-public",
        "degradation_family": "mask-consensus-hidden-capture-v9-public",
    },
}
RENDERER_FAMILIES = {
    "train": (
        "offset-fan-mixed-glyph-v9-train",
        "late-reversal-compact-legend-v9-train",
        "asymmetric-bracket-diamond-v9-train",
        "staggered-rise-triangle-v9-train",
        "broken-wave-square-v9-train",
    ),
    "validation": (
        "heldout-early-cycle-v9-validation",
        "heldout-reverse-step-v9-validation",
        "heldout-narrow-echo-v9-validation",
    ),
    "sealed_public": (
        "hidden-late-slope-v9-public",
        "hidden-forward-tail-v9-public",
        "hidden-offset-fan-v9-public",
    ),
}
DEGRADATION_FAMILIES = {
    "train": (
        "cross-axis-ink-v9-train",
        "soft-printer-v9-train",
        "quantized-copy-v9-train",
        "low-contrast-grain-v9-train",
        "artifact-underfill-v9-train",
        "artifact-overfill-v9-train",
    ),
    "validation": (
        "heldout-copy-blur-v9-validation",
        "heldout-neutral-cast-v9-validation",
        "heldout-display-compression-v9-validation",
    ),
    "sealed_public": (
        "hidden-printer-chain-v9-public",
        "hidden-screen-capture-v9-public",
        "hidden-paper-noise-v9-public",
    ),
}

HARD_NEGATIVE_TOLERANCE = parent.HARD_NEGATIVE_TOLERANCE
KIND_TO_INDEX = parent.KIND_TO_INDEX
MATCH_TOLERANCE = parent.MATCH_TOLERANCE
MINIMUM_CENTER_SEPARATION = parent.MINIMUM_CENTER_SEPARATION
PROHIBITED_KINDS = parent.PROHIBITED_KINDS
REQUIRED_DISJOINT_CLEARANCE = parent.REQUIRED_DISJOINT_CLEARANCE
feasibility_summary = parent.feasibility_summary
read_archive = parent.read_archive
validate_scene_feasibility = parent.validate_scene_feasibility
write_archive = parent.write_archive


def _ground_truth_sha256(scene: DenseScene, scene_id: str) -> str:
    value = {
        "artifact_target_sha256": hashlib.sha256(
            scene.artifact_target.tobytes(order="C")
        ).hexdigest(),
        "centers": scene.centers,
        "hard_negatives": scene.hard_negatives,
        "scene_id": scene_id,
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _render_indices(split: str, indices: tuple[int, ...]) -> tuple[DenseScene, ...]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    count = int(SPLITS[split]["count"])
    if any(index < 0 or index >= count for index in indices):
        raise IndexError(f"Scene index is outside the {split} split")
    original = (parent.SPLITS, parent.RENDERER_FAMILIES, parent.DEGRADATION_FAMILIES)
    parent.SPLITS = SPLITS
    parent.RENDERER_FAMILIES = RENDERER_FAMILIES
    parent.DEGRADATION_FAMILIES = DEGRADATION_FAMILIES
    try:
        scenes = tuple(parent._draw_scene(split, index) for index in indices)
    finally:
        parent.SPLITS, parent.RENDERER_FAMILIES, parent.DEGRADATION_FAMILIES = original
    result: list[DenseScene] = []
    for index, scene in zip(indices, scenes, strict=True):
        scene_id = f"marker-mask-consensus-v9-{split}-{index:04d}"
        updated = replace(
            scene,
            scene_id=scene_id,
            ground_truth_sha256=_ground_truth_sha256(scene, scene_id),
        )
        validate_scene_feasibility(updated)
        result.append(updated)
    return tuple(result)


def render_scene(split: str, index: int) -> DenseScene:
    """Render one deterministic scene without materializing a full split."""

    return _render_indices(split, (index,))[0]


def render_split(split: str) -> tuple[DenseScene, ...]:
    return _render_indices(split, tuple(range(int(SPLITS[split]["count"]))))


__all__ = [
    "DEGRADATION_FAMILIES",
    "HEIGHT",
    "HARD_NEGATIVE_TOLERANCE",
    "KIND_TO_INDEX",
    "MATCH_TOLERANCE",
    "MINIMUM_CENTER_SEPARATION",
    "PROHIBITED_KINDS",
    "PUBLIC_DATASET_SEED",
    "PUBLIC_SCENE_COUNT",
    "RENDERER_FAMILIES",
    "REQUIRED_DISJOINT_CLEARANCE",
    "SPLITS",
    "TRAIN_SCENE_COUNT",
    "VALIDATION_SCENE_COUNT",
    "WIDTH",
    "feasibility_summary",
    "read_archive",
    "render_scene",
    "render_split",
    "validate_scene_feasibility",
    "write_archive",
]
