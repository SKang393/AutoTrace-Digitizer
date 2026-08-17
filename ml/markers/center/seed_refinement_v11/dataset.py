# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh V11 scenes using disjoint renderer, degradation, and seed identities."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import ml.markers.center.mask_consensus_v8.dataset as parent
import numpy as np
from ml.markers.center.feasible_dense_v6.dataset import DenseScene


WIDTH = parent.WIDTH
HEIGHT = parent.HEIGHT
TRAIN_SCENE_COUNT = 512
VALIDATION_SCENE_COUNT = 128
PUBLIC_SCENE_COUNT = 160
PUBLIC_DATASET_SEED = 11_411
SPLITS = {
    "train": {
        "count": TRAIN_SCENE_COUNT,
        "seed_offset": 24_411_411,
        "renderer_family": "seed-refinement-layout-v11-train",
        "degradation_family": "seed-refinement-capture-v11-train",
    },
    "validation": {
        "count": VALIDATION_SCENE_COUNT,
        "seed_offset": 15_211_411,
        "renderer_family": "seed-refinement-layout-v11-validation",
        "degradation_family": "seed-refinement-capture-v11-validation",
    },
    "sealed_public": {
        "count": PUBLIC_SCENE_COUNT,
        "seed_offset": 16_011_411,
        "renderer_family": "seed-refinement-layout-v11-public",
        "degradation_family": "seed-refinement-capture-v11-public",
    },
}
RENDERER_FAMILIES = {
    "train": (
        "offset-plateau-mixed-glyph-v11-train",
        "crossed-rise-compact-legend-v11-train",
        "asymmetric-tail-diamond-v11-train",
        "staggered-reversal-triangle-v11-train",
        "broken-cycle-square-v11-train",
    ),
    "validation": (
        "heldout-forward-cycle-v11-validation",
        "heldout-late-step-v11-validation",
        "heldout-wide-echo-v11-validation",
    ),
    "sealed_public": (
        "hidden-early-slope-v11-public",
        "hidden-reverse-tail-v11-public",
        "hidden-staggered-fan-v11-public",
    ),
}
DEGRADATION_FAMILIES = {
    "train": (
        "cross-axis-grain-v11-train",
        "soft-toner-v11-train",
        "quantized-scan-v11-train",
        "low-contrast-copy-v11-train",
        "artifact-thin-v11-train",
        "artifact-thick-v11-train",
    ),
    "validation": (
        "heldout-toner-blur-v11-validation",
        "heldout-neutral-paper-v11-validation",
        "heldout-screen-compression-v11-validation",
    ),
    "sealed_public": (
        "hidden-copy-chain-v11-public",
        "hidden-display-capture-v11-public",
        "hidden-paper-grain-v11-public",
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


def _apply_v11_capture(scene: DenseScene, split: str, index: int) -> DenseScene:
    """Add a deterministic V11-only paper field before freezing source bytes."""

    split_phase = {"train": 3, "validation": 11, "sealed_public": 19}[split]
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    field = (
        np.sin((xx + split_phase + index % 17) / (13.0 + index % 5))
        + np.cos((yy * 1.3 + split_phase + index % 23) / (11.0 + index % 7))
    )
    amplitude = 0.75 + 0.25 * (index % 4)
    pixels = np.clip(
        scene.raster.astype(np.float32) + np.rint(field * amplitude),
        0,
        255,
    ).astype(np.uint8)
    tensor = scene.tensor.copy()
    tensor[0] = 1.0 - pixels.astype(np.float32) / 255.0
    return replace(
        scene,
        raster=pixels,
        tensor=tensor,
        source_sha256=hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
    )


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
        scene = _apply_v11_capture(scene, split, index)
        scene_id = f"marker-seed-refinement-v11-{split}-{index:04d}"
        updated = replace(
            scene,
            scene_id=scene_id,
            ground_truth_sha256=_ground_truth_sha256(scene, scene_id),
        )
        validate_scene_feasibility(updated)
        result.append(updated)
    return tuple(result)


def render_scene(split: str, index: int) -> DenseScene:
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
