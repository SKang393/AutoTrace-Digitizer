# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural marker scenes with low-frequency background drift."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ml.markers.center.line_aware_v1.dataset import LineAwareScene
from ml.markers.center.runtime_consistency_v2.dataset import (
    build_scene as build_source_scene,
    load_sealed_public_archive,
    save_sealed_public_archive,
)


DATASET_REVISION = "marker-center-background-invariant-procedural-v3"
WIDTH = 256
HEIGHT = 192
SELECTION_FAMILIES = (
    "tilted_arc_validation",
    "staggered_drop_validation",
    "late_reversal_validation",
    "compressed_plateau_validation",
)
SEALED_PUBLIC_FAMILIES = (
    "counter_slope_public",
    "offset_cycle_public",
    "delayed_rebound_public",
    "split_plateau_public",
    "wide_fall_public",
)
DEGRADATIONS = {
    "validation": (
        "vertical_quadratic_validation",
        "diagonal_plane_validation",
        "broad_wave_validation",
    ),
    "sealed_public": (
        "dual_axis_curve_public",
        "offcenter_vignette_public",
        "sinusoidal_field_public",
    ),
}
SELECTION_SEED_BASE = 1_711_000
SELECTION_VARIANTS = 4
SEALED_PUBLIC_VARIANTS = 4
_BASE_FAMILIES = {
    "tilted_arc_validation": "late_crossover_validation",
    "staggered_drop_validation": "compress_expand_validation",
    "late_reversal_validation": "asymmetric_wave_validation",
    "compressed_plateau_validation": "session_gap_validation",
    "counter_slope_public": "alternating_rise_train",
    "offset_cycle_public": "dense_cycle_train",
    "delayed_rebound_public": "paired_reversal_train",
    "split_plateau_public": "offset_plateau_train",
    "wide_fall_public": "stepped_fall_train",
}


def _background_field(
    degradation: str,
    *,
    seed: int,
    variant: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed ^ 0x5A17C3)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32),
        np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32),
        indexing="ij",
    )
    if degradation == "vertical_quadratic_validation":
        field = 0.010 + (0.045 * np.square((yy + (0.10 * (variant - 1.5))) / 1.2))
    elif degradation == "diagonal_plane_validation":
        field = 0.025 + (0.022 * xx) + (0.015 * yy)
    elif degradation == "broad_wave_validation":
        field = 0.025 + (0.022 * np.sin((xx * 1.25 + yy * 0.35 + variant * 0.2) * np.pi))
    elif degradation == "dual_axis_curve_public":
        field = 0.012 + (0.018 * np.square(xx)) + (0.026 * np.square(yy + 0.2))
    elif degradation == "offcenter_vignette_public":
        field = 0.010 + (0.038 * np.sqrt(np.square(xx - 0.22) + np.square(yy + 0.18)) / 1.7)
    elif degradation == "sinusoidal_field_public":
        field = 0.026 + (0.018 * np.sin((xx * 1.7 - yy * 0.8 + variant * 0.17) * np.pi))
    else:
        raise ValueError(f"Unsupported degradation {degradation!r}")
    noise = rng.normal(0.0, 0.0015, size=(HEIGHT, WIDTH)).astype(np.float32)
    return torch.from_numpy(np.clip(field + noise, 0.0, 0.065).astype(np.float32))


def build_scene(
    *,
    split: str,
    family: str,
    degradation: str,
    variant: int,
    seed: int,
) -> LineAwareScene:
    if family not in _BASE_FAMILIES:
        raise ValueError(f"Unsupported family {family!r}")
    source = build_source_scene(
        split="validation",
        family=_BASE_FAMILIES[family],
        degradation="none",
        variant=variant + 7,
        seed=seed,
    )
    tensor = source.tensor.clone()
    field = _background_field(degradation, seed=seed, variant=variant)
    tensor[0] = torch.clamp((tensor[0] * 0.94) + field, 0.0, 1.0)
    return LineAwareScene(
        scene_id=f"{split}-{family}-{variant}",
        split=split,
        family=family,
        degradation=degradation,
        seed=seed,
        tensor=tensor,
        centers=source.centers,
        radii=source.radii,
        prohibited=source.prohibited,
    )


def build_selection_scenes() -> tuple[LineAwareScene, ...]:
    return tuple(
        build_scene(
            split="validation",
            family=family,
            degradation=DEGRADATIONS["validation"][(family_index + variant) % 3],
            variant=variant,
            seed=SELECTION_SEED_BASE + (family_index * 100) + variant,
        )
        for family_index, family in enumerate(SELECTION_FAMILIES)
        for variant in range(SELECTION_VARIANTS)
    )


def build_sealed_public_scenes(secret_seed: int) -> tuple[LineAwareScene, ...]:
    return tuple(
        build_scene(
            split="sealed_public",
            family=family,
            degradation=DEGRADATIONS["sealed_public"][(family_index + variant) % 3],
            variant=variant,
            seed=secret_seed + (family_index * 100) + variant,
        )
        for family_index, family in enumerate(SEALED_PUBLIC_FAMILIES)
        for variant in range(SEALED_PUBLIC_VARIANTS)
    )


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def scene_manifest(scene: LineAwareScene, *, expose_truth: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "scene_id": scene.scene_id,
        "split": scene.split,
        "family": scene.family,
        "degradation": scene.degradation,
        "tensor_sha256": _array_hash(scene.tensor.numpy().astype("<f4", copy=False)),
        "center_count": len(scene.centers),
        "prohibited_kinds": sorted({point.kind for point in scene.prohibited}),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
    }
    if expose_truth:
        result.update(
            {
                "centers": [[x, y] for x, y in scene.centers],
                "radii": list(scene.radii),
                "prohibited": [
                    {"kind": point.kind, "x": point.x, "y": point.y}
                    for point in scene.prohibited
                ],
            }
        )
    return result


def selection_manifest() -> dict[str, object]:
    scenes = build_selection_scenes()
    return {
        "schema": "graphreader.marker-center-background-invariant-selection.v3",
        "task": "marker-center",
        "revision": "marker-center-background-invariant-v3",
        "dataset_revision": DATASET_REVISION,
        "scene_count": len(scenes),
        "family_ids": list(SELECTION_FAMILIES),
        "degradation_ids": list(DEGRADATIONS["validation"]),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "cases": [scene_manifest(scene, expose_truth=True) for scene in scenes],
    }


__all__ = [
    "DATASET_REVISION",
    "DEGRADATIONS",
    "SEALED_PUBLIC_FAMILIES",
    "SELECTION_FAMILIES",
    "build_sealed_public_scenes",
    "build_selection_scenes",
    "load_sealed_public_archive",
    "save_sealed_public_archive",
    "selection_manifest",
]
