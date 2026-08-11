# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes for normalized-input marker-center training."""

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


DATASET_REVISION = "marker-center-normalized-training-procedural-v4"
WIDTH = 256
HEIGHT = 192
SELECTION_FAMILIES = {
    "train": (
        "counter_cycle_train",
        "long_hold_train",
        "paired_drop_train",
        "sparse_recovery_train",
        "dense_offset_train",
        "late_plateau_train",
    ),
    "validation": (
        "double_turn_validation",
        "compressed_wave_validation",
        "delayed_switch_validation",
        "stair_rebound_validation",
    ),
}
SEALED_PUBLIC_FAMILIES = (
    "reverse_cycle_public",
    "wide_recovery_public",
    "offset_echo_public",
    "late_drop_public",
    "split_reversal_public",
)
DEGRADATIONS = {
    "train": (
        "diagonal_quadratic_train",
        "offaxis_wave_train",
        "radial_vignette_train",
        "piecewise_plane_train",
    ),
    "validation": (
        "saddle_field_validation",
        "cross_wave_validation",
        "elliptic_vignette_validation",
    ),
    "sealed_public": (
        "skew_curve_public",
        "two_wave_public",
        "corner_glow_public",
    ),
}
SELECTION_SEED_BASE = {"train": 1_913_000, "validation": 2_027_000}
SELECTION_VARIANTS = {"train": 5, "validation": 4}
SEALED_PUBLIC_VARIANTS = 4
_BASE_FAMILIES = {
    "counter_cycle_train": "dense_cycle_train",
    "long_hold_train": "offset_plateau_train",
    "paired_drop_train": "stepped_fall_train",
    "sparse_recovery_train": "sparse_probe_train",
    "dense_offset_train": "alternating_rise_train",
    "late_plateau_train": "paired_reversal_train",
    "double_turn_validation": "late_crossover_validation",
    "compressed_wave_validation": "compress_expand_validation",
    "delayed_switch_validation": "session_gap_validation",
    "stair_rebound_validation": "asymmetric_wave_validation",
    "reverse_cycle_public": "wide_cycle_public",
    "wide_recovery_public": "descending_echo_public",
    "offset_echo_public": "interleaved_plateau_public",
    "late_drop_public": "late_surge_public",
    "split_reversal_public": "double_rebound_public",
}


def _background_field(degradation: str, *, seed: int, variant: int) -> torch.Tensor:
    rng = np.random.default_rng(seed ^ 0x74C219)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32),
        np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32),
        indexing="ij",
    )
    phase = variant * 0.19
    if degradation == "diagonal_quadratic_train":
        field = 0.012 + 0.018 * np.square(xx + 0.35 * yy + phase - 0.25)
    elif degradation == "offaxis_wave_train":
        field = 0.026 + 0.021 * np.sin((1.35 * xx - 0.55 * yy + phase) * np.pi)
    elif degradation == "radial_vignette_train":
        field = 0.010 + 0.041 * np.sqrt(np.square(xx + 0.18) + np.square(yy - 0.24)) / 1.75
    elif degradation == "piecewise_plane_train":
        field = 0.020 + 0.018 * xx + 0.014 * yy + 0.009 * np.maximum(0.0, yy - xx)
    elif degradation == "saddle_field_validation":
        field = 0.027 + 0.018 * (np.square(xx) - 0.65 * np.square(yy))
    elif degradation == "cross_wave_validation":
        field = 0.026 + 0.012 * np.sin((1.5 * xx + phase) * np.pi) + 0.012 * np.cos((1.2 * yy - phase) * np.pi)
    elif degradation == "elliptic_vignette_validation":
        field = 0.011 + 0.038 * np.sqrt(np.square((xx - 0.28) / 1.3) + np.square((yy + 0.12) / 0.8)) / 1.8
    elif degradation == "skew_curve_public":
        field = 0.015 + 0.020 * np.square(0.7 * xx - yy + 0.2)
    elif degradation == "two_wave_public":
        field = 0.025 + 0.014 * np.sin((1.8 * xx + 0.4 * yy + phase) * np.pi) + 0.010 * np.cos((0.5 * xx - 1.4 * yy) * np.pi)
    elif degradation == "corner_glow_public":
        field = 0.010 + 0.042 * np.sqrt(np.square(xx + 0.72) + np.square(yy - 0.66)) / 2.35
    else:
        raise ValueError(f"Unsupported degradation {degradation!r}")
    noise = rng.normal(0.0, 0.0018, size=(HEIGHT, WIDTH)).astype(np.float32)
    return torch.from_numpy(np.clip(field + noise, 0.0, 0.075).astype(np.float32))


def build_scene(*, split: str, family: str, degradation: str, variant: int, seed: int) -> LineAwareScene:
    if family not in _BASE_FAMILIES:
        raise ValueError(f"Unsupported family {family!r}")
    source = build_source_scene(
        split="validation",
        family=_BASE_FAMILIES[family],
        degradation="none",
        variant=variant + 19,
        seed=seed,
    )
    tensor = source.tensor.clone()
    tensor[0] = torch.clamp((tensor[0] * 0.925) + _background_field(degradation, seed=seed, variant=variant), 0.0, 1.0)
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


def build_selection_scenes(split: str) -> tuple[LineAwareScene, ...]:
    if split not in SELECTION_FAMILIES:
        raise ValueError(f"Unsupported selection split {split!r}")
    return tuple(
        build_scene(
            split=split,
            family=family,
            degradation=DEGRADATIONS[split][(family_index + variant) % len(DEGRADATIONS[split])],
            variant=variant,
            seed=SELECTION_SEED_BASE[split] + family_index * 100 + variant,
        )
        for family_index, family in enumerate(SELECTION_FAMILIES[split])
        for variant in range(SELECTION_VARIANTS[split])
    )


def build_sealed_public_scenes(secret_seed: int) -> tuple[LineAwareScene, ...]:
    return tuple(
        build_scene(
            split="sealed_public",
            family=family,
            degradation=DEGRADATIONS["sealed_public"][(family_index + variant) % len(DEGRADATIONS["sealed_public"])],
            variant=variant,
            seed=secret_seed + family_index * 100 + variant,
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
                "seed": scene.seed,
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
    scenes = tuple(
        scene
        for split in ("train", "validation")
        for scene in build_selection_scenes(split)
    )
    return {
        "schema": "graphreader.marker-center-normalized-training-selection.v4",
        "task": "marker-center",
        "revision": "marker-center-normalized-training-v4",
        "dataset_revision": DATASET_REVISION,
        "scene_count": len(scenes),
        "train_scene_count": len(build_selection_scenes("train")),
        "validation_scene_count": len(build_selection_scenes("validation")),
        "split_families": {key: list(value) for key, value in SELECTION_FAMILIES.items()},
        "degradations": {key: list(value) for key, value in DEGRADATIONS.items()},
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
