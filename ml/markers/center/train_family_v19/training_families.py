# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic synthetic-only train families spanning measured input ranges."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from ml.markers.center.line_aware_v1.dataset import LineAwareScene
from ml.markers.center.proposal_geometry_v13 import dataset as v13_dataset

from .protocol import TRAIN_FAMILY_SPECS, TRAIN_SEED_BASE, TRAIN_VARIANTS_PER_FAMILY


def _roundtrip_channel(channel: np.ndarray, scale: float, *, quality: int, blur: float) -> np.ndarray:
    image = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8), mode="L")
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    width, height = image.size
    reduced = image.resize((max(8, round(width * scale)), max(8, round(height * scale))), Image.Resampling.BILINEAR)
    restored = reduced.resize((width, height), Image.Resampling.BICUBIC)
    if quality < 90:
        encoded = BytesIO()
        restored.convert("RGB").save(encoded, format="JPEG", quality=quality, optimize=False, subsampling=2)
        restored = Image.open(BytesIO(encoded.getvalue())).convert("L")
    return np.asarray(restored, dtype=np.float32) / 255.0


def _augment_scene(scene: LineAwareScene, *, scale: float, quality: int, rgba: bool, seed: int) -> LineAwareScene:
    rng = np.random.default_rng(seed)
    ink = 1.0 - scene.tensor[0].numpy()
    ink = _roundtrip_channel(ink, scale, quality=quality, blur=0.10 if scale < 0.5 else 0.0)
    if rgba:
        alpha = 0.82 + 0.12 * float(rng.random())
        ink = 1.0 - ((1.0 - ink) * alpha + (1.0 - alpha))
    ink = np.clip(ink + rng.normal(0.0, 0.0025, ink.shape), 0.0, 1.0)
    text = _roundtrip_channel(scene.tensor[1].numpy(), scale, quality=100, blur=0.0)
    artifact = _roundtrip_channel(scene.tensor[2].numpy(), scale, quality=100, blur=0.0)
    tensor = torch.from_numpy(np.stack((1.0 - ink, text, artifact), axis=0).astype(np.float32, copy=False))
    return LineAwareScene(
        scene_id=scene.scene_id,
        split="train",
        family=scene.family,
        degradation=scene.degradation,
        seed=scene.seed,
        tensor=tensor,
        centers=scene.centers,
        radii=scene.radii,
        prohibited=scene.prohibited,
    )


def build_train_scenes() -> tuple[LineAwareScene, ...]:
    """Build only the disjoint V19 synthetic train stream.

    The V13 proposal extractor remains the proposal stream. The unchanged V13
    dev builder is intentionally not called here, so train-family expansion
    cannot leak dev answers into training.
    """
    scenes: list[LineAwareScene] = []
    for family_index, (family, spec) in enumerate(TRAIN_FAMILY_SPECS.items()):
        source_family = ("geometry_small_train", "geometry_medium_train", "geometry_wide_train", "geometry_mixed_train")[family_index]
        scale = float(spec["resize_long_scale_range"][0])
        quality_min, quality_max = map(int, spec["jpeg_quality_range"])
        rgba = spec["color_modes"] == ["RGBA"]
        for variant in range(TRAIN_VARIANTS_PER_FAMILY):
            seed = TRAIN_SEED_BASE + family_index * 100 + variant
            base = v13_dataset.build_scene(split="train", family=source_family, variant=variant, seed=seed)
            quality = quality_min + ((quality_max - quality_min) * variant // max(1, TRAIN_VARIANTS_PER_FAMILY - 1))
            transformed = _augment_scene(base, scale=scale, quality=quality, rgba=rgba, seed=seed + 1)
            scenes.append(LineAwareScene(
                scene_id=f"train-{family}-{variant}", split="train", family=family,
                degradation=f"resize_{scale:.2f}_{'rgba' if rgba else 'rgb'}_jpeg_{quality}",
                seed=seed, tensor=transformed.tensor, centers=transformed.centers,
                radii=transformed.radii, prohibited=transformed.prohibited,
            ))
    return tuple(scenes)


__all__ = ["build_train_scenes"]
