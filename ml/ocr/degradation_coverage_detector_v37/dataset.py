# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""V35 full-box tiles over a deterministic, train-only degraded distribution."""

from __future__ import annotations

from dataclasses import replace
import io
from functools import lru_cache

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ml.ocr.real_range_classifier_finetune_v32.dataset import (
    SceneSample,
    build_split as build_v32_split,
    split_fingerprint as v32_split_fingerprint,
)
from ml.ocr.real_range_detector_v35.dataset import TileSample, tile_starts

from .protocol import (
    DEGRADATION_GRIDS,
    DEGRADATION_KINDS,
    TRAIN_VARIANT_SEEDS,
    TILE_SIZE,
)


def _jpeg_roundtrip(raster: np.ndarray, quality: int) -> np.ndarray:
    with io.BytesIO() as buffer:
        Image.fromarray(raster, mode="L").convert("RGB").save(
            buffer, format="JPEG", quality=quality, optimize=False,
            progressive=False, subsampling=0,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("L"), dtype=np.uint8).copy()


def _degrade(raster: np.ndarray, kind: str, value: float, seed: int) -> np.ndarray:
    image = Image.fromarray(raster, mode="L")
    if kind == "blur":
        result = image.filter(ImageFilter.GaussianBlur(radius=value))
        return np.asarray(result, dtype=np.uint8).copy()
    if kind == "contrast":
        result = ImageEnhance.Contrast(image).enhance(value)
        return np.asarray(result, dtype=np.uint8).copy()
    if kind == "gaussian_noise":
        generator = np.random.default_rng(seed)
        values = np.asarray(image, dtype=np.int16)
        noise = generator.normal(0.0, value, values.shape)
        return np.clip(np.rint(values + noise), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        return _jpeg_roundtrip(raster, int(round(value)))
    raise ValueError(f"Unsupported V37 degradation kind: {kind}")


def _variant(base: SceneSample, base_index: int) -> SceneSample:
    seed = TRAIN_VARIANT_SEEDS[base_index]
    raster = base.raster
    for kind_index, kind in enumerate(DEGRADATION_KINDS):
        raster = _degrade(raster, kind, DEGRADATION_GRIDS[kind][base_index], seed + kind_index)
    return replace(
        base,
        scene_id=f"{base.scene_id}-v37-composite-{seed}",
        degradation_family="v37_train_composite",
        raster=raster,
    )


@lru_cache(maxsize=2)
def build_split(split: str) -> tuple[SceneSample, ...]:
    """Return V32 dev unchanged and V32 train plus fresh train-only variants."""
    if split == "dev":
        return build_v32_split("dev")
    if split != "train":
        raise ValueError(f"V37 exposes only train and dev, not {split}")
    base_scenes = build_v32_split("train")
    variants = tuple(
        _variant(base, base_index)
        for base_index, base in enumerate(base_scenes)
    )
    return base_scenes + variants


def build_base_train_split() -> tuple[SceneSample, ...]:
    """Return the exact five V32 train scenes retained byte-for-byte."""
    return build_v32_split("train")


def split_fingerprint(split: str) -> str:
    if split == "dev":
        return v32_split_fingerprint("dev")
    if split == "train":
        import hashlib

        digest = hashlib.sha256()
        for scene in build_split("train"):
            digest.update(scene.scene_id.encode())
            digest.update(scene.raster.tobytes(order="C"))
            for truth in scene.truths:
                digest.update(f"{truth.left},{truth.top},{truth.right},{truth.bottom}\n".encode())
        return digest.hexdigest()
    raise ValueError(f"Unknown V37 split: {split}")


def _target_for_tile(scene: SceneSample, left: int, top: int, width: int, height: int) -> np.ndarray:
    target = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    for truth in scene.truths:
        x0 = max(left, truth.left)
        y0 = max(top, truth.top)
        x1 = min(left + width, truth.right)
        y1 = min(top + height, truth.bottom)
        if x1 > x0 and y1 > y0:
            target[int(y0 - top):int(y1 - top), int(x0 - left):int(x1 - left)] = 1
    return target


def build_tiles(split: str) -> tuple[TileSample, ...]:
    """Build V35-compatible padded tiles with unchanged full-box targets."""
    tiles: list[TileSample] = []
    for scene in build_split(split):
        height, width = scene.raster.shape
        for top in tile_starts(height):
            for left in tile_starts(width):
                valid_width = min(TILE_SIZE, width - left)
                valid_height = min(TILE_SIZE, height - top)
                image = np.full((TILE_SIZE, TILE_SIZE), 255, dtype=np.uint8)
                image[:valid_height, :valid_width] = scene.raster[top:top + valid_height, left:left + valid_width]
                tiles.append(TileSample(
                    scene.scene_id, left, top, valid_width, valid_height, image,
                    _target_for_tile(scene, left, top, valid_width, valid_height),
                ))
    if not tiles:
        raise RuntimeError(f"V37 {split} split produced no source-scale tiles")
    return tuple(tiles)


def to_arrays(tiles: tuple[TileSample, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not tiles:
        raise ValueError("At least one V37 tile is required")
    images = np.stack([(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in tiles]).astype(np.float32)
    targets = np.stack([tile.target for tile in tiles])[:, None, :, :].astype(np.float32)
    return images, targets


__all__ = [
    "SceneSample", "TileSample", "build_base_train_split", "build_split", "build_tiles",
    "split_fingerprint", "to_arrays",
]
