# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Source-scale tiles and truth masks from the committed V32 synthetic data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.ocr.real_range_classifier_finetune_v32.dataset import SceneSample, build_split

from .protocol import TILE_OVERLAP, TILE_SIZE


@dataclass(frozen=True)
class TileSample:
    scene_id: str
    left: int
    top: int
    valid_width: int
    valid_height: int
    image: np.ndarray
    target: np.ndarray


def tile_starts(length: int) -> tuple[int, ...]:
    if length <= TILE_SIZE:
        return (0,)
    step = TILE_SIZE - TILE_OVERLAP
    starts = list(range(0, length - TILE_SIZE + 1, step))
    if starts[-1] != length - TILE_SIZE:
        starts.append(length - TILE_SIZE)
    return tuple(starts)


def _target_for_tile(scene: SceneSample, left: int, top: int, width: int, height: int) -> np.ndarray:
    target = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    for truth in scene.truths:
        x0 = max(left, truth.left)
        y0 = max(top, truth.top)
        x1 = min(left + width, truth.right)
        y1 = min(top + height, truth.bottom)
        if x1 > x0 and y1 > y0:
            target[int(y0 - top) : int(y1 - top), int(x0 - left) : int(x1 - left)] = 1
    return target


def build_tiles(split: str) -> tuple[TileSample, ...]:
    tiles: list[TileSample] = []
    for scene in build_split(split):
        height, width = scene.raster.shape
        for top in tile_starts(height):
            for left in tile_starts(width):
                valid_width = min(TILE_SIZE, width - left)
                valid_height = min(TILE_SIZE, height - top)
                image = np.full((TILE_SIZE, TILE_SIZE), 255, dtype=np.uint8)
                image[:valid_height, :valid_width] = scene.raster[top : top + valid_height, left : left + valid_width]
                target = _target_for_tile(scene, left, top, valid_width, valid_height)
                tiles.append(TileSample(scene.scene_id, left, top, valid_width, valid_height, image, target))
    if not tiles:
        raise RuntimeError(f"V35 {split} split produced no source-scale tiles")
    return tuple(tiles)


def to_arrays(tiles: tuple[TileSample, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not tiles:
        raise ValueError("At least one V35 tile is required")
    images = np.stack([(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in tiles]).astype(np.float32)
    targets = np.stack([tile.target for tile in tiles])[:, None, :, :].astype(np.float32)
    return images, targets


__all__ = ["TileSample", "build_split", "build_tiles", "tile_starts", "to_arrays"]
