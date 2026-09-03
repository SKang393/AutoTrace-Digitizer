# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed V32 train/dev scenes represented as DB-style rectangular cores."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.real_range_classifier_finetune_v32.dataset import SceneSample, build_split

from .protocol import DB_SHRINK_RATIO, TILE_OVERLAP, TILE_SIZE


@dataclass(frozen=True)
class ShrinkGeometry:
    """Integer core and side insets for a reversible source rectangle."""

    source: Box
    core: Box
    insets: tuple[int, int, int, int]


@dataclass(frozen=True)
class TileSample:
    scene_id: str
    left: int
    top: int
    valid_width: int
    valid_height: int
    image: np.ndarray
    target: np.ndarray


def clip_box(box: Box, width: int, height: int) -> Box:
    if width < 1 or height < 1:
        raise ValueError("Canvas dimensions must be positive")
    left = min(max(int(box.left), 0), width)
    top = min(max(int(box.top), 0), height)
    right = min(max(int(box.right), left), width)
    bottom = min(max(int(box.bottom), top), height)
    if right <= left or bottom <= top:
        raise ValueError("Box must have positive clipped extent")
    return Box(left, top, right, bottom)


def shrink_box(box: Box, *, canvas_width: int | None = None, canvas_height: int | None = None) -> ShrinkGeometry:
    """Create one centered fixed-ratio core, retaining exact audit insets."""
    source = box
    if canvas_width is not None or canvas_height is not None:
        if canvas_width is None or canvas_height is None:
            raise ValueError("Both canvas dimensions are required when clipping")
        source = clip_box(box, canvas_width, canvas_height)
    source_width = int(source.width)
    source_height = int(source.height)
    core_width = max(1, min(source_width, int(round(source_width * DB_SHRINK_RATIO))))
    core_height = max(1, min(source_height, int(round(source_height * DB_SHRINK_RATIO))))
    left_inset = (source_width - core_width) // 2
    right_inset = source_width - core_width - left_inset
    top_inset = (source_height - core_height) // 2
    bottom_inset = source_height - core_height - top_inset
    core = Box(
        source.left + left_inset,
        source.top + top_inset,
        source.right - right_inset,
        source.bottom - bottom_inset,
    )
    return ShrinkGeometry(source, core, (left_inset, top_inset, right_inset, bottom_inset))


def expand_geometry(geometry: ShrinkGeometry) -> Box:
    """Pair operation for shrink_box; no rounding or inference is involved."""
    left, top, right, bottom = geometry.insets
    expanded = Box(
        geometry.core.left - left,
        geometry.core.top - top,
        geometry.core.right + right,
        geometry.core.bottom + bottom,
    )
    if expanded != geometry.source:
        raise RuntimeError("V36 shrink/expand geometry lost source coordinates")
    return expanded


def expand_core_box(core: Box, *, canvas_width: int, canvas_height: int) -> Box:
    """Expand a predicted core with the fixed DB ratio and clip to source pixels."""
    if core.width < 1 or core.height < 1:
        raise ValueError("Predicted core must have positive extent")
    # The ratio is fixed by the protocol. This is deliberately not a dev-tuned
    # threshold or per-scene margin. Recover the full integer extent with
    # half-up rounding, split the added pixels around the center, and clip.
    full_width = max(int(core.width), int(math.floor(core.width / DB_SHRINK_RATIO + 0.5)))
    full_height = max(int(core.height), int(math.floor(core.height / DB_SHRINK_RATIO + 0.5)))
    left_inset = (full_width - int(core.width)) // 2
    right_inset = full_width - int(core.width) - left_inset
    top_inset = (full_height - int(core.height)) // 2
    bottom_inset = full_height - int(core.height) - top_inset
    candidate = Box(
        core.left - left_inset,
        core.top - top_inset,
        core.right + right_inset,
        core.bottom + bottom_inset,
    )
    return clip_box(candidate, canvas_width, canvas_height)


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
    scene_height, scene_width = scene.raster.shape
    for truth in scene.truths:
        geometry = shrink_box(truth, canvas_width=scene_width, canvas_height=scene_height)
        core = geometry.core
        x0 = max(left, core.left)
        y0 = max(top, core.top)
        x1 = min(left + width, core.right)
        y1 = min(top + height, core.bottom)
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
        raise RuntimeError(f"V36 {split} split produced no source-scale tiles")
    return tuple(tiles)


def to_arrays(tiles: tuple[TileSample, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not tiles:
        raise ValueError("At least one V36 tile is required")
    images = np.stack([(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in tiles]).astype(np.float32)
    targets = np.stack([tile.target for tile in tiles])[:, None, :, :].astype(np.float32)
    return images, targets


__all__ = [
    "SceneSample", "ShrinkGeometry", "TileSample", "build_split", "build_tiles", "clip_box",
    "expand_core_box", "expand_geometry", "shrink_box", "tile_starts", "to_arrays",
]
