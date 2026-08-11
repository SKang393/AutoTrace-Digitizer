# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen P2 whole-frame tile composition from the unchanged V3 sources."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import cv2
import numpy as np

from .dataset import _production_resize, _render_source, _shrink_target, _supervision_mask
from .protocol import FRAME_HEIGHT, FRAME_WIDTH, PATCH_HEIGHT, PATCH_WIDTH, TRAIN_SAMPLE_COUNT


TILES_PER_SOURCE = 3
P2_TRAINING_SAMPLE_COUNT = TRAIN_SAMPLE_COUNT * TILES_PER_SOURCE


@dataclass(frozen=True)
class P2TrainingTile:
    tile_id: str
    source_index: int
    kind: str
    left: int
    top: int
    bgr: np.ndarray
    target: np.ndarray
    supervision_mask: np.ndarray


def _clamped_origin(center: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    left = min(max(0, center[0] - PATCH_WIDTH // 2), width - PATCH_WIDTH)
    top = min(max(0, center[1] - PATCH_HEIGHT // 2), height - PATCH_HEIGHT)
    return left, top


def _tile_origins(
    source_index: int,
    kind: str,
    full_target: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    height, width = full_target.shape
    grid = (
        (0, 0),
        ((width - PATCH_WIDTH) // 2, 0),
        (width - PATCH_WIDTH, 0),
        (0, (height - PATCH_HEIGHT) // 2),
        ((width - PATCH_WIDTH) // 2, (height - PATCH_HEIGHT) // 2),
        (width - PATCH_WIDTH, (height - PATCH_HEIGHT) // 2),
        (0, height - PATCH_HEIGHT),
        ((width - PATCH_WIDTH) // 2, height - PATCH_HEIGHT),
        (width - PATCH_WIDTH, height - PATCH_HEIGHT),
    )
    origins: list[tuple[int, int]] = []
    if kind == "text":
        ys, xs = np.nonzero(full_target)
        center = (
            int(round((float(xs.min()) + float(xs.max())) / 2.0)),
            int(round((float(ys.min()) + float(ys.max())) / 2.0)),
        )
        origins.append(_clamped_origin(center, width, height))
    start = (source_index * 5 + (1 if kind == "text" else 3)) % len(grid)
    for offset in range(len(grid)):
        origin = grid[(start + offset * 4) % len(grid)]
        if origin not in origins:
            origins.append(origin)
        if len(origins) == TILES_PER_SOURCE:
            break
    if len(origins) != TILES_PER_SOURCE:
        raise RuntimeError("P2 tile composition could not produce three distinct origins")
    return tuple(origins)


def render_p2_tiles(source_index: int) -> tuple[P2TrainingTile, ...]:
    if not 0 <= source_index < TRAIN_SAMPLE_COUNT:
        raise ValueError("P2 source index is out of range")
    source = _render_source("train", source_index)
    bgr = _production_resize(source.detector_bgr, cv2.INTER_LINEAR)
    full_target = _production_resize(source.target, cv2.INTER_NEAREST)
    source_positive = _shrink_target(source.target)
    target = _production_resize(source_positive, cv2.INTER_NEAREST)
    supervision = _production_resize(_supervision_mask(source.target, source_positive), cv2.INTER_NEAREST)
    tiles: list[P2TrainingTile] = []
    for ordinal, (left, top) in enumerate(_tile_origins(source_index, source.kind, full_target)):
        right, bottom = left + PATCH_WIDTH, top + PATCH_HEIGHT
        tiles.append(P2TrainingTile(
            tile_id=f"graph-text-ignore-band-v3-p2-{source_index:05d}-{ordinal}",
            source_index=source_index,
            kind=source.kind,
            left=left,
            top=top,
            bgr=np.ascontiguousarray(bgr[top:bottom, left:right, :]),
            target=np.ascontiguousarray(target[top:bottom, left:right]),
            supervision_mask=np.ascontiguousarray(supervision[top:bottom, left:right]),
        ))
    return tuple(tiles)


def build_p2_training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [tile for source_index in range(TRAIN_SAMPLE_COUNT) for tile in render_p2_tiles(source_index)]
    if len(samples) != P2_TRAINING_SAMPLE_COUNT:
        raise RuntimeError("P2 training composition sample count changed")
    return (
        np.stack([sample.bgr for sample in samples]).astype(np.uint8),
        np.stack([sample.target for sample in samples])[:, None, :, :].astype(np.uint8),
        np.stack([sample.supervision_mask for sample in samples])[:, None, :, :].astype(np.uint8),
    )


def p2_composition_fingerprint() -> str:
    records = []
    for source_index in range(TRAIN_SAMPLE_COUNT):
        for tile in render_p2_tiles(source_index):
            records.append({
                "tile_id": tile.tile_id,
                "source_index": tile.source_index,
                "kind": tile.kind,
                "left": tile.left,
                "top": tile.top,
                "bgr_sha256": sha256(tile.bgr.tobytes(order="C")).hexdigest(),
                "target_sha256": sha256(tile.target.tobytes(order="C")).hexdigest(),
                "supervision_mask_sha256": sha256(tile.supervision_mask.tobytes(order="C")).hexdigest(),
            })
    return sha256((json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


__all__ = [
    "P2TrainingTile",
    "P2_TRAINING_SAMPLE_COUNT",
    "TILES_PER_SOURCE",
    "build_p2_training_arrays",
    "p2_composition_fingerprint",
    "render_p2_tiles",
]
