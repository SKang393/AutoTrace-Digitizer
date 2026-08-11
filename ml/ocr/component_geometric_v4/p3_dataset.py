# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P3 shape-plus-geometry glyph encoding derived from P1 and P2 validation."""

from __future__ import annotations

import numpy as np

from .dataset import CANVAS_HEIGHT, _filtered_foreground, _normalize_glyph
from .protocol import GLYPH_HEIGHT, GLYPH_WIDTH


GEOMETRY_FEATURE_COUNT = 4
ENCODED_GLYPH_WIDTH = GLYPH_WIDTH + GEOMETRY_FEATURE_COUNT


def _encode_shape_and_geometry(
    raster: np.ndarray,
    mask: np.ndarray,
    component_width: int,
) -> np.ndarray:
    normalized = _normalize_glyph(raster, mask)
    ys, _ = np.nonzero(mask)
    height = int(ys.max() - ys.min() + 1)
    bbox_area = max(1, height * component_width)
    geometry = np.asarray(
        (
            height / CANVAS_HEIGHT,
            component_width / CANVAS_HEIGHT,
            ((float(ys.min()) + float(ys.max())) / 2.0) / (CANVAS_HEIGHT - 1),
            float(mask.sum()) / bbox_area,
        ),
        dtype=np.float32,
    )
    columns = np.broadcast_to(
        geometry[np.newaxis, np.newaxis, :],
        (1, GLYPH_HEIGHT, GEOMETRY_FEATURE_COUNT),
    ).copy()
    return np.concatenate((normalized, columns), axis=2).astype(np.float32)


def isolate_glyphs_shape_and_geometry(raster: np.ndarray) -> tuple[np.ndarray, ...]:
    foreground = _filtered_foreground(raster)
    active = np.any(foreground, axis=0)
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index if value and index == len(active) - 1 else index - 1
            intervals.append((start, end))
            start = None
    glyphs: list[np.ndarray] = []
    for left, right in intervals:
        local_mask = foreground[:, left : right + 1]
        if int(local_mask.sum()) < 2:
            continue
        local_raster = raster[:, left : right + 1]
        glyphs.append(_encode_shape_and_geometry(local_raster, local_mask, right - left + 1))
    return tuple(glyphs)


__all__ = [
    "ENCODED_GLYPH_WIDTH",
    "GEOMETRY_FEATURE_COUNT",
    "isolate_glyphs_shape_and_geometry",
]
