# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P2 absolute-scale glyph encoding derived only from the failed P1 validation defect."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .dataset import CANVAS_HEIGHT, CANVAS_WIDTH, _filtered_foreground
from .protocol import GLYPH_HEIGHT, GLYPH_WIDTH


def _normalize_absolute_scale(raster: np.ndarray) -> np.ndarray:
    """Preserve component height and vertical position while normalizing only its cell width."""

    if raster.ndim != 2 or raster.shape[0] != CANVAS_HEIGHT or raster.shape[1] < 1:
        raise ValueError("P2 glyph source must retain the complete 32-row label canvas")
    source = Image.fromarray(raster, mode="L")
    resized = source.resize((GLYPH_WIDTH, GLYPH_HEIGHT), resample=Image.Resampling.BILINEAR)
    value = 1.0 - np.asarray(resized, dtype=np.float32) / 255.0
    maximum = float(value.max())
    if maximum > 0:
        value /= maximum
    return value[np.newaxis, :, :].astype(np.float32)


def isolate_glyphs_absolute_scale(raster: np.ndarray) -> tuple[np.ndarray, ...]:
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
    return tuple(
        _normalize_absolute_scale(raster[:, left : right + 1])
        for left, right in intervals
        if int(foreground[:, left : right + 1].sum()) >= 2
    )


__all__ = ["isolate_glyphs_absolute_scale"]
