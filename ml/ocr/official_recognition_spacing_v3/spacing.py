# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Insert only strongly source-evidenced spaces without rewriting glyphs."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from PIL import Image


MINIMUM_GAP_PIXELS = 5
MINIMUM_GAP_TO_INK_HEIGHT_RATIO = 0.40
MINIMUM_SOURCE_GROUPS = 2
FOREGROUND_CONTRAST_FRACTION = 0.30
MINIMUM_FOREGROUND_CONTRAST = 10.0


def _source_group_widths(image: Image.Image) -> tuple[int, ...]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) == 0:
        return ()
    edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(edge))
    contrast = max(0.0, background - float(np.percentile(gray, 1.0)))
    threshold = max(MINIMUM_FOREGROUND_CONTRAST, contrast * FOREGROUND_CONTRAST_FRACTION)
    foreground = gray <= background - threshold
    coordinates = np.argwhere(foreground)
    if len(coordinates) == 0:
        return ()
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    foreground = foreground[int(top) : int(bottom) + 1, int(left) : int(right) + 1]
    active = np.flatnonzero(foreground.any(axis=0))
    if len(active) == 0:
        return ()
    ink_height = int(bottom - top + 1)
    large_gap = max(MINIMUM_GAP_PIXELS, int(math.ceil(ink_height * MINIMUM_GAP_TO_INK_HEIGHT_RATIO)))
    starts = [int(active[0])]
    ends: list[int] = []
    for prior, current in zip(active[:-1], active[1:], strict=True):
        if int(current - prior - 1) >= large_gap:
            ends.append(int(prior))
            starts.append(int(current))
    ends.append(int(active[-1]))
    return tuple(end - start + 1 for start, end in zip(starts, ends, strict=True))


def _partition_character_counts(character_count: int, widths: tuple[int, ...]) -> tuple[int, ...]:
    group_count = len(widths)
    if group_count < MINIMUM_SOURCE_GROUPS or group_count > character_count:
        return ()
    total_width = float(sum(widths))
    cumulative_widths = np.cumsum(np.asarray(widths[:-1], dtype=np.float64)) / total_width
    best: tuple[float, tuple[int, ...]] | None = None
    for boundaries in combinations(range(1, character_count), group_count - 1):
        normalized = np.asarray(boundaries, dtype=np.float64) / character_count
        cost = float(np.square(normalized - cumulative_widths).sum())
        counts = tuple(
            right - left
            for left, right in zip((0, *boundaries), (*boundaries, character_count), strict=True)
        )
        candidate = (cost, counts)
        if best is None or candidate < best:
            best = candidate
    return () if best is None else best[1]


def restore_conservative_source_spaces(image: Image.Image, raw_prediction: str) -> str:
    """Reconstruct source whitespace while preserving every recognized nonspace scalar."""

    characters = "".join(character for character in raw_prediction if not character.isspace())
    if len(characters) < 2:
        return raw_prediction
    counts = _partition_character_counts(len(characters), _source_group_widths(image))
    if not counts:
        return raw_prediction
    chunks: list[str] = []
    offset = 0
    for count in counts:
        chunks.append(characters[offset : offset + count])
        offset += count
    return " ".join(chunks)


__all__ = ["restore_conservative_source_spaces"]

