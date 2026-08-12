# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Recover only source-evidenced whitespace omitted by a CTC recognizer."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from PIL import Image


MINIMUM_GAP_PIXELS = 4
MINIMUM_GAP_TO_INK_HEIGHT_RATIO = 0.25
MINIMUM_SOURCE_GROUPS = 3
FOREGROUND_CONTRAST_FRACTION = 0.30
MINIMUM_FOREGROUND_CONTRAST = 10.0


def _source_groups(image: Image.Image) -> tuple[int, ...]:
    return tuple(item[0] for item in _source_group_features(image))


def _source_group_features(image: Image.Image) -> tuple[tuple[int, int, float, float], ...]:
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
    features: list[tuple[int, int, float, float]] = []
    for start, end in zip(starts, ends, strict=True):
        group = foreground[:, start : end + 1]
        group_coordinates = np.argwhere(group)
        group_top, group_left = group_coordinates.min(axis=0)
        group_bottom, group_right = group_coordinates.max(axis=0)
        tight = group[int(group_top) : int(group_bottom) + 1, int(group_left) : int(group_right) + 1]
        group_width = int(group_right - group_left + 1)
        group_height = int(group_bottom - group_top + 1)
        top_coverage = float(tight[0].sum()) / group_width
        bottom_coverage = float(tight[-1].sum()) / group_width
        features.append((group_width, group_height, top_coverage, bottom_coverage))
    return tuple(features)


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


def restore_source_evidenced_spaces(image: Image.Image, raw_prediction: str) -> str:
    """Insert spaces only when large blank source bands provide direct evidence."""

    if len(raw_prediction) < 2 or any(character.isspace() for character in raw_prediction):
        return raw_prediction
    widths = _source_groups(image)
    counts = _partition_character_counts(len(raw_prediction), widths)
    if not counts:
        return raw_prediction
    chunks: list[str] = []
    offset = 0
    for count in counts:
        chunks.append(raw_prediction[offset : offset + count])
        offset += count
    return " ".join(chunks)


def restore_source_evidenced_spaces_and_vertical_case(image: Image.Image, raw_prediction: str) -> str:
    """P2: also distinguish a serifed capital I from a lowercase l using source pixels."""

    if len(raw_prediction) < 2 or any(character.isspace() for character in raw_prediction):
        return raw_prediction
    features = _source_group_features(image)
    widths = tuple(item[0] for item in features)
    counts = _partition_character_counts(len(raw_prediction), widths)
    if not counts:
        return raw_prediction
    chunks: list[str] = []
    offset = 0
    for count, (width, height, top_coverage, bottom_coverage) in zip(counts, features, strict=True):
        chunk = raw_prediction[offset : offset + count]
        offset += count
        if (
            chunk == "l"
            and width / max(1, height) >= 0.25
            and top_coverage >= 0.75
            and bottom_coverage >= 0.75
        ):
            chunk = "I"
        chunks.append(chunk)
    return " ".join(chunks)


__all__ = ["restore_source_evidenced_spaces", "restore_source_evidenced_spaces_and_vertical_case"]
