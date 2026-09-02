# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic V34 raw proposal expansion over the committed V32 scenes."""

from __future__ import annotations

from ml.ocr.component_context_detector_v7.dataset import proposals
from ml.ocr.component_region_detector_v6.dataset import Component, group_lines

import numpy as np

from ml.ocr.real_range_classifier_finetune_v32.dataset import SceneSample, build_split

from .protocol import EXPANSION_MARGIN_PIXELS


def _contrast(gray: np.ndarray) -> np.ndarray:
    low, high = (float(value) for value in np.percentile(gray, (5.0, 95.0)))
    if high <= low:
        return gray.copy()
    return np.rint(np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0.0, 255.0)).astype(np.uint8)


def _expanded(component: Component, margin: int, width: int, height: int) -> Component:
    return Component(
        max(0, component.left - margin),
        max(0, component.top - margin),
        min(width - 1, component.right + margin),
        min(height - 1, component.bottom + margin),
        component.area,
        component.count,
    )


def _deduplicate(items: tuple[Component, ...]) -> tuple[Component, ...]:
    unique: dict[tuple[int, int, int, int], Component] = {}
    for item in items:
        unique.setdefault((item.left, item.top, item.right, item.bottom), item)
    return tuple(sorted(unique.values(), key=lambda item: (item.top, item.left, item.bottom, item.right)))


def repaired_proposals(gray: np.ndarray) -> tuple[Component, ...]:
    """Add deterministic contrast and one-pixel grouping candidates to V32."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("V34 proposal source must be uint8 Gray8")
    height, width = gray.shape
    base = proposals(gray)
    contrast = proposals(_contrast(gray))
    expanded = tuple(_expanded(item, EXPANSION_MARGIN_PIXELS, width, height) for item in (*base, *contrast))
    grouped = group_lines((*base, *contrast, *expanded))
    return _deduplicate((*base, *contrast, *expanded, *grouped))


__all__ = ["SceneSample", "build_split", "repaired_proposals"]
