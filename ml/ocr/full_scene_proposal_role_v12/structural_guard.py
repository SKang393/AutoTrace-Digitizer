# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen structural rejection rule for OCR V12 candidate P3."""

from __future__ import annotations

from ml.ocr.component_region_detector_v6.dataset import Component


MAXIMUM_INK_DENSITY = 0.30
MINIMUM_WIDTH_HEIGHT_RATIO = 2.75
REQUIRED_COMPONENT_COUNT = 1


def ink_density(candidate: Component) -> float:
    return candidate.area / max(1, candidate.width * candidate.height)


def width_height_ratio(candidate: Component) -> float:
    return candidate.width / max(1, candidate.height)


def is_rejected_structure(candidate: Component) -> bool:
    """Reject only an isolated, sparse, long horizontal stroke group."""

    return (
        candidate.count == REQUIRED_COMPONENT_COUNT
        and ink_density(candidate) <= MAXIMUM_INK_DENSITY
        and width_height_ratio(candidate) >= MINIMUM_WIDTH_HEIGHT_RATIO
    )


__all__ = [
    "MAXIMUM_INK_DENSITY",
    "MINIMUM_WIDTH_HEIGHT_RATIO",
    "REQUIRED_COMPONENT_COUNT",
    "ink_density",
    "is_rejected_structure",
    "width_height_ratio",
]
