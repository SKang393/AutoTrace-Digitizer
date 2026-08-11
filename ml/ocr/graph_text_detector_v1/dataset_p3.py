# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P3 training targets aligned to the fixed DB unclip geometry."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import cv2
import numpy as np

from .dataset_p2 import (
    P2_PATCH_HEIGHT,
    P2_PATCH_WIDTH,
    _production_resize,
    _render_source,
    _rng as p2_rng,
)
from .protocol import TRAIN_SAMPLE_COUNT


P3_SEED = 20260903
P3_PATCH_WIDTH = P2_PATCH_WIDTH
P3_PATCH_HEIGHT = P2_PATCH_HEIGHT
P3_SHRINK_RATIO = 0.40
P3_RENDERER_FAMILY = "production-scale-db-shrink-context-crops-v3"
P3_DEGRADATION_FAMILY = "source-render-production-resize-targeted-hard-negatives-v3"


@dataclass(frozen=True)
class DbShrinkPatch:
    sample_id: str
    kind: str
    bgr: np.ndarray
    target: np.ndarray
    renderer_family: str
    degradation_family: str


def _shrink_target(source_target: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(source_target)
    if len(xs) == 0:
        return np.zeros_like(source_target)
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    width = right - left
    height = bottom - top
    area = float(width * height)
    perimeter = float(2 * (width + height))
    distance = area * (1.0 - (P3_SHRINK_RATIO * P3_SHRINK_RATIO)) / perimeter
    shrunk_left = int(math.ceil(left + distance))
    shrunk_top = int(math.ceil(top + distance))
    shrunk_right = int(math.floor(right - distance))
    shrunk_bottom = int(math.floor(bottom - distance))
    if shrunk_right <= shrunk_left or shrunk_bottom <= shrunk_top:
        raise RuntimeError("P3 DB shrink target collapsed")
    result = np.zeros_like(source_target)
    result[shrunk_top:shrunk_bottom, shrunk_left:shrunk_right] = 255
    return result


def _text_crop_bounds(full_target: np.ndarray, index: int) -> tuple[int, int]:
    ys, xs = np.nonzero(full_target)
    if len(xs) == 0:
        raise RuntimeError("P3 text target disappeared during production resize")
    maximum_left = full_target.shape[1] - P3_PATCH_WIDTH
    maximum_top = full_target.shape[0] - P3_PATCH_HEIGHT
    rng = p2_rng(index)
    center_x = int(round((float(xs.min()) + float(xs.max())) / 2.0)) + int(rng.integers(-32, 33))
    center_y = int(round((float(ys.min()) + float(ys.max())) / 2.0)) + int(rng.integers(-20, 21))
    left = min(max(0, center_x - (P3_PATCH_WIDTH // 2)), maximum_left)
    top = min(max(0, center_y - (P3_PATCH_HEIGHT // 2)), maximum_top)
    if xs.min() < left or xs.max() >= left + P3_PATCH_WIDTH:
        left = min(max(0, int(xs.max()) - P3_PATCH_WIDTH + 8), maximum_left)
    if ys.min() < top or ys.max() >= top + P3_PATCH_HEIGHT:
        top = min(max(0, int(ys.max()) - P3_PATCH_HEIGHT + 8), maximum_top)
    return left, top


def _exclusion_crop_bounds(structure_family: str) -> tuple[int, int]:
    if structure_family == "compact_legend_and_arrow":
        return 512, 96
    if structure_family == "bracket_and_intersection":
        return 512, 80
    if structure_family == "paired_series_with_open_markers":
        return 256, 96
    if structure_family == "offset_phase_divider":
        return 256, 64
    raise RuntimeError(f"Unknown P3 hard-negative structure family: {structure_family}")


def render_db_shrink_patch(index: int) -> DbShrinkPatch:
    if not 0 <= index < TRAIN_SAMPLE_COUNT:
        raise ValueError("P3 training patch index is out of range")
    source_bgr, source_target, kind, degradation = _render_source(index)
    structure_family = degradation.split(":", 1)[0]
    detector_bgr = _production_resize(source_bgr, cv2.INTER_LINEAR)
    full_target = _production_resize(source_target, cv2.INTER_NEAREST)
    detector_target = _production_resize(_shrink_target(source_target), cv2.INTER_NEAREST)
    if kind == "text":
        left, top = _text_crop_bounds(full_target, index)
    else:
        left, top = _exclusion_crop_bounds(structure_family)
    right = left + P3_PATCH_WIDTH
    bottom = top + P3_PATCH_HEIGHT
    if right > detector_bgr.shape[1] or bottom > detector_bgr.shape[0]:
        raise RuntimeError("P3 deterministic crop exceeds the production tensor")
    return DbShrinkPatch(
        sample_id=f"graph-text-detector-v1-p3-train-{index:05d}",
        kind=kind,
        bgr=np.ascontiguousarray(detector_bgr[top:bottom, left:right, :]),
        target=np.ascontiguousarray(detector_target[top:bottom, left:right]),
        renderer_family=P3_RENDERER_FAMILY,
        degradation_family=f"{P3_DEGRADATION_FAMILY}:{degradation}",
    )


def build_p3_training_arrays() -> tuple[np.ndarray, np.ndarray]:
    samples = [render_db_shrink_patch(index) for index in range(TRAIN_SAMPLE_COUNT)]
    return (
        np.stack([sample.bgr for sample in samples]).astype(np.uint8),
        np.stack([sample.target for sample in samples])[:, None, :, :].astype(np.uint8),
    )


def p3_training_split_fingerprint() -> str:
    records: list[dict[str, object]] = []
    for index in range(TRAIN_SAMPLE_COUNT):
        sample = render_db_shrink_patch(index)
        records.append(
            {
                "sample_id": sample.sample_id,
                "kind": sample.kind,
                "bgr_sha256": sha256(sample.bgr.tobytes(order="C")).hexdigest(),
                "target_sha256": sha256(sample.target.tobytes(order="C")).hexdigest(),
                "renderer_family": sample.renderer_family,
                "degradation_family": sample.degradation_family,
            }
        )
    return sha256((json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


__all__ = [
    "P3_DEGRADATION_FAMILY",
    "P3_PATCH_HEIGHT",
    "P3_PATCH_WIDTH",
    "P3_RENDERER_FAMILY",
    "P3_SEED",
    "P3_SHRINK_RATIO",
    "DbShrinkPatch",
    "build_p3_training_arrays",
    "p3_training_split_fingerprint",
    "render_db_shrink_patch",
]
