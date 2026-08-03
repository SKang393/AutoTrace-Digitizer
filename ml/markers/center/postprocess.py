# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Runtime-identical 9x9 local maximum and radius-aware tiny-center NMS."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CENTER_THRESHOLD_DEFAULT = 0.36
ARTIFACT_THRESHOLD_DEFAULT = 0.35
LOCAL_MAX_WINDOW = 9
MINIMUM_RADIUS = 2.5
MINIMUM_NMS_DISTANCE = 5.0
RADIUS_NMS_SCALE = 1.25


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    radius: float
    confidence: float
    artifact_probability: float


def _local_maximum(heatmap: np.ndarray, window: int = LOCAL_MAX_WINDOW) -> np.ndarray:
    pad = window // 2
    padded = np.pad(heatmap, pad, mode="constant", constant_values=-np.inf)
    maximum = np.full_like(heatmap, -np.inf)
    for dy in range(window):
        for dx in range(window):
            maximum = np.maximum(
                maximum,
                padded[dy : dy + heatmap.shape[0], dx : dx + heatmap.shape[1]],
            )
    return heatmap >= maximum


def detect_heads(
    heads: np.ndarray,
    *,
    text_mask: np.ndarray | None = None,
    artifact_mask: np.ndarray | None = None,
    center_threshold: float = CENTER_THRESHOLD_DEFAULT,
    artifact_threshold: float = ARTIFACT_THRESHOLD_DEFAULT,
) -> tuple[Detection, ...]:
    """Decode one activated output and apply raw-mask gating at candidate centers."""
    value = np.asarray(heads, dtype=np.float32)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError("Expected activated output shaped [1, 3, height, width]")
    center, radius, artifact = value[0]
    expected_shape = center.shape

    def normalized_mask(mask: np.ndarray | None, name: str) -> np.ndarray:
        if mask is None:
            return np.zeros(expected_shape, dtype=np.float32)
        normalized = np.asarray(mask, dtype=np.float32)
        if normalized.shape != expected_shape:
            raise ValueError(f"{name} must match output spatial shape {expected_shape}")
        if not np.isfinite(normalized).all():
            raise ValueError(f"{name} values must be finite")
        return normalized

    raw_text = normalized_mask(text_mask, "text_mask")
    raw_artifact = normalized_mask(artifact_mask, "artifact_mask")
    candidate_pixels = np.argwhere(
        (center >= center_threshold)
        & (artifact < artifact_threshold)
        & _local_maximum(center)
    )
    ordered = sorted(
        candidate_pixels,
        key=lambda item: (-float(center[item[0], item[1]]), int(item[0]), int(item[1])),
    )
    accepted: list[Detection] = []
    for y, x in ordered:
        effective_artifact = max(
            float(artifact[y, x]),
            float(raw_text[y, x]),
            float(raw_artifact[y, x]),
        )
        if effective_artifact >= artifact_threshold:
            continue
        candidate_radius = max(MINIMUM_RADIUS, float(radius[y, x]))
        if any(
            math.hypot(float(x) - item.x, float(y) - item.y)
            < max(MINIMUM_NMS_DISTANCE, RADIUS_NMS_SCALE * (candidate_radius + item.radius))
            for item in accepted
        ):
            continue
        accepted.append(
            Detection(
                float(x),
                float(y),
                candidate_radius,
                float(center[y, x]),
                effective_artifact,
            )
        )
    return tuple(accepted)


__all__ = [
    "ARTIFACT_THRESHOLD_DEFAULT",
    "CENTER_THRESHOLD_DEFAULT",
    "Detection",
    "LOCAL_MAX_WINDOW",
    "MINIMUM_NMS_DISTANCE",
    "MINIMUM_RADIUS",
    "RADIUS_NMS_SCALE",
    "detect_heads",
]
