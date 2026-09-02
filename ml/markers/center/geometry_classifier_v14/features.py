# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic geometry features used without learned weights."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ml.markers.center.line_aware_v1.pipeline import ProposalBatch


@dataclass(frozen=True)
class GeometryFeatures:
    compactness: float
    isotropy: float
    radial_support: float
    line_evidence: float
    mask_clear: float
    score: float
    radius: float


def _sample(ink: np.ndarray, x: float, y: float) -> float:
    height, width = ink.shape
    ix, iy = int(round(x)), int(round(y))
    if not (0 <= ix < width and 0 <= iy < height):
        return 0.0
    return float(ink[iy, ix])


def score_proposal(tensor: torch.Tensor, x: int, y: int) -> GeometryFeatures:
    ink = tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
    text = tensor[1].detach().cpu().numpy().astype(np.float32, copy=False)
    artifact = tensor[2].detach().cpu().numpy().astype(np.float32, copy=False)
    height, width = ink.shape
    top, bottom = max(0, y - 12), min(height, y + 13)
    left, right = max(0, x - 12), min(width, x + 13)
    local = ink[top:bottom, left:right]
    binary = local >= 0.12
    yy, xx = np.nonzero(binary)
    if len(xx) < 5:
        return GeometryFeatures(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0)
    extent_x = float(xx.max() - xx.min() + 1)
    extent_y = float(yy.max() - yy.min() + 1)
    compactness = float(len(xx) / max(1.0, extent_x * extent_y))
    isotropy = float(min(extent_x, extent_y) / max(extent_x, extent_y))
    distances = np.hypot(xx + left - x, yy + top - y)
    radius = float(np.clip(np.percentile(distances, 75), 3.0, 12.0))
    radial_values = []
    for ring in range(3, 13):
        values = [_sample(ink, x + ring * math.cos(angle), y + ring * math.sin(angle)) for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
        radial_values.append(float(np.mean(values)))
    radial_support = float(np.clip(max(radial_values), 0.0, 1.0))
    line_evidence = float(np.clip((1.0 - isotropy) * (1.0 - compactness), 0.0, 1.0))
    mask_max = max(float(text[max(0, y - 2):min(height, y + 3), max(0, x - 2):min(width, x + 3)].max()), float(artifact[max(0, y - 2):min(height, y + 3), max(0, x - 2):min(width, x + 3)].max()))
    mask_clear = float(1.0 - np.clip(mask_max, 0.0, 1.0))
    score = float(np.clip(0.35 * radial_support + 0.25 * compactness + 0.20 * isotropy + 0.20 * mask_clear - 0.35 * line_evidence, 0.0, 1.0))
    return GeometryFeatures(compactness, isotropy, radial_support, line_evidence, mask_clear, score, radius)


def score_proposals(tensor: torch.Tensor, proposals: ProposalBatch) -> tuple[GeometryFeatures, ...]:
    return tuple(score_proposal(tensor, int(round(x)), int(round(y))) for x, y in proposals.coordinates.tolist())


__all__ = ["GeometryFeatures", "score_proposal", "score_proposals"]
