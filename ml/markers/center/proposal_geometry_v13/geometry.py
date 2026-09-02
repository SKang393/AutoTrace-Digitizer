# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic geometry veto for thin connected components."""

from __future__ import annotations

import math

import numpy as np
import torch

from ml.markers.center.line_aware_v1.pipeline import ProposalBatch

GEOMETRY_REVISION = "compact-isotropic-ink-veto-v1"
MASK_THRESHOLD = 0.35
INK_THRESHOLD = 0.12
MAX_EIGEN_RATIO = 12.0
MIN_AXIS_BALANCE = 0.25


def _keeps_coordinate(tensor: torch.Tensor, x: int, y: int) -> bool:
    height, width = tensor.shape[1:]
    if not (0 <= x < width and 0 <= y < height):
        return False
    if float(tensor[1, max(0, y - 2):min(height, y + 3), max(0, x - 2):min(width, x + 3)].max()) >= MASK_THRESHOLD:
        return False
    if float(tensor[2, max(0, y - 2):min(height, y + 3), max(0, x - 2):min(width, x + 3)].max()) >= MASK_THRESHOLD:
        return False
    radius = 12
    top, bottom = max(0, y - radius), min(height, y + radius + 1)
    left, right = max(0, x - radius), min(width, x + radius + 1)
    ink = tensor[0, top:bottom, left:right].detach().cpu().numpy() >= INK_THRESHOLD
    yy, xx = np.nonzero(ink)
    if len(xx) < 5:
        return False
    dx, dy = xx.astype(np.float64) + left - x, yy.astype(np.float64) + top - y
    weights = tensor[0, top:bottom, left:right].detach().cpu().numpy()[ink].astype(np.float64)
    weights = np.maximum(weights, 1e-6)
    covariance = np.cov(np.stack((dx, dy)), aweights=weights, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    ratio = float((eigenvalues[-1] + 1e-6) / (eigenvalues[0] + 1e-6))
    extent_x = int(xx.max() - xx.min() + 1)
    extent_y = int(yy.max() - yy.min() + 1)
    balance = min(extent_x, extent_y) / max(extent_x, extent_y)
    return math.isfinite(ratio) and ratio <= MAX_EIGEN_RATIO and balance >= MIN_AXIS_BALANCE


def filter_proposals(tensor: torch.Tensor, proposals: ProposalBatch) -> ProposalBatch:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected [3,height,width] tensor")
    keep = [
        index for index, coordinate in enumerate(proposals.coordinates.tolist())
        if _keeps_coordinate(tensor, int(round(coordinate[0])), int(round(coordinate[1])))
    ]
    indices = torch.tensor(keep, dtype=torch.long)
    return ProposalBatch(proposals.patches.index_select(0, indices), proposals.coordinates.index_select(0, indices))


__all__ = ["GEOMETRY_REVISION", "filter_proposals"]
