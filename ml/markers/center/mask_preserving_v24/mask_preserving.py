# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Proposal and postprocessing changes for mask-preserving V24."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.markers.center.line_aware_v1.pipeline import MATCH_TOLERANCE, MarkerPrediction, ProposalBatch
PATCH_SIZE = 33
STRIDE = 4
INK_SUPPORT_WINDOW = 17
INK_SUPPORT_THRESHOLD = 0.11
INK_THRESHOLD = 0.12


def extract_proposals(tensor: torch.Tensor) -> ProposalBatch:
    """Extract every ink-supported grid proposal, retaining both mask channels."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("expected [3,height,width] tensor")
    height, width = tensor.shape[1:]
    patches = F.unfold(tensor.unsqueeze(0), kernel_size=PATCH_SIZE,
                       padding=PATCH_SIZE // 2, stride=STRIDE).squeeze(0).transpose(0, 1)
    patches = patches.reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)
    support = F.max_pool2d(tensor[0:1].unsqueeze(0), kernel_size=INK_SUPPORT_WINDOW,
                           stride=STRIDE, padding=INK_SUPPORT_WINDOW // 2).flatten()
    grid_width = math.ceil(width / STRIDE)
    indices = torch.nonzero(support >= INK_SUPPORT_THRESHOLD, as_tuple=False).flatten()
    if not len(indices):
        return ProposalBatch(patches[:0], torch.empty((0, 2), dtype=torch.float32))
    y = torch.div(indices, grid_width, rounding_mode="floor") * STRIDE
    x = torch.remainder(indices, grid_width) * STRIDE
    return ProposalBatch(patches.index_select(0, indices), torch.stack((x, y), dim=1).float())


def _consensus(scene: Any, x: float, y: float) -> bool:
    ix, iy = int(round(x)), int(round(y)); ink = scene.tensor[0]
    density = float(ink[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3].mean())
    if density >= 0.28: return True
    return any(sum(1 for px, py in ((ix-r,iy),(ix+r,iy),(ix,iy-r),(ix,iy+r),(ix-r,iy-r),(ix+r,iy-r),(ix-r,iy+r),(ix+r,iy+r)) if 0 <= px < ink.shape[1] and 0 <= py < ink.shape[0] and float(ink[py,px]) >= INK_THRESHOLD) >= 3 for r in range(3, 13))


def postprocess(scene: Any, proposals: ProposalBatch, output: np.ndarray) -> tuple[MarkerPrediction, ...]:
    if output.shape != (len(proposals.patches), 4): raise ValueError("expected [N,4] output")
    candidates = []
    for i in np.flatnonzero(output[:, 0] >= 0.25):
        bx, by = proposals.coordinates[i].tolist()
        x, y = float(bx + output[i, 1] * 4.0), float(by + output[i, 2] * 4.0)
        radius = float(np.clip(output[i, 3], 2.5, 8.0))
        if _consensus(scene, x, y): candidates.append(MarkerPrediction(x, y, radius, float(output[i, 0])))
    accepted = []
    for candidate in sorted(candidates, key=lambda p: (-p.confidence, p.y, p.x)):
        if any(math.hypot(candidate.x-p.x, candidate.y-p.y) < max(5.0, 1.25*max(candidate.radius,p.radius)) for p in accepted): continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda p: (p.y, p.x, -p.confidence)))


def prohibited_hits(predictions: tuple[MarkerPrediction, ...], scene: Any) -> Counter[str]:
    hits: Counter[str] = Counter()
    for prediction in predictions:
        for kind, x, y in scene.hard_negatives:
            if math.hypot(prediction.x-x, prediction.y-y) <= MATCH_TOLERANCE: hits[kind] += 1
    return hits
