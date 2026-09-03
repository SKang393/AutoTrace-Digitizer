# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""The sole V21 objective change: fixed binary focal classification loss."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def binary_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Return unreduced binary focal loss using the RetinaNet defaults.

    ``alpha`` and ``gamma`` are fixed preregistered constants. This wrapper
    deliberately does not choose them from train/dev observations.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    targets = targets.to(dtype=logits.dtype)
    bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * torch.pow(1.0 - p_t, gamma) * bce


def regression_terms(raw: Tensor, labels: Tensor, offsets: Tensor, radii: Tensor) -> tuple[Tensor, Tensor]:
    """Keep V16/V20 offset and radius terms byte-for-byte in formulation."""
    positive = labels > 0.5
    if torch.any(positive):
        offset = functional.smooth_l1_loss(torch.tanh(raw[positive, 1:3]) * 0.75, offsets[positive])
        radius = functional.smooth_l1_loss(
            2.5 + torch.sigmoid(raw[positive, 3]) * 5.5,
            radii[positive].clamp(2.5, 8.0),
        )
    else:
        offset = raw[:, 1:3].sum() * 0
        radius = raw[:, 3].sum() * 0
    return offset, radius


def v21_loss(
    raw: Tensor,
    labels: Tensor,
    offsets: Tensor,
    radii: Tensor,
    hard: Tensor,
    *,
    positive_weight: float,
    hard_weight: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Apply focal classification with the unchanged V20 sample weights."""
    weights = torch.where(
        labels > 0.5,
        torch.full_like(labels, positive_weight),
        torch.where(hard, torch.full_like(labels, hard_weight), torch.ones_like(labels)),
    )
    classification = (binary_focal_loss_with_logits(raw[:, 0], labels, alpha=alpha, gamma=gamma) * weights).mean()
    offset, radius = regression_terms(raw, labels, offsets, radii)
    return classification + 1.25 * offset + 0.25 * radius


__all__ = ["binary_focal_loss_with_logits", "regression_terms", "v21_loss"]
