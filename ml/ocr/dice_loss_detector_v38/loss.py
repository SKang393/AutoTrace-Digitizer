# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""The only V38 training change: equal-weight BCE plus batch soft-Dice."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .protocol import BCE_LOSS_WEIGHT, DICE_EPSILON, DICE_LOSS_WEIGHT, POSITIVE_WEIGHT


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the V37 positive-weighted BCE term without changing its reduction."""
    positive_weight = torch.tensor([POSITIVE_WEIGHT], dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)


def batch_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = DICE_EPSILON,
) -> torch.Tensor:
    """Return one soft-Dice loss over all samples and pixels in the batch."""
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum()
    denominator = probabilities.sum() + target.sum()
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon))


def composite_pixel_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, unchanged V37 BCE, and deterministic batch Dice terms."""
    bce = weighted_bce_loss(logits, target)
    dice = batch_soft_dice_loss(logits, target)
    return (BCE_LOSS_WEIGHT * bce) + (DICE_LOSS_WEIGHT * dice), bce, dice


__all__ = ["batch_soft_dice_loss", "composite_pixel_loss", "weighted_bce_loss"]
