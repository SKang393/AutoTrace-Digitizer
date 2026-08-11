# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Differentiable-binarization losses for graph text detector V5."""

from __future__ import annotations

import torch

from .protocol import (
    DB_BINARY_LOSS_WEIGHT,
    DB_SHRINK_LOSS_WEIGHT,
    DB_THRESHOLD_LOSS_WEIGHT,
)


NEGATIVE_TO_POSITIVE_RATIO = 3
MINIMUM_NEGATIVE_PIXELS = 256
EMPTY_TARGET_NEGATIVE_PIXELS = 2048


def _balanced_binary_cross_entropy(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.finfo(probabilities.dtype).eps
    bounded = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    valid = mask > 0.5
    positive = (target > 0.5) & valid
    negative = (target <= 0.5) & valid
    sample_losses: list[torch.Tensor] = []
    for sample_index in range(probabilities.shape[0]):
        positive_loss = -torch.log(bounded[sample_index][positive[sample_index]])
        negative_loss = -torch.log(1.0 - bounded[sample_index][negative[sample_index]])
        negative_count = min(
            negative_loss.numel(),
            max(MINIMUM_NEGATIVE_PIXELS, positive_loss.numel() * NEGATIVE_TO_POSITIVE_RATIO)
            if positive_loss.numel() > 0
            else EMPTY_TARGET_NEGATIVE_PIXELS,
        )
        selected_negative = (
            torch.topk(negative_loss, k=negative_count, largest=True, sorted=False).values
            if negative_count > 0
            else negative_loss
        )
        denominator = max(1, positive_loss.numel() + selected_negative.numel())
        sample_losses.append((positive_loss.sum() + selected_negative.sum()) / denominator)
    return torch.stack(sample_losses).mean()


def _masked_dice(probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (mask > 0.5).to(dtype=probabilities.dtype)
    predicted = probabilities * valid
    expected = target * valid
    intersection = (predicted * expected).sum(dim=(1, 2, 3))
    denominator = predicted.sum(dim=(1, 2, 3)) + expected.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def db_objective_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shrink_target: torch.Tensor,
    shrink_mask: torch.Tensor,
    threshold_target: torch.Tensor,
    threshold_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shrink, threshold, binary = outputs
    tensors = (shrink, threshold, binary, shrink_target, shrink_mask, threshold_target, threshold_mask)
    if any(value.shape != shrink.shape for value in tensors):
        raise ValueError("DB-objective loss tensors must have identical shapes")
    shrink_loss = _balanced_binary_cross_entropy(shrink, shrink_target, shrink_mask)
    threshold_valid = (threshold_mask > 0.5).to(dtype=threshold.dtype)
    threshold_loss = (
        (torch.abs(threshold - threshold_target) * threshold_valid).sum()
        / threshold_valid.sum().clamp_min(1.0)
    )
    binary_loss = _masked_dice(binary, shrink_target, shrink_mask)
    total = (
        DB_SHRINK_LOSS_WEIGHT * shrink_loss
        + DB_THRESHOLD_LOSS_WEIGHT * threshold_loss
        + DB_BINARY_LOSS_WEIGHT * binary_loss
    )
    return total, shrink_loss, threshold_loss, binary_loss


__all__ = ["db_objective_loss"]

