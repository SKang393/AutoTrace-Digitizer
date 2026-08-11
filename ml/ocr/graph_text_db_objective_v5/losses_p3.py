# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P3 DB objective with explicit negative supervision in the ignored boundary."""

from __future__ import annotations

import torch

from .losses import db_objective_loss


def db_objective_loss_with_boundary_negative(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shrink_target: torch.Tensor,
    shrink_mask: torch.Tensor,
    threshold_target: torch.Tensor,
    threshold_mask: torch.Tensor,
    *,
    boundary_negative_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base, shrink_loss, threshold_loss, binary_loss = db_objective_loss(
        outputs,
        shrink_target,
        shrink_mask,
        threshold_target,
        threshold_mask,
    )
    shrink = outputs[0]
    boundary = (shrink_mask <= 0.5) & (shrink_target <= 0.5)
    per_sample: list[torch.Tensor] = []
    for sample_index in range(shrink.shape[0]):
        values = shrink[sample_index][boundary[sample_index]]
        if values.numel() == 0:
            per_sample.append(shrink[sample_index].sum() * 0.0)
        else:
            per_sample.append(-torch.log1p(-values.clamp(max=1.0 - 1e-6)).mean())
    boundary_negative = torch.stack(per_sample).mean()
    total = base + boundary_negative_loss_weight * boundary_negative
    return total, shrink_loss, threshold_loss, binary_loss, boundary_negative


__all__ = ["db_objective_loss_with_boundary_negative"]
