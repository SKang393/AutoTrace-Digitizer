# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Anchor-preserving affine output calibration for OCR V15 P3."""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import LayoutConditionedProposalRoleNet


class AnchorScaledCandidate(nn.Module):
    """Scale logits while preserving one fixed proposal-probability boundary."""

    def __init__(self, base: LayoutConditionedProposalRoleNet, scale: float, anchor: float) -> None:
        super().__init__()
        if not 0.0 < scale < 1.0:
            raise ValueError("OCR V15 P3 output scale must be between zero and one")
        if not 0.5 < anchor < 1.0:
            raise ValueError("OCR V15 P3 anchor threshold must be between 0.5 and one")
        self.base = base
        self.scale = float(scale)
        anchor_logit = math.log(anchor / (1.0 - anchor))
        offset = (1.0 - scale) * anchor_logit
        self.register_buffer(
            "proposal_bias",
            torch.tensor((-offset / 2.0, offset / 2.0), dtype=torch.float32),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.base(value)
        proposal = output[:, :2] * self.scale + self.proposal_bias
        roles = output[:, 2:] * self.scale
        return torch.cat((proposal, roles), dim=1)


__all__ = ["AnchorScaledCandidate"]
