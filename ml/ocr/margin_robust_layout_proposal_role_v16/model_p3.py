# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Output-scaled proposal-only repair candidate for OCR V16 P3."""

from __future__ import annotations

import torch
from torch import nn

from .model import MarginRobustLayoutProposalRoleNet


class OutputScaledMarginCandidate(nn.Module):
    """Apply one fixed export scale without changing proposal or role ordering."""

    def __init__(self, base: MarginRobustLayoutProposalRoleNet, *, output_scale: float) -> None:
        super().__init__()
        if not 0.0 < output_scale <= 1.0:
            raise ValueError("OCR V16 P3 output scale must be in (0,1]")
        self.base = base
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) * self.output_scale


__all__ = ["OutputScaledMarginCandidate"]
