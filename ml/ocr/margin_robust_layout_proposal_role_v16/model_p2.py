# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed proposal-logit calibration candidate for OCR V16 P2."""

from __future__ import annotations

import torch
from torch import nn

from .model import MarginRobustLayoutProposalRoleNet


class CalibratedMarginCandidate(nn.Module):
    """Preserve P1 weights and roles while shifting only the accepted-proposal logit."""

    def __init__(self, base: MarginRobustLayoutProposalRoleNet, *, positive_logit_bias: float) -> None:
        super().__init__()
        if positive_logit_bias >= 0.0:
            raise ValueError("OCR V16 P2 calibration bias must be negative")
        self.base = base
        self.register_buffer("positive_logit_bias", torch.tensor(float(positive_logit_bias)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.base(value)
        proposal = torch.cat((output[:, :1], output[:, 1:2] + self.positive_logit_bias), dim=1)
        return torch.cat((proposal, output[:, 2:]), dim=1)


__all__ = ["CalibratedMarginCandidate"]
