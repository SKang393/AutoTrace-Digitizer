# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P2 export-only logit scaling over the exact P1 classifier."""

from __future__ import annotations

import torch
from torch import nn

from .model import LineContextAmbiguityNet


class ParityScaledLineContextNet(nn.Module):
    def __init__(self, base: LineContextAmbiguityNet) -> None:
        super().__init__()
        self.base = base

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) * 0.5


__all__ = ["ParityScaledLineContextNet"]
