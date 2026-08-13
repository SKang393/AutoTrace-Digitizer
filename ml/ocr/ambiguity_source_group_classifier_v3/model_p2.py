# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

import torch
from torch import nn

from .model import SourceGroupAmbiguityNet


class ParityScaledSourceGroupNet(nn.Module):
    """Preserve exact P1 weights and reduce only the exported logit scale."""

    def __init__(self, base: SourceGroupAmbiguityNet) -> None:
        super().__init__()
        self.base = base

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) * 0.125


__all__ = ["ParityScaledSourceGroupNet"]
