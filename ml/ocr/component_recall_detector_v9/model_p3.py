# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Exact P2 checkpoint with a fixed output-logit scale for V9 P3."""

from __future__ import annotations

import torch
from torch import nn

from .model import ComponentRecallNet


OUTPUT_LOGIT_SCALE = 0.5


class ScaledComponentRecallNet(nn.Module):
    def __init__(self, base: ComponentRecallNet) -> None:
        super().__init__()
        self.base = base

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) * OUTPUT_LOGIT_SCALE


__all__ = ["OUTPUT_LOGIT_SCALE", "ScaledComponentRecallNet"]
