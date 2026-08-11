# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P2 graph text detector with an explicit probability clip."""

from __future__ import annotations

import torch

from .model import GraphTextRegionNet


class GraphTextRegionNetP2(GraphTextRegionNet):
    """Retain the P1 network and enforce the strict probability output contract."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(super().forward(value), min=0.0, max=1.0)


__all__ = ["GraphTextRegionNetP2"]
