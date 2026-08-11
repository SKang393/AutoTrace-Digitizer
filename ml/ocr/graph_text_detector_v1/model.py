# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Small export-safe encoder-decoder for graph text probability maps."""

from __future__ import annotations

import torch
from torch import nn


class GraphTextRegionNet(nn.Module):
    """Predict a filled text-region probability map at the input resolution."""

    def __init__(self, seed: int = 20260901) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(24, 24, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu", generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("Graph text detector input must be [batch,3,H,W]")
            if value.shape[2] % 8 or value.shape[3] % 8:
                raise ValueError("Graph text detector height and width must be divisible by eight")
        return self.network(value)


__all__ = ["GraphTextRegionNet"]
