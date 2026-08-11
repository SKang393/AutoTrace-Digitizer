# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe stride-4 probability-map model."""

from __future__ import annotations

import torch
from torch import nn


class Stride4TextRegionNet(nn.Module):
    """Preserve fine glyph strokes with a fourfold bottleneck and shallow skip."""

    def __init__(self, seed: int = 20260911) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.encoder1 = nn.Conv2d(3, 12, kernel_size=5, stride=2, padding=2)
        self.encoder2 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.context_in = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.context2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.context4 = nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.context8 = nn.Conv2d(32, 32, kernel_size=3, padding=8, dilation=8)
        self.decoder = nn.ConvTranspose2d(32, 12, kernel_size=4, stride=2, padding=1)
        self.fine = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.output = nn.ConvTranspose2d(12, 1, kernel_size=4, stride=2, padding=1)
        self.activation = nn.ReLU(inplace=False)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu", generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("Stride-4 detector input must be [batch,3,H,W]")
            if value.shape[2] % 4 or value.shape[3] % 4:
                raise ValueError("Stride-4 detector dimensions must be divisible by four")
        level1 = self.activation(self.encoder1(value))
        level2 = self.activation(self.encoder2(level1))
        context = self.activation(self.context_in(level2))
        context = self.activation(self.context2(context) + context)
        context = self.activation(self.context4(context) + context)
        context = self.activation(self.context8(context) + context)
        decoded = self.activation(self.decoder(context) + self.fine(level1))
        return torch.clamp(torch.sigmoid(self.output(decoded)), min=0.0, max=1.0)


__all__ = ["Stride4TextRegionNet"]
