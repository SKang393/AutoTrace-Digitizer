# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe probability-map model for ignore-band DB supervision."""

from __future__ import annotations

import torch
from torch import nn


class IgnoreBandTextRegionNet(nn.Module):
    """Retain the reviewed V2 topology while changing the supervision defect class."""

    def __init__(self, seed: int = 20260907) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.encoder2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.encoder3 = nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1)
        self.context1 = nn.Conv2d(48, 48, kernel_size=3, padding=2, dilation=2)
        self.context2 = nn.Conv2d(48, 48, kernel_size=3, padding=4, dilation=4)
        self.decoder2 = nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1)
        self.decoder1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.output = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        self.activation = nn.ReLU(inplace=False)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu", generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("Ignore-band detector input must be [batch,3,H,W]")
            if value.shape[2] % 8 or value.shape[3] % 8:
                raise ValueError("Ignore-band detector dimensions must be divisible by eight")
        level1 = self.activation(self.encoder1(value))
        level2 = self.activation(self.encoder2(level1))
        level3 = self.activation(self.encoder3(level2))
        context = self.activation(self.context1(level3))
        context = self.activation(self.context2(context) + level3)
        decoded2 = self.activation(self.decoder2(context) + level2)
        decoded1 = self.activation(self.decoder1(decoded2) + level1)
        return torch.clamp(torch.sigmoid(self.output(decoded1)), min=0.0, max=1.0)


__all__ = ["IgnoreBandTextRegionNet"]
