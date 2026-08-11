# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe dual-head differentiable-binarization detector."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import DB_BINARIZATION_K, SEED


class DbObjectiveTextRegionNet(nn.Module):
    """Share a stride-4 encoder while learning shrink and threshold maps together."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.encoder2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.context_in = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.context2 = nn.Conv2d(48, 48, kernel_size=3, padding=2, dilation=2)
        self.context4 = nn.Conv2d(48, 48, kernel_size=3, padding=4, dilation=4)
        self.decoder = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.fine = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.shared = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.shrink_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.threshold_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu", generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _features(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("DB-objective detector input must be [batch,3,H,W]")
            if value.shape[2] % 4 or value.shape[3] % 4:
                raise ValueError("DB-objective detector dimensions must be divisible by four")
        level1 = self.activation(self.encoder1(value))
        level2 = self.activation(self.encoder2(level1))
        context = self.activation(self.context_in(level2))
        context = self.activation(self.context2(context) + context)
        context = self.activation(self.context4(context) + context)
        decoded = F.interpolate(context, scale_factor=2.0, mode="bilinear", align_corners=False)
        decoded = self.activation(self.decoder(decoded) + self.fine(level1))
        full = F.interpolate(decoded, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.activation(self.shared(full))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return only the production shrink-probability map."""
        shrink = torch.sigmoid(self.shrink_head(self._features(value)))
        return torch.clamp(shrink, min=0.0, max=1.0)

    def forward_training(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three DB training maps without changing the ONNX contract."""
        features = self._features(value)
        shrink = torch.clamp(torch.sigmoid(self.shrink_head(features)), min=0.0, max=1.0)
        threshold = torch.clamp(torch.sigmoid(self.threshold_head(features)), min=0.0, max=1.0)
        binary = torch.sigmoid(DB_BINARIZATION_K * (shrink - threshold))
        return shrink, threshold, binary


__all__ = ["DbObjectiveTextRegionNet"]

