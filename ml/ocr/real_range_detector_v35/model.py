# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Small export-safe source-scale segmentation model."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import INPUT_CHANNELS, TILE_SIZE


class SourceScaleProposalNet(nn.Module):
    """Predict raw text-region logits while preserving full-resolution detail."""

    def __init__(self, seed: int = 20260935) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.detail = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(72, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, TILE_SIZE, TILE_SIZE):
            raise ValueError(f"V35 input must be [tile_count,{INPUT_CHANNELS},{TILE_SIZE},{TILE_SIZE}]")
        detail = self.detail(value)
        context = self.context(detail)
        context = torch.nn.functional.interpolate(context, size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False)
        return self.head(torch.cat((detail, context), dim=1))


__all__ = ["SourceScaleProposalNet"]
