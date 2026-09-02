# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Richer export-safe proposal classifier for V33."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import CROP_HEIGHT, CROP_WIDTH, ENCODED_WIDTH, GEOMETRY_FEATURE_COUNT, INPUT_CHANNELS


class RealRangeClassifierV33(nn.Module):
    """Fuse high-resolution glyph detail, multiscale context, and geometry."""

    def __init__(self, seed: int = 20260933) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.detail = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d((2, 4)), nn.Flatten(),
        )
        self.detail_pool = nn.Sequential(nn.AvgPool2d((8, 16)), nn.Flatten())
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 48), nn.ReLU(),
            nn.Linear(48, 32), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear((32 + 64) * 4 * 8 + 32, 160), nn.ReLU(),
            nn.Dropout(p=0.10), nn.Linear(160, 2),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("V33 proposal tensor must be [proposal_count,2,32,140]")
        visual = value[:, :, :, :CROP_WIDTH]
        geometry = value[:, 0, 0, CROP_WIDTH:]
        detail = self.detail(visual)
        context = self.context(detail)
        detail_summary = self.detail_pool(detail)
        return self.classifier(torch.cat((detail_summary, context, self.geometry(geometry)), dim=1))


__all__ = ["RealRangeClassifierV33"]
