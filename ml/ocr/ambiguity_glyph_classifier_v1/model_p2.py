# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Profile-preserving export-safe P2 classifier for O/o/l/I glyphs."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import GLYPHS, IMAGE_SIZE


class ProfileAwareAmbiguityGlyphNet(nn.Module):
    def __init__(self, seed: int = 20261318) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((3, 3)),
        )
        profile_count = IMAGE_SIZE * 2 + 4
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 3 + profile_count, 96), nn.ReLU(), nn.Linear(96, len(GLYPHS)),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Profile-aware ambiguity input must be [batch,1,24,24]")
        convolution = self.convolution(value).flatten(1)
        row_profile = value.mean(dim=3).squeeze(1)
        column_profile = value.mean(dim=2).squeeze(1)
        row_peak = value.amax(dim=3).squeeze(1)
        column_peak = value.amax(dim=2).squeeze(1)
        height = (row_peak > 0.08).float().mean(dim=1, keepdim=True)
        width = (column_peak > 0.08).float().mean(dim=1, keepdim=True)
        top = row_peak[:, : IMAGE_SIZE // 2].mean(dim=1, keepdim=True)
        bottom = row_peak[:, IMAGE_SIZE // 2 :].mean(dim=1, keepdim=True)
        features = torch.cat((convolution, row_profile, column_profile, height, width, top, bottom), dim=1)
        return self.classifier(features)


__all__ = ["ProfileAwareAmbiguityGlyphNet"]

