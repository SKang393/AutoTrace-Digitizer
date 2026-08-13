# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Compact export-safe CNN for O/o/l/I source glyphs."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import GLYPHS, IMAGE_SIZE


class AmbiguityGlyphNet(nn.Module):
    def __init__(self, seed: int = 20261317) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 3 * 3, 64), nn.GELU(), nn.Linear(64, len(GLYPHS)),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Ambiguity glyph input must be [batch,1,24,24]")
        return self.classifier(self.features(value))


__all__ = ["AmbiguityGlyphNet"]
