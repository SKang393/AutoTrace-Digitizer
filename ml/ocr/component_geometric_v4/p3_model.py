# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P3 component-geometric classifier with explicit source-geometry features."""

from __future__ import annotations

import torch
from torch import nn

from .model import ComponentGeometricGlyphNet
from .p3_dataset import ENCODED_GLYPH_WIDTH
from .protocol import CLASS_COUNT, GLYPH_WIDTH


class ScaleAwareComponentGeometricGlyphNet(ComponentGeometricGlyphNet):
    """Retain normalized glyph shape and append four source-geometry scalars."""

    FEATURE_COUNT = ComponentGeometricGlyphNet.FEATURE_COUNT + (ENCODED_GLYPH_WIDTH - GLYPH_WIDTH)

    def __init__(self, seed: int = 20260812) -> None:
        super().__init__(seed=seed)
        generator = torch.Generator().manual_seed(seed)
        self.classifier = nn.Sequential(
            nn.Linear(self.FEATURE_COUNT, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, CLASS_COUNT),
        )
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)

    def features(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 1 or value.shape[3] != ENCODED_GLYPH_WIDTH:
            raise ValueError("P3 glyph tensor must be [glyph_count,1,24,24]")
        normalized_shape = value[:, :, :, :GLYPH_WIDTH]
        geometry = value[:, :, :, GLYPH_WIDTH:].mean(dim=(1, 2))
        return torch.cat((super().features(normalized_shape), geometry), dim=1)


__all__ = ["ScaleAwareComponentGeometricGlyphNet"]
