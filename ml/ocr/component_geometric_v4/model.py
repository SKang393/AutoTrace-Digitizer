# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Non-convolutional component-geometric glyph classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import CLASS_COUNT, GLYPH_HEIGHT, GLYPH_WIDTH


class ComponentGeometricGlyphNet(nn.Module):
    """Classify canonical glyphs from fixed grid, profile, and radial projections."""

    FEATURE_COUNT = 60 + GLYPH_HEIGHT + GLYPH_WIDTH + 12

    def __init__(self, seed: int = 20260810) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.classifier = nn.Sequential(
            nn.Linear(self.FEATURE_COUNT, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, CLASS_COUNT),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, GLYPH_HEIGHT),
            torch.linspace(-1.0, 1.0, GLYPH_WIDTH),
            indexing="ij",
        )
        radius = torch.sqrt(xx * xx + yy * yy)
        angle_masks = [
            (xx > 0).float(),
            (xx < 0).float(),
            (yy > 0).float(),
            (yy < 0).float(),
            (xx + yy > 0).float(),
            (xx + yy < 0).float(),
            (xx - yy > 0).float(),
            (xx - yy < 0).float(),
        ]
        radial_masks = [
            (radius <= 0.35).float(),
            ((radius > 0.35) & (radius <= 0.70)).float(),
            ((radius > 0.70) & (radius <= 1.05)).float(),
            (radius > 1.05).float(),
        ]
        projections = torch.stack([*angle_masks, *radial_masks])
        projections = projections / projections.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        self.register_buffer("projection_masks", projections, persistent=True)

    def features(self, value: torch.Tensor) -> torch.Tensor:
        pooled_mean = F.avg_pool2d(value, kernel_size=(4, 4), stride=(4, 4)).flatten(1)
        pooled_max = F.max_pool2d(value, kernel_size=(4, 4), stride=(4, 4)).flatten(1)
        row_profile = value.mean(dim=3).squeeze(1)
        column_profile = value.mean(dim=2).squeeze(1)
        radial = torch.einsum("nchw,khw->nk", value, self.projection_masks)
        return torch.cat((pooled_mean, pooled_max, row_profile, column_profile, radial), dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(value))
