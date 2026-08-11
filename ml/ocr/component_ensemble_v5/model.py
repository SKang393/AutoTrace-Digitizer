# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed multi-resolution feature ensemble and MLP glyph classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import CLASS_COUNT, ENCODED_GLYPH_WIDTH, GEOMETRY_FEATURE_COUNT, GLYPH_HEIGHT, GLYPH_WIDTH


class ComponentEnsembleGlyphNet(nn.Module):
    """Combine fixed shape, profile, edge, radial, and source-geometry evidence."""

    SHAPE_FEATURE_COUNT = 60 + 40 + GLYPH_HEIGHT + GLYPH_WIDTH + (GLYPH_HEIGHT - 1) + (GLYPH_WIDTH - 1) + 12
    FEATURE_COUNT = SHAPE_FEATURE_COUNT + GEOMETRY_FEATURE_COUNT

    def __init__(self, seed: int = 20260820) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.classifier = nn.Sequential(
            nn.Linear(self.FEATURE_COUNT, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, CLASS_COUNT),
        )
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, GLYPH_HEIGHT),
            torch.linspace(-1.0, 1.0, GLYPH_WIDTH),
            indexing="ij",
        )
        radius = torch.sqrt(xx * xx + yy * yy)
        masks = torch.stack(
            [
                (xx > 0).float(),
                (xx < 0).float(),
                (yy > 0).float(),
                (yy < 0).float(),
                (xx + yy > 0).float(),
                (xx + yy < 0).float(),
                (xx - yy > 0).float(),
                (xx - yy < 0).float(),
                (radius <= 0.35).float(),
                ((radius > 0.35) & (radius <= 0.70)).float(),
                ((radius > 0.70) & (radius <= 1.05)).float(),
                (radius > 1.05).float(),
            ]
        )
        masks = masks / masks.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        self.register_buffer("projection_masks", masks, persistent=True)

    def features(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (1, GLYPH_HEIGHT, ENCODED_GLYPH_WIDTH):
            raise ValueError("OCR V5 glyph tensor must be [glyph_count,1,24,26]")
        shape = value[:, :, :, :GLYPH_WIDTH]
        geometry = value[:, :, :, GLYPH_WIDTH:].mean(dim=(1, 2))
        pooled_mean = F.avg_pool2d(shape, kernel_size=(4, 4), stride=(4, 4)).flatten(1)
        pooled_max = F.max_pool2d(shape, kernel_size=(4, 4), stride=(4, 4)).flatten(1)
        multiscale = F.avg_pool2d(shape, kernel_size=(3, 4), stride=(3, 4)).flatten(1)
        row_profile = shape.mean(dim=3).squeeze(1)
        column_profile = shape.mean(dim=2).squeeze(1)
        row_edges = torch.abs(shape[:, :, 1:, :] - shape[:, :, :-1, :]).mean(dim=3).squeeze(1)
        column_edges = torch.abs(shape[:, :, :, 1:] - shape[:, :, :, :-1]).mean(dim=2).squeeze(1)
        radial = torch.einsum("nchw,khw->nk", shape, self.projection_masks)
        return torch.cat(
            (
                pooled_mean,
                pooled_max,
                multiscale,
                row_profile,
                column_profile,
                row_edges,
                column_edges,
                radial,
                geometry,
            ),
            dim=1,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(value))


__all__ = ["ComponentEnsembleGlyphNet"]
