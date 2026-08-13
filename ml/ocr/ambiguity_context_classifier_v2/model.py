# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Profile-aware context classifier with parity-bounded logits."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import GLYPHS, IMAGE_SIZE


class LineContextAmbiguityNet(nn.Module):
    def __init__(self, seed: int = 20261421) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        profile_count = IMAGE_SIZE * 2 + 6
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4 + profile_count, 96), nn.ReLU(), nn.Linear(96, len(GLYPHS)),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Line-context ambiguity input must be [batch,1,32,32]")
        convolution = self.convolution(value).flatten(1)
        row = value.mean(dim=3).squeeze(1)
        column = value.mean(dim=2).squeeze(1)
        row_peak = value.amax(dim=3).squeeze(1)
        column_peak = value.amax(dim=2).squeeze(1)
        active_rows = (row_peak > 0.08).float()
        active_columns = (column_peak > 0.08).float()
        geometry = torch.cat((
            active_rows.mean(dim=1, keepdim=True), active_columns.mean(dim=1, keepdim=True),
            row[:, : IMAGE_SIZE // 2].mean(dim=1, keepdim=True),
            row[:, IMAGE_SIZE // 2 :].mean(dim=1, keepdim=True),
            column[:, : IMAGE_SIZE // 2].mean(dim=1, keepdim=True),
            column[:, IMAGE_SIZE // 2 :].mean(dim=1, keepdim=True),
        ), dim=1)
        features = torch.cat((convolution, row, column, geometry), dim=1)
        return self.classifier(features) * 0.125


__all__ = ["LineContextAmbiguityNet"]
