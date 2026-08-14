# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Topology-spectrum proposal and role model for OCR V14."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import (
    CROP_HEIGHT, CROP_WIDTH, ENCODED_WIDTH, GEOMETRY_FEATURE_COUNT,
    INPUT_CHANNELS, ROLE_ORDER, SEED,
)


class StructuralGraphProposalRoleNet(nn.Module):
    """Return two proposal logits followed by eight role logits."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.tight = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(), nn.Linear(32 * 4 * 8, 64), nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(1, 14, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(14, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 28, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(), nn.Linear(28 * 4 * 8, 56), nn.ReLU(),
        )
        self.topology = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 28, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 8)), nn.Flatten(), nn.Linear(28 * 4 * 8, 64), nn.ReLU(),
        )
        self.occupancy_spectrum = nn.Sequential(
            nn.Linear(52, 64), nn.ReLU(), nn.Linear(64, 48), nn.ReLU(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 48), nn.ReLU(), nn.Linear(48, 40), nn.ReLU(),
        )
        self.shared = nn.Sequential(nn.Linear(272, 160), nn.ReLU(), nn.Dropout(0.06))
        self.proposal_residual = nn.Linear(40, 2, bias=False)
        self.proposal_head = nn.Sequential(nn.Linear(160, 72), nn.ReLU(), nn.Linear(72, 2))
        self.role_head = nn.Sequential(nn.Linear(160, 80), nn.ReLU(), nn.Linear(80, len(ROLE_ORDER)))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _edge_planes(pixels: torch.Tensor) -> torch.Tensor:
        tight = pixels[:, 0:1]
        horizontal = F.pad(torch.abs(tight[:, :, :, 1:] - tight[:, :, :, :-1]), (0, 1, 0, 0))
        vertical = F.pad(torch.abs(tight[:, :, 1:, :] - tight[:, :, :-1, :]), (0, 0, 0, 1))
        return torch.cat((pixels, horizontal, vertical), dim=1)

    @staticmethod
    def _occupancy(pixels: torch.Tensor) -> torch.Tensor:
        tight = pixels[:, 0:1]
        row_mean = F.adaptive_avg_pool2d(tight, (8, 1)).flatten(1)
        column_mean = F.adaptive_avg_pool2d(tight, (1, 18)).flatten(1)
        row_peak = F.adaptive_max_pool2d(tight, (8, 1)).flatten(1)
        column_peak = F.adaptive_max_pool2d(tight, (1, 18)).flatten(1)
        return torch.cat((row_mean, column_mean, row_peak, column_peak), dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V14 proposal tensor must be [proposal_count,2,32,144]")
        pixels = value[:, :, :, :CROP_WIDTH]
        geometry = self.geometry(value[:, 0, 0, CROP_WIDTH:])
        shared = self.shared(torch.cat((
            self.tight(pixels[:, 0:1]),
            self.context(pixels[:, 1:2]),
            self.topology(self._edge_planes(pixels)),
            self.occupancy_spectrum(self._occupancy(pixels)),
            geometry,
        ), dim=1))
        proposal = self.proposal_head(shared) + self.proposal_residual(geometry)
        return torch.cat((proposal, self.role_head(shared)), dim=1) * 0.05


__all__ = ["StructuralGraphProposalRoleNet"]
