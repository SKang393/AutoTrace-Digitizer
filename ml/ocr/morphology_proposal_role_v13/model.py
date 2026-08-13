# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Anisotropic morphology-aware proposal and role model for OCR V13."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import (
    CROP_HEIGHT, CROP_WIDTH, ENCODED_WIDTH, GEOMETRY_FEATURE_COUNT,
    INPUT_CHANNELS, ROLE_ORDER, SEED,
)


class MorphologyProposalRoleNet(nn.Module):
    """Return two proposal logits followed by eight role logits."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.tight = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 28, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(28, 40, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(), nn.Linear(40 * 4 * 8, 80), nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 28, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(28, 36, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(), nn.Linear(36 * 4 * 8, 72), nn.ReLU(),
        )
        self.horizontal_morphology = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 12, (1, 9), padding=(0, 4)), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(12, 20, (3, 7), padding=(1, 3)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((2, 8)), nn.Flatten(), nn.Linear(20 * 2 * 8, 48), nn.ReLU(),
        )
        self.vertical_morphology = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 12, (9, 1), padding=(4, 0)), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(12, 20, (7, 3), padding=(3, 1)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4)), nn.Flatten(), nn.Linear(20 * 4 * 4, 48), nn.ReLU(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 56), nn.ReLU(), nn.Linear(56, 40), nn.ReLU(),
        )
        self.mixture_gate = nn.Sequential(nn.Linear(40, 4), nn.Softmax(dim=1))
        self.shared = nn.Sequential(nn.Linear(288, 176), nn.ReLU(), nn.Dropout(0.08))
        self.proposal_head = nn.Sequential(nn.Linear(176, 72), nn.ReLU(), nn.Linear(72, 2))
        self.role_head = nn.Sequential(nn.Linear(176, 88), nn.ReLU(), nn.Linear(88, len(ROLE_ORDER)))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V13 proposal tensor must be [proposal_count,2,32,144]")
        pixels = value[:, :, :, :CROP_WIDTH]
        tight = self.tight(pixels[:, 0:1])
        context = self.context(pixels[:, 1:2])
        horizontal = self.horizontal_morphology(pixels)
        vertical = self.vertical_morphology(pixels)
        geometry = self.geometry(value[:, 0, 0, CROP_WIDTH:])
        weights = self.mixture_gate(geometry)
        visual = torch.cat((
            tight * weights[:, 0:1], context * weights[:, 1:2],
            horizontal * weights[:, 2:3], vertical * weights[:, 3:4],
        ), dim=1)
        shared = self.shared(torch.cat((visual, geometry), dim=1))
        return torch.cat((self.proposal_head(shared), self.role_head(shared)), dim=1) * 0.10


__all__ = ["MorphologyProposalRoleNet"]

