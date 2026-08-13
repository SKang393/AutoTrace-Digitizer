# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh multi-head proposal and role model for OCR V11."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import (
    CROP_HEIGHT,
    CROP_WIDTH,
    ENCODED_WIDTH,
    GEOMETRY_FEATURE_COUNT,
    INPUT_CHANNELS,
    ROLE_ORDER,
    SEED,
)


class CompositeProposalRoleNet(nn.Module):
    """Return two proposal logits followed by eight role logits."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.tight = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
            nn.Linear(48 * 4 * 8, 96), nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 40, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
            nn.Linear(40 * 4 * 8, 80), nn.ReLU(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU(),
        )
        self.shared = nn.Sequential(nn.Linear(208, 128), nn.ReLU(), nn.Dropout(0.12))
        self.proposal_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        self.role_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ROLE_ORDER)))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V11 proposal tensor must be [proposal_count,2,32,144]")
        tight = value[:, 0:1, :, :CROP_WIDTH]
        context = value[:, 1:2, :, :CROP_WIDTH]
        geometry = value[:, 0, 0, CROP_WIDTH:]
        shared = self.shared(torch.cat((self.tight(tight), self.context(context), self.geometry(geometry)), dim=1))
        return torch.cat((self.proposal_head(shared), self.role_head(shared)), dim=1)


__all__ = ["CompositeProposalRoleNet"]
