# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Margin-robust layout proposal and role model for OCR V16."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.layout_conditioned_proposal_role_v15.model import LayoutConditionedProposalRoleNet
from .protocol import (
    BASE_GEOMETRY_FEATURE_COUNT,
    CROP_HEIGHT,
    CROP_WIDTH,
    ENCODED_WIDTH,
    INPUT_CHANNELS,
    PLOT_GEOMETRY_FEATURE_COUNT,
    SEED,
)


class MarginRobustLayoutProposalRoleNet(nn.Module):
    """Add a proposal-specific plot-layout residual to a fresh V15-shaped network."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed + 1)
        self.base = LayoutConditionedProposalRoleNet(seed=seed)
        self.layout_proposal = nn.Sequential(
            nn.Linear(PLOT_GEOMETRY_FEATURE_COUNT, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        nn.init.kaiming_uniform_(self.layout_proposal[0].weight, generator=generator, nonlinearity="relu")
        nn.init.zeros_(self.layout_proposal[0].bias)
        nn.init.kaiming_uniform_(self.layout_proposal[2].weight, generator=generator, nonlinearity="linear")
        nn.init.zeros_(self.layout_proposal[2].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V16 proposal tensor must be [proposal_count,2,32,152]")
        base_output = self.base(value)
        layout_start = CROP_WIDTH + BASE_GEOMETRY_FEATURE_COUNT
        layout = value[:, 0, 0, layout_start:]
        proposal = base_output[:, :2] + self.layout_proposal(layout) * 0.04
        return torch.cat((proposal, base_output[:, 2:]), dim=1)


__all__ = ["MarginRobustLayoutProposalRoleNet"]
