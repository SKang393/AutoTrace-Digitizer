# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Layout-conditioned proposal and role model for OCR V15."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.structural_graph_proposal_role_v14.model_p2 import StructuralGraphProposalRoleP2Net
from .protocol import (
    BASE_GEOMETRY_FEATURE_COUNT,
    CROP_HEIGHT,
    CROP_WIDTH,
    ENCODED_WIDTH,
    INPUT_CHANNELS,
    PLOT_GEOMETRY_FEATURE_COUNT,
    ROLE_ORDER,
    SEED,
)


class LayoutConditionedProposalRoleNet(nn.Module):
    """Train a fresh export-safe proposal model with a separate plot-layout role branch."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.base = StructuralGraphProposalRoleP2Net(seed=seed)
        self.layout_role = nn.Sequential(
            nn.Linear(PLOT_GEOMETRY_FEATURE_COUNT, 48),
            nn.ReLU(),
            nn.Linear(48, len(ROLE_ORDER)),
        )
        nn.init.kaiming_uniform_(self.layout_role[0].weight, generator=generator, nonlinearity="relu")
        nn.init.zeros_(self.layout_role[0].bias)
        nn.init.kaiming_uniform_(self.layout_role[2].weight, generator=generator, nonlinearity="linear")
        nn.init.zeros_(self.layout_role[2].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V15 proposal tensor must be [proposal_count,2,32,152]")
        base_width = CROP_WIDTH + BASE_GEOMETRY_FEATURE_COUNT
        base_output = self.base(value[:, :, :, :base_width])
        layout = value[:, 0, 0, base_width:]
        role = base_output[:, 2:] + self.layout_role(layout) * 0.04
        return torch.cat((base_output[:, :2], role), dim=1)


__all__ = ["LayoutConditionedProposalRoleNet"]
