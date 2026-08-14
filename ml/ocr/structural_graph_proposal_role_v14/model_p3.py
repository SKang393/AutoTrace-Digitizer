# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Head-only role and proposal repair for structural graph OCR V14 P3."""

from __future__ import annotations

import torch
from torch import nn

from .model_p2 import StructuralGraphProposalRoleP2Net
from .protocol import CROP_WIDTH, GEOMETRY_FEATURE_COUNT, ROLE_ORDER, SEED


class StructuralGraphProposalRoleP3Net(StructuralGraphProposalRoleP2Net):
    """Add a zero-initialized raw-geometry role residual to exact P1 state."""

    def __init__(self, *, base_seed: int = SEED, residual_seed: int = 20262044) -> None:
        super().__init__(seed=base_seed)
        generator = torch.Generator().manual_seed(residual_seed)
        self.role_geometry_residual = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 32), nn.ReLU(), nn.Linear(32, len(ROLE_ORDER)),
        )
        nn.init.kaiming_uniform_(
            self.role_geometry_residual[0].weight, generator=generator, nonlinearity="relu",
        )
        nn.init.zeros_(self.role_geometry_residual[0].bias)
        nn.init.zeros_(self.role_geometry_residual[2].weight)
        nn.init.zeros_(self.role_geometry_residual[2].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = super().forward(value)
        raw_geometry = value[:, 0, 0, CROP_WIDTH:]
        role_residual = self.role_geometry_residual(raw_geometry) * 0.05
        return torch.cat((output[:, :2], output[:, 2:] + role_residual), dim=1)


class OutputScaledCandidate(nn.Module):
    """Apply the fixed P3 export scale after head-only training."""

    def __init__(self, candidate: StructuralGraphProposalRoleP3Net, scale: float) -> None:
        super().__init__()
        self.candidate = candidate
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.candidate(value) * self.scale


__all__ = ["OutputScaledCandidate", "StructuralGraphProposalRoleP3Net"]
