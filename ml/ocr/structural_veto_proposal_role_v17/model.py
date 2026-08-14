# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V16 base with a proposal-only structural-veto branch for OCR V17."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.margin_robust_layout_proposal_role_v16.model import MarginRobustLayoutProposalRoleNet
from .protocol import CROP_HEIGHT, CROP_WIDTH, ENCODED_WIDTH, INPUT_CHANNELS, SEED


class StructuralVetoProposalRoleNet(nn.Module):
    """Lower only positive proposal evidence while preserving frozen V16 role ordering."""

    def __init__(self, seed: int = SEED, *, base_output_scale: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < base_output_scale <= 1.0:
            raise ValueError("OCR V17 base output scale must be in (0,1]")
        generator = torch.Generator().manual_seed(seed)
        self.base = MarginRobustLayoutProposalRoleNet(seed=seed)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.structural_veto_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
        )
        self.structural_veto_head = nn.Sequential(
            nn.Linear(96 + (ENCODED_WIDTH - CROP_WIDTH), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        for group in (self.structural_veto_encoder, self.structural_veto_head):
            for module in group.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module is not self.structural_veto_head[2]:
                    nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.structural_veto_head[2].weight)
        nn.init.constant_(self.structural_veto_head[2].bias, -4.0)
        self.register_buffer("base_output_scale", torch.tensor(float(base_output_scale)))
        self.register_buffer("maximum_veto", torch.tensor(5.0))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for module in (self.structural_veto_encoder, self.structural_veto_head)
            for parameter in module.parameters()
        ]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V17 proposal tensor must be [proposal_count,2,32,152]")
        base = self.base(value) * self.base_output_scale
        crop = value[:, :1, :, :CROP_WIDTH]
        geometry = value[:, 0, 0, CROP_WIDTH:]
        features = torch.cat((self.structural_veto_encoder(crop), geometry), dim=1)
        veto = torch.sigmoid(self.structural_veto_head(features)) * self.maximum_veto
        proposal = torch.cat((base[:, :1], base[:, 1:2] - veto), dim=1)
        return torch.cat((proposal, base[:, 2:]), dim=1)


__all__ = ["StructuralVetoProposalRoleNet"]
