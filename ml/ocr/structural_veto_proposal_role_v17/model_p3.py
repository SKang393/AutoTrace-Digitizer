# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen P2 model with a context-topology-only veto branch for OCR V17 P3."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from .model import StructuralVetoProposalRoleNet
from .protocol import CROP_HEIGHT, CROP_WIDTH, ENCODED_WIDTH, INPUT_CHANNELS, SEED


class ContextTopologyVetoProposalRoleNet(nn.Module):
    """Lower only positive proposal evidence using context topology around frozen P2."""

    def __init__(self, seed: int = SEED, *, maximum_context_veto: float = 2.0) -> None:
        super().__init__()
        if maximum_context_veto <= 0.0:
            raise ValueError("OCR V17 P3 maximum context veto must be positive")
        generator = torch.Generator().manual_seed(seed)
        self.base = StructuralVetoProposalRoleNet(seed=seed)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((2, 4)),
            nn.Flatten(),
        )
        self.context_veto_head = nn.Sequential(
            nn.Linear(96 + 32 + 16 + (ENCODED_WIDTH - CROP_WIDTH), 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        for group in (self.context_encoder, self.context_veto_head):
            for module in group.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module is not self.context_veto_head[2]:
                    nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.context_veto_head[2].weight)
        nn.init.constant_(self.context_veto_head[2].bias, -4.0)
        self.register_buffer("maximum_context_veto", torch.tensor(float(maximum_context_veto)))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for module in (self.context_encoder, self.context_veto_head)
            for parameter in module.parameters()
        ]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V17 P3 proposal tensor must be [proposal_count,2,32,152]")
        base = self.base(value)
        crops = value[:, :, :, :CROP_WIDTH]
        context = value[:, 1, :, :CROP_WIDTH]
        row_projection = context.mean(dim=2)
        column_projection = functional.adaptive_avg_pool1d(
            context.mean(dim=1, keepdim=True), 16,
        ).squeeze(1)
        geometry = value[:, 0, 0, CROP_WIDTH:]
        features = torch.cat((
            self.context_encoder(crops), row_projection, column_projection, geometry,
        ), dim=1)
        veto = torch.sigmoid(self.context_veto_head(features)) * self.maximum_context_veto
        proposal = torch.cat((base[:, :1], base[:, 1:2] - veto), dim=1)
        return torch.cat((proposal, base[:, 2:]), dim=1)


__all__ = ["ContextTopologyVetoProposalRoleNet"]
