# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Dynamic proposal-set relational OCR model V21."""

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


class RelationalMessageBlock(nn.Module):
    """Propagate learned context across every proposal in one graph scene."""

    def __init__(self, width: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, width),
        )
        self.norm = nn.LayerNorm(width)
        self.scale = width ** -0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        query = self.query(values)
        key = self.key(values)
        scores = torch.matmul(query, key.transpose(1, 2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        message = torch.matmul(weights, self.value(values))
        return self.norm(values + self.update(torch.cat((values, message), dim=-1)))


class RelationalSceneProposalRoleNet(nn.Module):
    """Return proposal and role logits for a complete dynamic proposal set."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.tight = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(24 * 4 * 8, 64),
            nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(1, 12, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 20, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(20 * 4 * 8, 56),
            nn.ReLU(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 48),
            nn.ReLU(),
            nn.Linear(48, 40),
            nn.ReLU(),
        )
        self.proposal_embedding = nn.Sequential(nn.Linear(160, 128), nn.ReLU())
        self.relational = nn.Sequential(
            RelationalMessageBlock(128, seed + 1),
            RelationalMessageBlock(128, seed + 2),
        )
        self.proposal_head = nn.Sequential(nn.Linear(128, 48), nn.ReLU(), nn.Linear(48, 2))
        self.role_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ROLE_ORDER)))
        for name, module in self.named_modules():
            if name.startswith("relational."):
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            value.ndim != 5 or value.shape[2:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH)
        ):
            raise ValueError("OCR V21 input must be [batch,proposal_count,2,32,152]")
        batch, proposal_count = value.shape[:2]
        flattened = value.reshape(batch * proposal_count, INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH)
        pixels = flattened[:, :, :, :CROP_WIDTH]
        geometry_values = flattened[:, 0, 0, CROP_WIDTH:]
        encoded = self.proposal_embedding(torch.cat((
            self.tight(pixels[:, 0:1]),
            self.context(pixels[:, 1:2]),
            self.geometry(geometry_values),
        ), dim=1)).reshape(batch, proposal_count, 128)
        relational = self.relational(encoded)
        return torch.cat((self.proposal_head(relational), self.role_head(relational)), dim=-1) * 0.125


__all__ = ["RelationalSceneProposalRoleNet"]
