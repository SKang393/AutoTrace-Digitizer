# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Dynamic role-anchor proposal-set model for OCR V23."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import FEATURE_COUNT, ROLE_ORDER, SEED


class RoleAnchorSetNet(nn.Module):
    """Pool role-conditioned scene anchors, then classify every proposal."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        width = 96
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_COUNT * 2, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.role_queries = nn.Parameter(torch.empty(len(ROLE_ORDER), width))
        self.anchor_key = nn.Linear(width, width, bias=False)
        self.anchor_value = nn.Linear(width, width, bias=False)
        self.proposal_query = nn.Linear(width, width, bias=False)
        self.anchor_key_for_proposal = nn.Linear(width, width, bias=False)
        self.anchor_value_for_proposal = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.Linear(width * 4, 128),
            nn.GELU(),
            nn.Linear(128, width),
            nn.GELU(),
        )
        self.proposal_head = nn.Sequential(
            nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 2),
        )
        self.role_head = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(ROLE_ORDER)),
        )
        self.scale = width ** -0.5
        nn.init.normal_(self.role_queries, mean=0.0, std=0.05, generator=generator)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, proposal_evidence: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            proposal_evidence.ndim != 3
            or proposal_evidence.shape[0] != 1
            or proposal_evidence.shape[2] != FEATURE_COUNT
        ):
            raise ValueError(f"Expected [1,proposal_count,{FEATURE_COUNT}] proposal evidence")
        lifted = torch.cat((proposal_evidence, proposal_evidence.square()), dim=2)
        encoded = self.encoder(lifted)
        role_queries = self.role_queries.unsqueeze(0)
        anchor_scores = torch.matmul(
            role_queries, self.anchor_key(encoded).transpose(1, 2),
        ) * self.scale
        anchors = torch.matmul(
            torch.softmax(anchor_scores, dim=-1), self.anchor_value(encoded),
        )
        proposal_scores = torch.matmul(
            self.proposal_query(encoded),
            self.anchor_key_for_proposal(anchors).transpose(1, 2),
        ) * self.scale
        role_context = torch.matmul(
            torch.softmax(proposal_scores, dim=-1),
            self.anchor_value_for_proposal(anchors),
        )
        scene_mean = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        scene_maximum = encoded.amax(dim=1, keepdim=True).expand_as(encoded)
        updated = self.update(torch.cat(
            (encoded, role_context, scene_mean, scene_maximum), dim=2,
        ))
        return torch.cat(
            (self.proposal_head(updated), self.role_head(updated)), dim=2,
        ) * 0.125


__all__ = ["RoleAnchorSetNet"]
