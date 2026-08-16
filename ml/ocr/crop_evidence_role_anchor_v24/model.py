# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Dynamic crop-evidence role-anchor model for OCR V24."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import (
    CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH, FEATURE_COUNT, ROLE_ORDER, SEED,
)


class CropEvidenceRoleAnchorNet(nn.Module):
    """Fuse proposal crops and evidence before role-conditioned scene pooling."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        width = 96
        self.evidence_encoder = nn.Sequential(
            nn.Linear(FEATURE_COUNT * 2, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(CROP_CHANNELS, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 2 * 8, 64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.role_queries = nn.Parameter(torch.empty(len(ROLE_ORDER), width))
        self.anchor_key = nn.Linear(width, width, bias=False)
        self.anchor_value = nn.Linear(width, width, bias=False)
        self.proposal_query = nn.Linear(width, width, bias=False)
        self.anchor_key_for_proposal = nn.Linear(width, width, bias=False)
        self.anchor_value_for_proposal = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.Linear(width * 4, 128), nn.GELU(),
            nn.Linear(128, width), nn.GELU(),
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
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, proposal_evidence: torch.Tensor, proposal_crops: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            proposal_evidence.ndim != 3
            or proposal_evidence.shape[0] != 1
            or proposal_evidence.shape[2] != FEATURE_COUNT
            or proposal_crops.ndim != 5
            or proposal_crops.shape[:2] != proposal_evidence.shape[:2]
            or proposal_crops.shape[2:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH)
        ):
            raise ValueError(
                "Expected evidence [1,N,31] and crops [1,N,2,32,128]"
            )
        count = proposal_evidence.shape[1]
        lifted = torch.cat((proposal_evidence, proposal_evidence.square()), dim=2)
        evidence = self.evidence_encoder(lifted)
        crops = self.crop_encoder(proposal_crops.reshape(
            -1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
        )).reshape(1, count, 64)
        encoded = self.fusion(torch.cat((evidence, crops), dim=2))
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


__all__ = ["CropEvidenceRoleAnchorNet"]
