# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Pairwise-geometry relational proposal model for OCR V28."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.crop_evidence_role_anchor_v24.model_p2 import (
    FrozenRoleAnchorCropResidualNet,
)

from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    FEATURE_COUNT,
    OUTPUT_LOGIT_SCALE,
    RELATION_FEATURE_COUNT,
    SEED,
)


class PairwiseGeometryMessageBlock(nn.Module):
    """Propagate proposal context with an explicit ordered-pair geometry bias."""

    def __init__(self, width: int, relation_width: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.relation_bias = nn.Sequential(
            nn.Linear(relation_width, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.relation_value = nn.Sequential(
            nn.Linear(relation_width, 48), nn.GELU(), nn.Linear(48, width),
        )
        self.update = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.norm = nn.LayerNorm(width)
        self.scale = width ** -0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, nodes: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        query = self.query(nodes)
        key = self.key(nodes)
        scores = torch.matmul(query, key.transpose(1, 2)) * self.scale
        scores = scores + self.relation_bias(relations).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        node_values = self.value(nodes).unsqueeze(1)
        pair_values = node_values + self.relation_value(relations)
        message = torch.sum(weights.unsqueeze(-1) * pair_values, dim=2)
        return self.norm(nodes + self.update(torch.cat((nodes, message), dim=-1)))


class RelationalNeighborhoodProposalNet(nn.Module):
    """Learn proposal decisions from complete scene neighborhoods and frozen roles."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.role_parent = FrozenRoleAnchorCropResidualNet()
        for parameter in self.role_parent.parameters():
            parameter.requires_grad_(False)
        self.crop_stem = nn.Sequential(
            nn.Conv2d(CROP_CHANNELS, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Flatten(),
            nn.Linear(24 * 4 * 8, 72),
            nn.GELU(),
        )
        self.evidence_stem = nn.Sequential(
            nn.Linear(FEATURE_COUNT * 2, 80),
            nn.GELU(),
            nn.Linear(80, 56),
            nn.GELU(),
        )
        self.node_projection = nn.Sequential(
            nn.Linear(72 + 56, 128),
            nn.GELU(),
            nn.Linear(128, 96),
            nn.GELU(),
        )
        self.message_one = PairwiseGeometryMessageBlock(
            96, RELATION_FEATURE_COUNT, seed + 1,
        )
        self.message_two = PairwiseGeometryMessageBlock(
            96, RELATION_FEATURE_COUNT, seed + 2,
        )
        self.proposal_head = nn.Sequential(
            nn.Linear(96 + FEATURE_COUNT + 2, 96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 2),
        )
        for name, module in self.named_modules():
            if name.startswith("role_parent.") or name.startswith("message_"):
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def load_role_parent_state_dict(
        self, state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.role_parent.load_state_dict(state_dict, strict=True)
        for parameter in self.role_parent.parameters():
            parameter.requires_grad_(False)
        self.role_parent.eval()

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def train(self, mode: bool = True) -> "RelationalNeighborhoodProposalNet":
        super().train(mode)
        self.role_parent.eval()
        return self

    def forward(
        self,
        proposal_evidence: torch.Tensor,
        proposal_crops: torch.Tensor,
        proposal_relations: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            proposal_evidence.ndim != 3
            or proposal_evidence.shape[0] != 1
            or proposal_evidence.shape[2] != FEATURE_COUNT
            or proposal_crops.ndim != 5
            or proposal_crops.shape[:2] != proposal_evidence.shape[:2]
            or proposal_crops.shape[2:] != (
                CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
            )
            or proposal_relations.ndim != 4
            or proposal_relations.shape[:3] != (
                proposal_evidence.shape[0],
                proposal_evidence.shape[1],
                proposal_evidence.shape[1],
            )
            or proposal_relations.shape[3] != RELATION_FEATURE_COUNT
        ):
            raise ValueError(
                "Expected evidence [1,N,31], crops [1,N,2,32,128], and "
                "relations [1,N,N,19]"
            )
        count = proposal_evidence.shape[1]
        with torch.no_grad():
            parent = self.role_parent(proposal_evidence, proposal_crops)
        crop_values = self.crop_stem(proposal_crops.reshape(
            -1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
        )).reshape(1, count, 72)
        evidence_values = self.evidence_stem(torch.cat((
            proposal_evidence, proposal_evidence.square(),
        ), dim=2))
        nodes = self.node_projection(torch.cat((crop_values, evidence_values), dim=2))
        nodes = self.message_one(nodes, proposal_relations)
        nodes = self.message_two(nodes, proposal_relations)
        proposal_logits = self.proposal_head(torch.cat((
            nodes, proposal_evidence, parent[:, :, :2],
        ), dim=2))
        return torch.cat((proposal_logits, parent[:, :, 2:]), dim=2) * (
            OUTPUT_LOGIT_SCALE
        )


__all__ = ["PairwiseGeometryMessageBlock", "RelationalNeighborhoodProposalNet"]
