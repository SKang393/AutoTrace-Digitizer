# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Independent attention and relation-summary proposal consensus for OCR V29."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.relational_neighborhood_proposal_v28.model import (
    RelationalNeighborhoodProposalNet,
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


class RelationSummaryProposalRoute(nn.Module):
    """Classify proposals from local evidence and invariant neighbor summaries."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
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
            nn.Linear(FEATURE_COUNT * 2, 72),
            nn.GELU(),
            nn.Linear(72, 48),
            nn.GELU(),
        )
        self.relation_stem = nn.Sequential(
            nn.Linear(RELATION_FEATURE_COUNT * 3, 80),
            nn.GELU(),
            nn.Linear(80, 56),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(72 + 48 + 56, 112),
            nn.GELU(),
            nn.Linear(112, 48),
            nn.GELU(),
            nn.Linear(48, 2),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        proposal_evidence: torch.Tensor,
        proposal_crops: torch.Tensor,
        proposal_relations: torch.Tensor,
    ) -> torch.Tensor:
        count = proposal_evidence.shape[1]
        crop_values = self.crop_stem(proposal_crops.reshape(
            -1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
        )).reshape(1, count, 72)
        evidence_values = self.evidence_stem(torch.cat((
            proposal_evidence, proposal_evidence.square(),
        ), dim=2))
        relation_values = torch.cat((
            proposal_relations.mean(dim=2),
            torch.amax(proposal_relations, dim=2),
            torch.amin(proposal_relations, dim=2),
        ), dim=2)
        relation_values = self.relation_stem(relation_values)
        return self.head(torch.cat((
            crop_values, evidence_values, relation_values,
        ), dim=2)) * OUTPUT_LOGIT_SCALE


class DualRouteConsensusProposalNet(nn.Module):
    """Average independent proposal routes and emit deterministic graph roles."""

    RELATIVE_CENTER_X_INDEX = 25
    RELATIVE_CENTER_Y_INDEX = 26
    AXIS_TITLE_Y_BOUNDARY = 1.15
    ROLE_LOGIT_MAGNITUDE = 8.0

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self.attention_route = RelationalNeighborhoodProposalNet(seed=seed)
        self.summary_route = RelationSummaryProposalRoute(seed=seed + 101)

    def load_role_parent_state_dict(
        self, state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.attention_route.load_role_parent_state_dict(state_dict)

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def train(self, mode: bool = True) -> "DualRouteConsensusProposalNet":
        super().train(mode)
        self.attention_route.role_parent.eval()
        return self

    @classmethod
    def role_logits(cls, proposal_evidence: torch.Tensor) -> torch.Tensor:
        x = proposal_evidence[:, :, cls.RELATIVE_CENTER_X_INDEX]
        y = proposal_evidence[:, :, cls.RELATIVE_CENTER_Y_INDEX]
        left = x < 0.0
        right = x > 1.0
        middle = ~(left | right)
        above = y < 0.0
        below = y > 1.0
        masks = (
            left & ~above,
            middle & below & (y < cls.AXIS_TITLE_Y_BOUNDARY),
            middle & (y >= cls.AXIS_TITLE_Y_BOUNDARY),
            middle & above,
            right & ~below,
            right & below,
            middle & ~above & ~below,
            left & above,
        )
        memberships = torch.stack(
            tuple(mask.to(dtype=proposal_evidence.dtype) for mask in masks), dim=2,
        )
        return memberships * (2.0 * cls.ROLE_LOGIT_MAGNITUDE) - (
            cls.ROLE_LOGIT_MAGNITUDE
        )

    def proposal_routes(
        self,
        proposal_evidence: torch.Tensor,
        proposal_crops: torch.Tensor,
        proposal_relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = self.attention_route(
            proposal_evidence, proposal_crops, proposal_relations,
        )[:, :, :2]
        summary = self.summary_route(
            proposal_evidence, proposal_crops, proposal_relations,
        )
        return (attention + summary) / 2.0, attention, summary

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
        consensus, _, _ = self.proposal_routes(
            proposal_evidence, proposal_crops, proposal_relations,
        )
        return torch.cat((consensus, self.role_logits(proposal_evidence)), dim=2)


__all__ = ["DualRouteConsensusProposalNet", "RelationSummaryProposalRoute"]
