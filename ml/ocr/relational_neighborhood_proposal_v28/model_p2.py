# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V28 P1 proposal model with a relational role-only residual."""

from __future__ import annotations

import torch
from torch import nn

from .model import RelationalNeighborhoodProposalNet
from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    FEATURE_COUNT,
    RELATION_FEATURE_COUNT,
    ROLE_ORDER,
    SEED,
)


class FrozenP1RelationalRoleResidualNet(nn.Module):
    """Preserve exact P1 proposals while refining only its role logits."""

    NODE_WIDTH = 96

    def __init__(
        self,
        *,
        p1_seed: int = SEED,
        residual_seed: int = SEED + 1,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.p1 = RelationalNeighborhoodProposalNet(seed=p1_seed)
        for parameter in self.p1.parameters():
            parameter.requires_grad_(False)

        generator = torch.Generator().manual_seed(residual_seed)
        residual_input_width = self.NODE_WIDTH + FEATURE_COUNT + len(ROLE_ORDER)
        self.role_residual = nn.Sequential(
            nn.Linear(residual_input_width, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, len(ROLE_ORDER)),
        )
        self.residual_scale = residual_scale
        for module in self.role_residual.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.role_residual[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("OCR V28 P2 role residual final layer changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def load_p1_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.p1.load_state_dict(state_dict, strict=True)
        for parameter in self.p1.parameters():
            parameter.requires_grad_(False)
        self.p1.eval()

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def train(self, mode: bool = True) -> "FrozenP1RelationalRoleResidualNet":
        super().train(mode)
        self.p1.eval()
        return self

    def _p1_nodes(
        self,
        proposal_evidence: torch.Tensor,
        proposal_crops: torch.Tensor,
        proposal_relations: torch.Tensor,
    ) -> torch.Tensor:
        count = proposal_evidence.shape[1]
        crop_values = self.p1.crop_stem(proposal_crops.reshape(
            -1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
        )).reshape(1, count, 72)
        evidence_values = self.p1.evidence_stem(torch.cat((
            proposal_evidence, proposal_evidence.square(),
        ), dim=2))
        nodes = self.p1.node_projection(torch.cat(
            (crop_values, evidence_values), dim=2,
        ))
        nodes = self.p1.message_one(nodes, proposal_relations)
        return self.p1.message_two(nodes, proposal_relations)

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

        p1_output = self.p1(
            proposal_evidence, proposal_crops, proposal_relations,
        )
        nodes = self._p1_nodes(
            proposal_evidence, proposal_crops, proposal_relations,
        )
        residual = self.role_residual(torch.cat((
            nodes, proposal_evidence, p1_output[:, :, 2:],
        ), dim=2)) * self.residual_scale
        refined_roles = p1_output[:, :, 2:] + residual
        return torch.cat((p1_output[:, :, :2], refined_roles), dim=2)


__all__ = ["FrozenP1RelationalRoleResidualNet"]
