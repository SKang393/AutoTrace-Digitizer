# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V28 P1 proposals with a deterministic geometry role partition."""

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
    SEED,
)


class FrozenP1GeometryRolePartitionNet(nn.Module):
    """Preserve P1 proposal logits and resolve roles from plot-relative centers."""

    RELATIVE_CENTER_X_INDEX = 25
    RELATIVE_CENTER_Y_INDEX = 26
    AXIS_TITLE_Y_BOUNDARY = 1.15
    ROLE_LOGIT_MAGNITUDE = 8.0

    def __init__(self, *, p1_seed: int = SEED) -> None:
        super().__init__()
        self.p1 = RelationalNeighborhoodProposalNet(seed=p1_seed)
        for parameter in self.p1.parameters():
            parameter.requires_grad_(False)

    def load_p1_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.p1.load_state_dict(state_dict, strict=True)
        for parameter in self.p1.parameters():
            parameter.requires_grad_(False)
        self.p1.eval()

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def train(self, mode: bool = True) -> "FrozenP1GeometryRolePartitionNet":
        super().train(mode)
        self.p1.eval()
        return self

    @classmethod
    def _role_logits(cls, proposal_evidence: torch.Tensor) -> torch.Tensor:
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
            tuple(mask.to(dtype=proposal_evidence.dtype) for mask in masks),
            dim=2,
        )
        return memberships * (2.0 * cls.ROLE_LOGIT_MAGNITUDE) - (
            cls.ROLE_LOGIT_MAGNITUDE
        )

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
        return torch.cat((p1_output[:, :, :2], self._role_logits(proposal_evidence)), dim=2)


__all__ = ["FrozenP1GeometryRolePartitionNet"]
