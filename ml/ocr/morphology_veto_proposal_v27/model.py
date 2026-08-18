# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V26 parent with a morphology veto head for OCR V27."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.scene_topology_proposal_v26.model_p3 import (
    FrozenP2FinalTailProposalNet,
)

from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    FEATURE_COUNT,
    OUTPUT_LOGIT_SCALE,
    SEED,
    STRUCTURE_FEATURE_COUNT,
)


class FrozenV26MorphologyVetoNet(nn.Module):
    """Add a bounded structural residual while preserving V26 role output."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.parent = FrozenP2FinalTailProposalNet()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.veto_head = nn.Sequential(
            nn.Linear(STRUCTURE_FEATURE_COUNT + 2, 48),
            nn.GELU(),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 2),
        )
        for module in self.veto_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.veto_head[-1].weight)
        nn.init.zeros_(self.veto_head[-1].bias)

    def load_parent_state_dict(
        self, state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.parent.load_state_dict(state_dict, strict=True)
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def train(self, mode: bool = True) -> "FrozenV26MorphologyVetoNet":
        super().train(mode)
        self.parent.eval()
        return self

    def forward(
        self,
        proposal_evidence: torch.Tensor,
        proposal_crops: torch.Tensor,
        structure: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            proposal_evidence.ndim != 3
            or proposal_evidence.shape[0] != 1
            or proposal_evidence.shape[2] != FEATURE_COUNT
            or proposal_crops.ndim != 5
            or proposal_crops.shape[:2] != proposal_evidence.shape[:2]
            or proposal_crops.shape[2:] != (
                CROP_CHANNELS,
                CROP_HEIGHT,
                CROP_WIDTH,
            )
            or structure.ndim != 3
            or structure.shape[:2] != proposal_evidence.shape[:2]
            or structure.shape[2] != STRUCTURE_FEATURE_COUNT
        ):
            raise ValueError(
                "Expected evidence [1,N,31], crops [1,N,2,32,128], and "
                "structure [1,N,24]"
            )
        with torch.no_grad():
            parent = self.parent(proposal_evidence, proposal_crops)
        residual = self.veto_head(torch.cat((structure, parent[:, :, :2]), dim=2))
        return torch.cat((parent[:, :, :2] + residual, parent[:, :, 2:]), dim=2) * (
            OUTPUT_LOGIT_SCALE
        )


__all__ = ["FrozenV26MorphologyVetoNet"]
