# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen-role axial topology proposal model for OCR V26."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.crop_evidence_role_anchor_v24.model_p2 import (
    FrozenRoleAnchorCropResidualNet,
)

from .protocol import CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH, FEATURE_COUNT, SEED


class FrozenRoleAxialTopologyProposalNet(nn.Module):
    """Learn proposals from scratch while preserving the exact parent roles."""

    def __init__(self, seed: int = SEED, dropout_probability: float = 0.08) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.role_parent = FrozenRoleAnchorCropResidualNet()
        for parameter in self.role_parent.parameters():
            parameter.requires_grad_(False)
        self.crop_stem = nn.Sequential(
            nn.Conv2d(CROP_CHANNELS, 12, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(
                12, 12, kernel_size=3, stride=2, padding=1, groups=12,
            ),
            nn.GELU(),
            nn.Conv2d(12, 24, kernel_size=1),
            nn.GELU(),
        )
        self.crop_projection = nn.Sequential(
            nn.Linear(24 * (2 + 8 + 16), 128),
            nn.GELU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(128, 80),
            nn.GELU(),
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(FEATURE_COUNT * 2, 80),
            nn.GELU(),
            nn.Linear(80, 64),
            nn.GELU(),
        )
        self.proposal_head = nn.Sequential(
            nn.Linear(80 + 64 + FEATURE_COUNT * 2 + 2, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        for module in (
            *self.crop_stem.modules(),
            *self.crop_projection.modules(),
            *self.evidence_projection.modules(),
            *self.proposal_head.modules(),
        ):
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
        self.role_parent.eval()

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

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
            raise ValueError("Expected evidence [1,N,31] and crops [1,N,2,32,128]")
        count = proposal_evidence.shape[1]
        with torch.no_grad():
            parent = self.role_parent(proposal_evidence, proposal_crops)
        encoded = self.crop_stem(proposal_crops.reshape(
            -1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
        ))
        global_mean = encoded.mean(dim=(2, 3))
        global_maximum = encoded.amax(dim=(2, 3))
        row_projection = encoded.mean(dim=3, keepdim=True).flatten(1)
        column_projection = nn.functional.avg_pool2d(
            encoded.mean(dim=2, keepdim=True), kernel_size=(1, 2), stride=(1, 2),
        ).flatten(1)
        crop_features = self.crop_projection(torch.cat(
            (global_mean, global_maximum, row_projection, column_projection), dim=1,
        )).reshape(1, count, 80)
        evidence_features = self.evidence_projection(torch.cat(
            (proposal_evidence, proposal_evidence.square()), dim=2,
        ))
        scene_mean = proposal_evidence.mean(dim=1, keepdim=True).expand_as(
            proposal_evidence,
        )
        scene_maximum = proposal_evidence.amax(dim=1, keepdim=True).expand_as(
            proposal_evidence,
        )
        proposal_logits = self.proposal_head(torch.cat(
            (
                crop_features,
                evidence_features,
                scene_mean,
                scene_maximum,
                parent[:, :, :2],
            ),
            dim=2,
        ))
        return torch.cat((proposal_logits, parent[:, :, 2:]), dim=2)


__all__ = ["FrozenRoleAxialTopologyProposalNet"]
