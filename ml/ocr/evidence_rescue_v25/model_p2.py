# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V24 parent with a trainable pooled-crop proposal residual."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.crop_evidence_role_anchor_v24.model_p2 import (
    FrozenRoleAnchorCropResidualNet,
)

from .protocol import CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH, FEATURE_COUNT, SEED


class FrozenParentPooledEvidenceResidualNet(nn.Module):
    """Preserve parent roles while learning proposal rescue and veto evidence."""

    def __init__(self, seed: int = SEED + 1, residual_scale: float = 1.0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.parent = FrozenRoleAnchorCropResidualNet()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.mean_pool = nn.AvgPool2d(kernel_size=(8, 8), stride=(8, 8))
        self.max_pool = nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8))
        pooled_width = CROP_CHANNELS * 4 * 16 * 2
        self.residual_head = nn.Sequential(
            nn.Linear(FEATURE_COUNT + pooled_width + 2, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.residual_scale = residual_scale
        for module in self.residual_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("OCR V25 P2 residual head final layer changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def load_parent_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.parent.load_state_dict(state_dict, strict=True)
        self.parent.eval()

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
        parent_output = self.parent(proposal_evidence, proposal_crops)
        flat_crops = proposal_crops.reshape(
            -1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
        )
        pooled = torch.cat(
            (self.mean_pool(flat_crops), self.max_pool(flat_crops)), dim=1,
        ).reshape(1, count, -1)
        residual = self.residual_head(torch.cat(
            (proposal_evidence, pooled, parent_output[:, :, :2]), dim=2,
        )) * self.residual_scale
        proposal_output = parent_output[:, :, :2] + torch.cat(
            (-residual, residual), dim=2,
        )
        return torch.cat((proposal_output, parent_output[:, :, 2:]), dim=2)


__all__ = ["FrozenParentPooledEvidenceResidualNet"]
