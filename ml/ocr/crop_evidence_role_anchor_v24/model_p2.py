# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V23 role anchor with a crop-conditioned proposal residual."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.role_anchor_set_v23.model import RoleAnchorSetNet

from .protocol import (
    CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH, FEATURE_COUNT, ROLE_ORDER, SEED,
)


class FrozenRoleAnchorCropResidualNet(nn.Module):
    """Preserve V23 roles while learning only a crop-conditioned proposal veto."""

    def __init__(self, seed: int = SEED, residual_scale: float = 0.0625) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.backbone = RoleAnchorSetNet()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(CROP_CHANNELS, 12, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(12, 20, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(20, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(24 * 2 * 8, 48),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(48 + 2, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.residual_scale = residual_scale
        for module in tuple(self.crop_encoder.modules()) + tuple(self.residual_head.modules()):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(
                    module.weight, generator=generator, nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("OCR V24 P2 residual head final layer changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def load_backbone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.eval()

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
            raise ValueError(
                "Expected evidence [1,N,31] and crops [1,N,2,32,128]"
            )
        count = proposal_evidence.shape[1]
        backbone_output = self.backbone(proposal_evidence)
        crop_features = self.crop_encoder(proposal_crops.reshape(
            -1, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH,
        )).reshape(1, count, 48)
        residual = self.residual_head(torch.cat(
            (crop_features, backbone_output[:, :, :2]), dim=2,
        )) * self.residual_scale
        proposal_output = backbone_output[:, :, :2] + torch.cat(
            (-residual, residual), dim=2,
        )
        return torch.cat((proposal_output, backbone_output[:, :, 2:]), dim=2)


__all__ = ["FrozenRoleAnchorCropResidualNet"]
