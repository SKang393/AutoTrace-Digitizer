# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen OCR V24 composition with a fixed recognition-evidence rescue."""

from __future__ import annotations

import torch
from torch import nn

from ml.ocr.crop_evidence_role_anchor_v24.model_p2 import (
    FrozenRoleAnchorCropResidualNet,
)

from .protocol import (
    ACCEPTED_LOGIT_MAGNITUDE,
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    CTC_ALNUM_FRACTION_MINIMUM,
    CTC_BLANK_RATIO_MAXIMUM,
    CTC_ENTROPY_MEAN_MAXIMUM,
    CTC_LENGTH_FRACTION_MINIMUM,
    CTC_MARGIN_MEAN_MINIMUM,
    CTC_SELECTED_MEAN_MINIMUM,
    CTC_TOP1_MEAN_MINIMUM,
    FEATURE_COUNT,
    PARENT_ACCEPTANCE_MINIMUM,
    ROLE_ORDER,
)


class FrozenCropResidualCtcRescueNet(nn.Module):
    """Retain V24 P2 output and rescue only strong recognized-text evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.parent = FrozenRoleAnchorCropResidualNet()
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)

    def load_parent_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.parent.load_state_dict(state_dict, strict=True)
        self.parent.eval()

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
        parent_output = self.parent(proposal_evidence, proposal_crops)
        anchor_output = self.parent.backbone(proposal_evidence)
        parent_probability = torch.softmax(parent_output[:, :, :2], dim=2)[:, :, 1]
        anchor_probability = torch.softmax(anchor_output[:, :, :2], dim=2)[:, :, 1]
        alphanumeric = proposal_evidence[:, :, 16] + proposal_evidence[:, :, 17]
        recognition_rescue = (
            (anchor_probability >= PARENT_ACCEPTANCE_MINIMUM)
            & (proposal_evidence[:, :, 10] >= CTC_SELECTED_MEAN_MINIMUM)
            & (proposal_evidence[:, :, 11] >= CTC_TOP1_MEAN_MINIMUM)
            & (proposal_evidence[:, :, 12] >= CTC_MARGIN_MEAN_MINIMUM)
            & (proposal_evidence[:, :, 13] <= CTC_ENTROPY_MEAN_MAXIMUM)
            & (proposal_evidence[:, :, 14] <= CTC_BLANK_RATIO_MAXIMUM)
            & (proposal_evidence[:, :, 15] >= CTC_LENGTH_FRACTION_MINIMUM)
            & (alphanumeric >= CTC_ALNUM_FRACTION_MINIMUM)
        )
        accepted = (parent_probability >= PARENT_ACCEPTANCE_MINIMUM) | recognition_rescue
        magnitude = torch.full_like(parent_probability, ACCEPTED_LOGIT_MAGNITUDE)
        positive = torch.where(accepted, magnitude, -magnitude)
        proposal_output = torch.stack((-positive, positive), dim=2)
        return torch.cat((proposal_output, parent_output[:, :, 2:]), dim=2)


__all__ = ["FrozenCropResidualCtcRescueNet"]
