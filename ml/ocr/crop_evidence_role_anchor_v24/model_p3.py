# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Zero-optimizer parent-recall and crop-residual consensus gate."""

from __future__ import annotations

from math import log

import torch
from torch import nn

from .model_p2 import FrozenRoleAnchorCropResidualNet


class ParentRecallResidualVetoNet(nn.Module):
    """Keep parent recall while requiring a non-vetoed P2 crop residual."""

    def __init__(
        self,
        candidate: FrozenRoleAnchorCropResidualNet,
        *,
        parent_probability_minimum: float = 0.35,
        crop_residual_margin_minimum: float = -0.25,
        accepted_logit_magnitude: float = 8.0,
    ) -> None:
        super().__init__()
        if not 0.0 < parent_probability_minimum < 1.0:
            raise ValueError("parent_probability_minimum must be between zero and one")
        if accepted_logit_magnitude <= 0.0:
            raise ValueError("accepted_logit_magnitude must be positive")
        self.candidate = candidate
        self.register_buffer(
            "parent_margin_minimum",
            torch.tensor(log(parent_probability_minimum / (1.0 - parent_probability_minimum))),
        )
        self.register_buffer(
            "crop_residual_margin_minimum",
            torch.tensor(crop_residual_margin_minimum),
        )
        self.register_buffer(
            "accepted_logit_magnitude",
            torch.tensor(accepted_logit_magnitude),
        )

    def forward(
        self, proposal_evidence: torch.Tensor, proposal_crops: torch.Tensor,
    ) -> torch.Tensor:
        candidate_output = self.candidate(proposal_evidence, proposal_crops)
        parent_output = self.candidate.backbone(proposal_evidence)
        parent_margin = parent_output[:, :, 1] - parent_output[:, :, 0]
        candidate_margin = candidate_output[:, :, 1] - candidate_output[:, :, 0]
        crop_residual_margin = candidate_margin - parent_margin
        accepted = torch.logical_and(
            parent_margin >= self.parent_margin_minimum,
            crop_residual_margin >= self.crop_residual_margin_minimum,
        )
        magnitude = self.accepted_logit_magnitude.to(candidate_output.dtype)
        signed = torch.where(accepted, magnitude, -magnitude)
        proposal_output = torch.stack((-signed, signed), dim=2)
        return torch.cat((proposal_output, candidate_output[:, :, 2:]), dim=2)


__all__ = ["ParentRecallResidualVetoNet"]
