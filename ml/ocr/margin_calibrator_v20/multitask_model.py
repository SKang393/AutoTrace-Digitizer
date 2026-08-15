# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Complete-stream proposal and role calibrator for the final V20 candidate."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import FEATURE_COUNT, ROLE_ORDER


class CompleteStreamMultitaskCalibrator(nn.Module):
    """Calibrate proposal acceptance and text role from the same frozen evidence."""

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(
                nn.Linear(FEATURE_COUNT * 2, 64),
                nn.ReLU(),
            )
            self.proposal_head = nn.Linear(64, 2)
            self.role_head = nn.Linear(64, len(ROLE_ORDER))
        finally:
            torch.random.set_rng_state(state)

    def forward(self, proposal_evidence: torch.Tensor) -> torch.Tensor:
        if proposal_evidence.ndim != 2 or proposal_evidence.shape[1] != FEATURE_COUNT:
            raise ValueError(f"Expected [proposal_count,{FEATURE_COUNT}] proposal evidence")
        lifted = torch.cat((proposal_evidence, proposal_evidence.square()), dim=1)
        encoded = self.encoder(lifted)
        return torch.cat((self.proposal_head(encoded), self.role_head(encoded)), dim=1)


__all__ = ["CompleteStreamMultitaskCalibrator"]
