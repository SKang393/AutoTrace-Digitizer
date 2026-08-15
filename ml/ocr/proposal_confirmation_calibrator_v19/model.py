# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Small project-owned proposal confirmation calibrator."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import FEATURE_COUNT


class ProposalConfirmationCalibrator(nn.Module):
    def __init__(self, *, seed: int) -> None:
        super().__init__()
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(FEATURE_COUNT, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
        finally:
            torch.random.set_rng_state(state)

    def forward(self, proposal_evidence: torch.Tensor) -> torch.Tensor:
        if proposal_evidence.ndim != 2 or proposal_evidence.shape[1] != FEATURE_COUNT:
            raise ValueError(f"Expected [proposal_count,{FEATURE_COUNT}] proposal evidence")
        return self.network(proposal_evidence)


__all__ = ["ProposalConfirmationCalibrator"]
