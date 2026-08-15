# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Quadratic proposal calibrator for the V20 large-margin defect class."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import FEATURE_COUNT


class MarginSeparatedProposalCalibrator(nn.Module):
    """Small MLP over frozen evidence and its deterministic quadratic lift."""

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(FEATURE_COUNT * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
        finally:
            torch.random.set_rng_state(state)

    def forward(self, proposal_evidence: torch.Tensor) -> torch.Tensor:
        if proposal_evidence.ndim != 2 or proposal_evidence.shape[1] != FEATURE_COUNT:
            raise ValueError(f"Expected [proposal_count,{FEATURE_COUNT}] proposal evidence")
        lifted = torch.cat((proposal_evidence, proposal_evidence.square()), dim=1)
        return self.network(lifted)


__all__ = ["MarginSeparatedProposalCalibrator"]

