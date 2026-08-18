# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen-P2 monotonic veto-only repair for OCR V27 P3."""

from __future__ import annotations

import torch
from torch import nn

from .model_p2 import FrozenP1FinalVetoProposalNet
from .protocol import SEED, STRUCTURE_FEATURE_COUNT


class FrozenP2MonotonicVetoProposalNet(FrozenP1FinalVetoProposalNet):
    """Freeze P2 and learn only a nonnegative acceptance suppression."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__(seed=seed)
        generator = torch.Generator().manual_seed(seed + 3)
        self.monotonic_veto = nn.Sequential(
            nn.Linear(STRUCTURE_FEATURE_COUNT + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.kaiming_uniform_(
            self.monotonic_veto[0].weight,
            generator=generator,
            nonlinearity="relu",
        )
        nn.init.zeros_(self.monotonic_veto[0].bias)
        nn.init.zeros_(self.monotonic_veto[2].weight)
        nn.init.constant_(self.monotonic_veto[2].bias, -2.0)

    def load_p2_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            "monotonic_veto.0.bias",
            "monotonic_veto.0.weight",
            "monotonic_veto.2.bias",
            "monotonic_veto.2.weight",
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError("OCR V27 P3 checkpoint boundary changed")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.monotonic_veto.parameters():
            parameter.requires_grad_(True)
        self.parent.eval()
        self.veto_head.eval()
        self.monotonic_veto.eval()

    def train(self, mode: bool = True) -> "FrozenP2MonotonicVetoProposalNet":
        super().train(False)
        self.monotonic_veto.train(mode)
        return self

    def forward(
        self,
        proposal_evidence: torch.Tensor,
        proposal_crops: torch.Tensor,
        structure: torch.Tensor,
    ) -> torch.Tensor:
        p2_output = self.frozen_p2_output(proposal_evidence, proposal_crops, structure)
        suppression = nn.functional.softplus(
            self.monotonic_veto(torch.cat((structure, p2_output[:, :, :2]), dim=2))
        )
        proposal = torch.cat((
            p2_output[:, :, 0:1] + suppression,
            p2_output[:, :, 1:2] - suppression,
        ), dim=2)
        return torch.cat((proposal, p2_output[:, :, 2:]), dim=2)

    def frozen_p2_output(
        self,
        proposal_evidence: torch.Tensor,
        proposal_crops: torch.Tensor,
        structure: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact frozen P2 output before monotonic suppression."""

        with torch.no_grad():
            return super().forward(proposal_evidence, proposal_crops, structure)


__all__ = ["FrozenP2MonotonicVetoProposalNet"]
