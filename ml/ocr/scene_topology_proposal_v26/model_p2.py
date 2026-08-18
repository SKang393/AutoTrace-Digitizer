# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen-P1 bounded-margin proposal repair for OCR V26 P2."""

from __future__ import annotations

import torch

from .model import FrozenRoleAxialTopologyProposalNet
from .protocol import SEED


class FrozenP1BoundedMarginProposalNet(FrozenRoleAxialTopologyProposalNet):
    """Retrain only the P1 proposal head while all feature and role weights stay fixed."""

    def __init__(
        self,
        seed: int = SEED,
        dropout_probability: float = 0.08,
    ) -> None:
        super().__init__(seed=seed, dropout_probability=dropout_probability)

    def load_p1_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=True)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.proposal_head.parameters():
            parameter.requires_grad_(True)
        self.role_parent.eval()
        self.crop_stem.eval()
        self.crop_projection.eval()
        self.evidence_projection.eval()

    def train(self, mode: bool = True) -> "FrozenP1BoundedMarginProposalNet":
        super().train(mode)
        self.role_parent.eval()
        self.crop_stem.eval()
        self.crop_projection.eval()
        self.evidence_projection.eval()
        self.proposal_head.train(mode)
        return self


__all__ = ["FrozenP1BoundedMarginProposalNet"]
