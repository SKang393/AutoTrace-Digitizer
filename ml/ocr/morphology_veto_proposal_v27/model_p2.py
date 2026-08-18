# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen-P1 final-linear hard-negative repair for OCR V27 P2."""

from __future__ import annotations

import torch

from .model import FrozenV26MorphologyVetoNet
from .protocol import SEED


class FrozenP1FinalVetoProposalNet(FrozenV26MorphologyVetoNet):
    """Train only the final veto linear layer from the exact P1 state."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__(seed=seed)

    def load_p1_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=True)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.veto_head[4].parameters():
            parameter.requires_grad_(True)
        self.parent.eval()
        self.veto_head.eval()

    def train(self, mode: bool = True) -> "FrozenP1FinalVetoProposalNet":
        super().train(False)
        self.veto_head[4].train(mode)
        return self


__all__ = ["FrozenP1FinalVetoProposalNet"]
