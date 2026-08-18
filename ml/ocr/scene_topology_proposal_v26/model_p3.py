# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen-P2 final-tail proposal repair for OCR V26 P3."""

from __future__ import annotations

import torch

from .model_p2 import FrozenP1BoundedMarginProposalNet
from .protocol import SEED


class FrozenP2FinalTailProposalNet(FrozenP1BoundedMarginProposalNet):
    """Train only the final proposal linear layer from the exact P2 state."""

    def __init__(
        self,
        seed: int = SEED,
        dropout_probability: float = 0.08,
    ) -> None:
        super().__init__(seed=seed, dropout_probability=dropout_probability)

    def load_p2_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=True)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.proposal_head[5].parameters():
            parameter.requires_grad_(True)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenP2FinalTailProposalNet":
        # Frozen dropout and feature layers remain in inference mode. The final
        # Linear layer has no mode-dependent behavior but is left trainable.
        super().train(False)
        self.proposal_head[5].train(mode)
        return self


__all__ = ["FrozenP2FinalTailProposalNet"]
