# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Two-of-three robust route quorum for OCR V31."""

from __future__ import annotations

import torch

from ml.ocr.unanimous_structure_veto_v30.model import (
    UnanimousStructureVetoProposalNet,
)

from .protocol import SEED


class RobustQuorumRecallProposalNet(UnanimousStructureVetoProposalNet):
    """Accept the median route margin, tolerating one underconfident route."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__(seed=seed)

    @staticmethod
    def unanimous_logits(
        attention: torch.Tensor,
        summary: torch.Tensor,
        local_veto: torch.Tensor,
    ) -> torch.Tensor:
        attention_margin = attention[:, :, 1] - attention[:, :, 0]
        summary_margin = summary[:, :, 1] - summary[:, :, 0]
        local_margin = local_veto[:, :, 1] - local_veto[:, :, 0]
        lowest = torch.minimum(
            torch.minimum(attention_margin, summary_margin), local_margin,
        )
        highest = torch.maximum(
            torch.maximum(attention_margin, summary_margin), local_margin,
        )
        median = attention_margin + summary_margin + local_margin - lowest - highest
        return torch.stack((-median / 2.0, median / 2.0), dim=2)


__all__ = ["RobustQuorumRecallProposalNet"]
