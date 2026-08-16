# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Dynamic complete-proposal-set attention model for OCR V22."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import FEATURE_COUNT, ROLE_ORDER, SEED


class EvidenceAttentionBlock(nn.Module):
    """Propagate context without depending on proposal ordering."""

    def __init__(self, width: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.norm = nn.LayerNorm(width)
        self.scale = width ** -0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(
            self.query(values), self.key(values).transpose(1, 2),
        ) * self.scale
        message = torch.matmul(torch.softmax(scores, dim=-1), self.value(values))
        return self.norm(values + self.update(torch.cat((values, message), dim=-1)))


class SceneEvidenceAttentionNet(nn.Module):
    """Return proposal and role logits for every proposal in one scene."""

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_COUNT * 2, 96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.attention = nn.Sequential(
            EvidenceAttentionBlock(96, seed + 1),
            EvidenceAttentionBlock(96, seed + 2),
        )
        self.proposal_head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 2))
        self.role_head = nn.Sequential(
            nn.Linear(96, 64), nn.GELU(), nn.Linear(64, len(ROLE_ORDER)),
        )
        for name, module in self.named_modules():
            if name.startswith("attention."):
                continue
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, proposal_evidence: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            proposal_evidence.ndim != 3
            or proposal_evidence.shape[0] != 1
            or proposal_evidence.shape[2] != FEATURE_COUNT
        ):
            raise ValueError(f"Expected [1,proposal_count,{FEATURE_COUNT}] proposal evidence")
        lifted = torch.cat((proposal_evidence, proposal_evidence.square()), dim=2)
        encoded = self.attention(self.encoder(lifted))
        return torch.cat((self.proposal_head(encoded), self.role_head(encoded)), dim=2) * 0.125


__all__ = ["EvidenceAttentionBlock", "SceneEvidenceAttentionNet"]
