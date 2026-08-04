# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Whole-crop global semantic-slot graph-numeric recognizer."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import (
    BLANK_CLASS_INDEX,
    CLASS_COUNT,
    MAX_TOKENS,
    ROLE_COUNT,
    SLOT_TIME_INDICES,
    TIME_STEPS,
)


class GlobalSemanticSlotRecognizer(nn.Module):
    """Classify semantic positions from one global 2D feature bottleneck."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 16, 160),
            nn.GELU(),
            nn.LayerNorm(160),
        )
        self.slot_classifier = nn.Linear(160, MAX_TOKENS * CLASS_COUNT)
        self.role_classifier = nn.Linear(160, ROLE_COUNT)

        projector = torch.zeros(TIME_STEPS, MAX_TOKENS, dtype=torch.float32)
        for slot, time_index in enumerate(SLOT_TIME_INDICES):
            projector[time_index, slot] = 1.0
        base_logits = torch.full((TIME_STEPS, CLASS_COUNT), -12.0)
        base_logits[:, BLANK_CLASS_INDEX] = 12.0
        for time_index in SLOT_TIME_INDICES:
            base_logits[time_index] = 0.0
        self.register_buffer("slot_projector", projector, persistent=True)
        self.register_buffer("base_time_logits", base_logits, persistent=True)

    def semantic_logits(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = inputs - inputs.mean(dim=(2, 3), keepdim=True)
        variance = (centered * centered).mean(dim=(2, 3), keepdim=True)
        standardized = centered / torch.sqrt(variance + 1e-6)
        encoded = self.bottleneck(self.encoder(standardized))
        slots = self.slot_classifier(encoded).reshape(-1, MAX_TOKENS, CLASS_COUNT)
        roles = self.role_classifier(encoded)
        return slots, roles

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot_logits, role_logits = self.semantic_logits(inputs)
        time_logits = self.base_time_logits.unsqueeze(0) + torch.einsum(
            "ts,nsc->ntc", self.slot_projector, slot_logits
        )
        return time_logits, role_logits
