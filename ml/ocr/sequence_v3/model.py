# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Compact canonical-slot convolutional recognizer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .dataset import BLANK_CLASS_INDEX, CLASS_COUNT, SLOT_COUNT, TIME_STEPS


class CanonicalSlotRecognizer(nn.Module):
    """Classify normalized glyph slots and emit runtime-compatible logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(32, CLASS_COUNT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = functional.adaptive_avg_pool2d(features, (1, SLOT_COUNT))
        slot_logits = self.classifier(pooled.squeeze(2).transpose(1, 2))
        separators = torch.full_like(slot_logits, -8.0)
        separators[:, :, BLANK_CLASS_INDEX] = 8.0
        return torch.stack((slot_logits, separators), dim=2).reshape(
            inputs.shape[0], TIME_STEPS, CLASS_COUNT
        )
