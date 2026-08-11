# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe dual-context classifier for deterministic OCR V7 proposals."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import CROP_HEIGHT, ENCODED_WIDTH, INPUT_CHANNELS


class ComponentContextNet(nn.Module):
    """Classify one tight plus scene-context proposal tensor."""

    def __init__(self, seed: int = 20261112) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 17, 96),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(96, 2),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V7 proposal tensor must be [proposal_count,2,32,140]")
        return self.classifier(self.features(value))


__all__ = ["ComponentContextNet"]

