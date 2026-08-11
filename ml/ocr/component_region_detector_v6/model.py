# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Small export-safe binary classifier for deterministic OCR V6 proposals."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import CROP_HEIGHT, ENCODED_WIDTH


class ComponentRegionNet(nn.Module):
    """Classify one checksum-bound proposal tensor as non-text or text."""

    def __init__(self, seed: int = 20261111) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 4 * 17, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (1, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V6 proposal tensor must be [proposal_count,1,32,136]")
        return self.classifier(self.features(value))


__all__ = ["ComponentRegionNet"]

