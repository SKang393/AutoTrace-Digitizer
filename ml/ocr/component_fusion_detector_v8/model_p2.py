# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe fixed-pooling repair for OCR component-fusion V8 P2."""

from __future__ import annotations

import torch
from torch import nn

from .protocol import CROP_HEIGHT, CROP_WIDTH, ENCODED_WIDTH, GEOMETRY_FEATURE_COUNT, INPUT_CHANNELS


class ComponentFusionP2Net(nn.Module):
    """Preserve the P1 branch design with an ONNX-exportable fixed pool."""

    def __init__(self, seed: int = 20261208) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.visual = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(48 * 4 * 8, 128),
            nn.ReLU(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_COUNT, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(152, 80),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(80, 2),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, generator=generator, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH):
            raise ValueError("OCR V8 P2 proposal tensor must be [proposal_count,2,32,140]")
        visual = value[:, :, :, :CROP_WIDTH]
        geometry = value[:, 0, 0, CROP_WIDTH:]
        return self.classifier(torch.cat((self.visual(visual), self.geometry(geometry)), dim=1))


__all__ = ["ComponentFusionP2Net"]
