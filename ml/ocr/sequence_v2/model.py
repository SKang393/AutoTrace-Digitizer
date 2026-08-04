# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Spatially preserving sequence-logit model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .dataset import CLASS_COUNT


class SpatialAlignedSequenceModel(nn.Module):
    """Reduce height while retaining horizontal glyph alignment."""

    def __init__(self, contrast_standardization: bool = False) -> None:
        super().__init__()
        self.contrast_standardization = contrast_standardization
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(32, 48, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(48, CLASS_COUNT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.contrast_standardization:
            centered = inputs - inputs.mean(dim=(2, 3), keepdim=True)
            variance = (centered * centered).mean(dim=(2, 3), keepdim=True)
            inputs = centered / torch.sqrt(variance + 1e-6)
        features = self.features(inputs).mean(dim=2)
        features = functional.avg_pool1d(features, kernel_size=4, stride=4)
        return self.classifier(features.transpose(1, 2))
