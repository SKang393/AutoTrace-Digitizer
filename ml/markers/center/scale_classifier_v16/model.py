# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Small multi-scale CNN for marker versus connected-component proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "scale-separated-multiscale-patch-cnn-v16"
    seed: int = 20260902


class _Tower(nn.Sequential):
    def __init__(self, channels: int, widths: tuple[int, int, int]) -> None:
        first, second, third = widths
        super().__init__(
            nn.Conv2d(channels, first, 5, padding=2, bias=False),
            nn.BatchNorm2d(first),
            nn.SiLU(),
            nn.Conv2d(first, second, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.SiLU(),
            nn.Conv2d(second, third, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.SiLU(),
            nn.AvgPool2d(2),
        )


class ScaleClassifierNet(nn.Module):
    """Separate ink and mask towers retain both 5x5 local and 3x3 context."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.config.seed)
            self.ink_tower = _Tower(1, (16, 24, 32))
            self.mask_tower = _Tower(2, (8, 12, 16))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear((32 + 16) * 4 * 4, 64),
                nn.SiLU(),
                nn.Linear(64, 4),
            )
        finally:
            torch.random.set_rng_state(state)

    def forward_raw(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1:] != (3, 33, 33):
            raise ValueError("ScaleClassifierNet requires NCHW [candidate_count,3,33,33] patches")
        return self.head(torch.cat((self.ink_tower(value[:, 0:1]), self.mask_tower(value[:, 1:3])), dim=1))

    def forward(self, value: Tensor) -> Tensor:
        raw = self.forward_raw(value)
        return torch.cat((torch.sigmoid(raw[:, 0:1]), torch.tanh(raw[:, 1:3]) * 0.75, 2.5 + torch.sigmoid(raw[:, 3:4]) * 5.5), dim=1)

    def export_contract(self) -> dict[str, object]:
        return {"architecture": self.config.architecture, "model": asdict(self.config), "input_shape": ["candidate_count", 3, 33, 33], "output_shape": ["candidate_count", 4], "output_columns": ["marker_probability", "offset_x_grid", "offset_y_grid", "radius_pixels"], "coordinate_space": "model_tensor", "radius_contract": {"minimum_pixels": 2.5, "maximum_pixels": 8.0}}
