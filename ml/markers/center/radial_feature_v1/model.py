# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed radial-topology projections with a small trainable MLP head."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ml.markers.center.line_aware_v1.model import LineAwareTensorContract


@dataclass(frozen=True)
class RadialFeatureModelConfig:
    architecture: str = "radial-topology-projection-mlp-v1"
    seed: int = 20261001
    projection_count_per_channel: int = 12
    hidden_width: int = 56


class RadialFeatureNet(nn.Module):
    """Non-convolutional patch model with fixed geometry projections."""

    contract = LineAwareTensorContract(runtime_revision="marker-center-radial-feature-runtime-v1")

    def __init__(self, config: RadialFeatureModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or RadialFeatureModelConfig()
        projection = self._projection_bank()
        self.register_buffer("projection", projection, persistent=True)
        feature_count = projection.shape[0]
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.config.seed)
            self.head = nn.Sequential(
                nn.Linear(feature_count * 2, self.config.hidden_width),
                nn.SiLU(),
                nn.Linear(self.config.hidden_width, 28),
                nn.SiLU(),
                nn.Linear(28, 4),
            )
        finally:
            torch.random.set_rng_state(state)

    @staticmethod
    def _projection_bank() -> Tensor:
        axis = torch.arange(33, dtype=torch.float32) - 16.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt((xx * xx) + (yy * yy))
        angle = torch.atan2(yy, xx)
        masks = (
            radius <= 2.5,
            (radius > 2.5) & (radius <= 5.0),
            (radius > 5.0) & (radius <= 8.0),
            (radius > 8.0) & (radius <= 12.0),
            (radius > 12.0) & (radius <= 16.5),
            torch.abs(xx) <= 1.5,
            torch.abs(yy) <= 1.5,
            torch.abs(xx - yy) <= 1.75,
            torch.abs(xx + yy) <= 1.75,
            (torch.cos(angle) > 0.70) & (radius <= 12.0),
            (torch.sin(angle) > 0.70) & (radius <= 12.0),
            ((torch.cos(angle * 4.0) > 0.65) & (radius >= 3.0) & (radius <= 10.0)),
        )
        rows: list[Tensor] = []
        for channel in range(3):
            for mask in masks:
                spatial = mask.to(torch.float32)
                spatial = spatial / torch.clamp(spatial.sum(), min=1.0)
                row = torch.zeros((3, 33, 33), dtype=torch.float32)
                row[channel] = spatial
                rows.append(row.flatten())
        return torch.stack(rows)

    def forward_raw(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1:] != (3, 33, 33):
            raise ValueError("RadialFeatureNet requires NCHW [N,3,33,33] patches")
        projected = torch.matmul(value.flatten(1), self.projection.transpose(0, 1))
        features = torch.cat((projected, torch.abs(projected - projected.mean(dim=1, keepdim=True))), dim=1)
        return self.head(features)

    @staticmethod
    def activate(raw: Tensor) -> Tensor:
        return torch.cat(
            (
                torch.sigmoid(raw[:, 0:1]),
                torch.tanh(raw[:, 1:3]) * 0.75,
                2.5 + (torch.sigmoid(raw[:, 3:4]) * 5.5),
            ),
            dim=1,
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.activate(self.forward_raw(value))

    def export_contract(self) -> dict[str, object]:
        return {
            "architecture": self.config.architecture,
            "model": asdict(self.config),
            "tensor_contract": asdict(self.contract),
            "fixed_projection_sha_semantics": "12 normalized radial/topology masks per input channel",
        }


__all__ = ["RadialFeatureModelConfig", "RadialFeatureNet"]
