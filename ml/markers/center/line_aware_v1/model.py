# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Dual-branch patch model for marker ink and exclusion-mask context."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


@dataclass(frozen=True)
class LineAwareTensorContract:
    runtime_revision: str = "marker-center-line-aware-runtime-v1"
    input_name: str = "candidate_patches"
    input_layout: str = "NCHW"
    input_dtype: str = "float32"
    input_shape: tuple[str | int, ...] = ("candidate_count", 3, 33, 33)
    input_channels: tuple[str, ...] = ("ink_probability", "text_mask", "artifact_mask")
    output_name: str = "candidate_predictions"
    output_layout: str = "NC"
    output_shape: tuple[str | int, ...] = ("candidate_count", 4)
    output_columns: tuple[str, ...] = (
        "marker_probability", "offset_x_grid", "offset_y_grid", "radius_pixels"
    )
    coordinate_space: str = "model_tensor"
    patch_size: int = 33
    proposal_stride: int = 4


@dataclass(frozen=True)
class LineAwareModelConfig:
    architecture: str = "line-aware-dual-branch-patch-cnn-v1"
    seed: int = 20260831


class _Branch(nn.Sequential):
    def __init__(self, input_channels: int, widths: tuple[int, int, int]) -> None:
        a, b, c = widths
        super().__init__(
            nn.Conv2d(input_channels, a, 5, padding=2, bias=False),
            nn.BatchNorm2d(a),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(a, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(b, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )


class LineAwarePatchNet(nn.Module):
    contract = LineAwareTensorContract()

    def __init__(self, config: LineAwareModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or LineAwareModelConfig()
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.config.seed)
            self.ink_branch = _Branch(1, (16, 24, 32))
            self.mask_branch = _Branch(2, (8, 12, 16))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear((32 + 16) * 3 * 3, 64),
                nn.SiLU(),
                nn.Dropout(0.04),
                nn.Linear(64, 4),
            )
        finally:
            torch.random.set_rng_state(state)

    def forward_raw(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1:] != (3, 33, 33):
            raise ValueError("LineAwarePatchNet requires NCHW [N,3,33,33] patches")
        features = torch.cat((self.ink_branch(value[:, 0:1]), self.mask_branch(value[:, 1:3])), dim=1)
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
        }


def candidate_loss(
    raw: Tensor,
    labels: Tensor,
    offsets: Tensor,
    radii: Tensor,
    *,
    positive_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    marker = functional.binary_cross_entropy_with_logits(
        raw[:, 0], labels, pos_weight=torch.tensor(positive_weight, device=raw.device)
    )
    positive = labels > 0.5
    if torch.any(positive):
        offset = functional.smooth_l1_loss(torch.tanh(raw[positive, 1:3]) * 0.75, offsets[positive])
        radius = functional.smooth_l1_loss(2.5 + (torch.sigmoid(raw[positive, 3]) * 5.5), radii[positive])
    else:
        offset = raw[:, 1:3].sum() * 0
        radius = raw[:, 3].sum() * 0
    total = marker + (1.5 * offset) + (0.35 * radius)
    return total, {
        "total": float(total.detach()),
        "marker": float(marker.detach()),
        "offset": float(offset.detach()),
        "radius": float(radius.detach()),
    }


__all__ = ["LineAwareModelConfig", "LineAwarePatchNet", "LineAwareTensorContract", "candidate_loss"]
