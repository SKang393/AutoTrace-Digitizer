# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Compact patch classifier with center-offset and radius regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


@dataclass(frozen=True)
class CandidateTensorContract:
    runtime_revision: str = "marker-center-candidate-runtime-v1"
    input_name: str = "candidate_patches"
    input_layout: str = "NCHW"
    input_dtype: str = "float32"
    input_shape: tuple[str | int, ...] = ("candidate_count", 3, 33, 33)
    input_channels: tuple[str, ...] = (
        "ink_probability",
        "text_mask",
        "artifact_mask",
    )
    output_name: str = "candidate_predictions"
    output_layout: str = "NC"
    output_shape: tuple[str | int, ...] = ("candidate_count", 4)
    output_columns: tuple[str, ...] = (
        "marker_probability",
        "offset_x_grid",
        "offset_y_grid",
        "radius_pixels",
    )
    coordinate_space: str = "model_tensor"
    patch_size: int = 33
    proposal_stride: int = 4


@dataclass(frozen=True)
class CandidateModelConfig:
    architecture: str = "candidate-spatial-patch-cnn-v1"
    channels: tuple[int, ...] = (16, 24, 32, 48)
    seed: int = 20260821


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )


class CandidatePatchNet(nn.Module):
    """Candidate-wise spatial model, intentionally distinct from the dense FPN detector."""

    contract = CandidateTensorContract()

    def __init__(self, config: CandidateModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or CandidateModelConfig()
        generator_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.config.seed)
            c0, c1, c2, c3 = self.config.channels
            self.features = nn.Sequential(
                ConvBlock(3, c0, stride=1),
                ConvBlock(c0, c1, stride=2),
                ConvBlock(c1, c2, stride=2),
                ConvBlock(c2, c3, stride=2),
            )
            self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c3 * 3 * 3, 64),
                nn.SiLU(),
                nn.Dropout(p=0.05),
                nn.Linear(64, 4),
            )
        finally:
            torch.random.set_rng_state(generator_state)

    def forward_raw(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1:] != (3, 33, 33):
            raise ValueError("CandidatePatchNet requires NCHW [N,3,33,33] patches")
        return self.head(self.spatial_pool(self.features(value)))

    @staticmethod
    def activate(raw: Tensor) -> Tensor:
        marker_probability = torch.sigmoid(raw[:, 0:1])
        offsets = torch.tanh(raw[:, 1:3]) * 0.75
        radius = 2.5 + (torch.sigmoid(raw[:, 3:4]) * 5.5)
        return torch.cat((marker_probability, offsets, radius), dim=1)

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
    marker_loss = functional.binary_cross_entropy_with_logits(
        raw[:, 0],
        labels,
        pos_weight=torch.tensor(positive_weight, device=raw.device),
    )
    positive = labels > 0.5
    if torch.any(positive):
        offset_loss = functional.smooth_l1_loss(
            torch.tanh(raw[positive, 1:3]) * 0.75,
            offsets[positive],
        )
        predicted_radius = 2.5 + (torch.sigmoid(raw[positive, 3]) * 5.5)
        radius_loss = functional.smooth_l1_loss(predicted_radius, radii[positive])
    else:
        offset_loss = raw[:, 1:3].sum() * 0
        radius_loss = raw[:, 3].sum() * 0
    total = marker_loss + (1.5 * offset_loss) + (0.35 * radius_loss)
    return total, {
        "total": float(total.detach()),
        "marker": float(marker_loss.detach()),
        "offset": float(offset_loss.detach()),
        "radius": float(radius_loss.detach()),
    }


__all__ = [
    "CandidateModelConfig",
    "CandidatePatchNet",
    "CandidateTensorContract",
    "candidate_loss",
]
