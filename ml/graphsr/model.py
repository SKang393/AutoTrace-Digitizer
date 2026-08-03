# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Compact chart-preserving x2 super-resolution network and tensor contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


@dataclass(frozen=True)
class TensorContract:
    """Runtime tensor boundary shared by PyTorch and ONNX inference."""

    input_name: str = "image"
    input_layout: str = "NCHW"
    input_dtype: str = "float32"
    input_channels: tuple[str, ...] = ("red", "green", "blue")
    input_range: tuple[float, float] = (0.0, 1.0)
    output_name: str = "enhanced"
    output_layout: str = "NCHW"
    output_dtype: str = "float32"
    output_channels: tuple[str, ...] = ("red", "green", "blue")
    output_range: tuple[float, float] = (0.0, 1.0)
    scale: int = 2
    coordinate_space: str = "enhanced_pixels"
    coordinate_mapping: str = "output_xy = input_xy * 2; map detections back by dividing by 2"
    dynamic_axes: tuple[str, ...] = ("batch", "height", "width")


@dataclass(frozen=True)
class GraphSRConfig:
    """Small SRVGG configuration suitable for CPU tests and local training."""

    architecture: str = "graphsr-srvgg-x2-v1"
    input_channels: int = 3
    output_channels: int = 3
    channels: int = 24
    blocks: int = 4
    scale: int = 2
    seed: int = 20260803

    def __post_init__(self) -> None:
        if self.input_channels != 3 or self.output_channels != 3:
            raise ValueError("GraphSR requires three-channel RGB input and output")
        if self.channels < 4 or self.channels > 256:
            raise ValueError("channels must be in the range 4 through 256")
        if self.blocks < 1 or self.blocks > 32:
            raise ValueError("blocks must be in the range 1 through 32")
        if self.scale != 2:
            raise ValueError("GraphSR-x2 supports scale=2 only")


class ResidualConvBlock(nn.Module):
    """Two local convolutions with a low-amplitude residual update."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.convolution2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.convolution2(self.activation(self.convolution1(value)))
        return value + residual * 0.2


class GraphSRx2(nn.Module):
    """SRVGG/Real-ESRNet-style x2 network with an interpolation residual.

    The explicit interpolation path anchors geometry. The learned branch predicts
    a bounded correction instead of reconstructing the entire image from scratch.
    """

    contract = TensorContract()

    def __init__(self, config: GraphSRConfig | None = None) -> None:
        super().__init__()
        self.config = config or GraphSRConfig()
        channels = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.input_channels, channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )
        self.body = nn.Sequential(*(ResidualConvBlock(channels) for _ in range(self.config.blocks)))
        self.trunk = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                self.config.output_channels * self.config.scale * self.config.scale,
                3,
                padding=1,
            ),
            nn.PixelShuffle(self.config.scale),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.1,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                    generator=generator,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.upsample[0]
        if not isinstance(final, nn.Conv2d):
            raise TypeError("The GraphSR output projection must be a convolution")
        # Starting from the interpolation anchor is safer for chart geometry.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, value: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1] != self.config.input_channels:
                raise ValueError("Expected float RGB tensor shaped [N, 3, height, width]")
            if value.shape[-2] < 2 or value.shape[-1] < 2:
                raise ValueError("Input height and width must both be at least 2")
            if not value.is_floating_point():
                raise ValueError("GraphSR input must use a floating-point dtype")
        baseline = functional.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        shallow = self.stem(value)
        learned = self.upsample(self.trunk(self.body(shallow)) + shallow)
        return torch.clamp(baseline + learned, 0.0, 1.0)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_artifact_outside_repository(path: Path) -> Path:
    """Confine local artifacts to Session 07's explicit Git-ignored roots.

    The historical function name means outside Git tracking, not outside the
    project directory. Keeping local artifacts inside this workspace prevents
    the training tool from broadening its filesystem scope.
    """

    resolved = path.expanduser().resolve()
    root = _repository_root()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("Model artifacts must remain inside the current project directory") from error
    allowed_roots = (
        Path("ml/graphsr/checkpoints"),
        Path("ml/graphsr/runs"),
        Path("ml/graphsr/cache"),
    )
    if any(relative == allowed or allowed in relative.parents for allowed in allowed_roots):
        return resolved
    raise ValueError(
        "Model artifacts must use ml/graphsr/checkpoints, ml/graphsr/runs, or ml/graphsr/cache"
    )


def save_checkpoint(
    path: Path,
    model: GraphSRx2,
    *,
    dataset_identity: str,
    training_revision: str,
    loss_weights: dict[str, float],
    seed: int,
) -> Path:
    """Save a local training checkpoint after enforcing the no-weights-in-Git rule."""

    output = ensure_artifact_outside_repository(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "kind": "trained-pytorch-graphsr-x2-checkpoint",
            "config": asdict(model.config),
            "tensor_contract": asdict(model.contract),
            "dataset_identity": dataset_identity,
            "training_revision": training_revision,
            "loss_weights": dict(loss_weights),
            "seed": int(seed),
            "state_dict": model.state_dict(),
        },
        output,
    )
    return output


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[GraphSRx2, dict[str, Any]]:
    """Load a trusted local checkpoint without downloading any artifact."""

    payload = torch.load(path, map_location=device, weights_only=False)
    config = GraphSRConfig(**dict(payload["config"]))
    model = GraphSRx2(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "GraphSRConfig",
    "GraphSRx2",
    "ResidualConvBlock",
    "TensorContract",
    "ensure_artifact_outside_repository",
    "load_checkpoint",
    "save_checkpoint",
]
