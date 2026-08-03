# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Compact trainable marker-center network and frozen tensor contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


@dataclass(frozen=True)
class TensorContract:
    input_name: str = "image_and_masks"
    input_layout: str = "NCHW"
    input_dtype: str = "float32"
    input_channels: tuple[str, ...] = (
        "ink_probability",
        "text_mask",
        "artifact_mask",
    )
    input_range: tuple[float, float] = (0.0, 1.0)
    output_name: str = "marker_heads"
    output_layout: str = "NCHW"
    output_channels: tuple[str, ...] = (
        "center_probability",
        "radius_pixels",
        "artifact_probability",
    )
    output_stride: int = 1
    coordinate_space: str = "model_tensor"
    dynamic_axes: tuple[str, ...] = ("batch", "height", "width")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "compact-pplcnet-depthwise-fpn-v1"
    channels: tuple[int, ...] = (12, 16, 24, 32)
    decoder_channels: int = 16
    seed: int = 20260803


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, *, stride: int = 1) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(),
        )


class DepthwiseSeparableBlock(nn.Module):
    """PP-LCNet/MobileNet-style depthwise spatial and pointwise channel mixing."""

    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.Hardswish(),
        )
        self.pointwise = ConvNormAct(input_channels, output_channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(value))


class CompactCenterNet(nn.Module):
    """Stride-1 three-head detector with a compact depthwise encoder and FPN decoder."""

    contract = TensorContract()

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c0, c1, c2, c3 = self.config.channels
        decoder = self.config.decoder_channels
        self.stem = ConvNormAct(3, c0, 3)
        self.encoder1 = DepthwiseSeparableBlock(c0, c1, stride=2)
        self.encoder2 = DepthwiseSeparableBlock(c1, c2, stride=2)
        self.encoder3 = DepthwiseSeparableBlock(c2, c3, stride=2)
        self.lateral3 = nn.Conv2d(c3, decoder, 1)
        self.lateral2 = nn.Conv2d(c2, decoder, 1)
        self.lateral1 = nn.Conv2d(c1, decoder, 1)
        self.lateral0 = nn.Conv2d(c0, decoder, 1)
        self.refine2 = DepthwiseSeparableBlock(decoder, decoder, stride=1)
        self.refine1 = DepthwiseSeparableBlock(decoder, decoder, stride=1)
        self.refine0 = DepthwiseSeparableBlock(decoder, decoder, stride=1)
        self.head = nn.Sequential(
            DepthwiseSeparableBlock(decoder, decoder, stride=1),
            nn.Conv2d(decoder, 3, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu", generator=generator)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.head[-1]
        if not isinstance(final, nn.Conv2d):
            raise TypeError("The final head must be a convolution")
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor((-4.0, 1.0, -4.0)))

    @staticmethod
    def _upsample_add(value: Tensor, lateral: Tensor) -> Tensor:
        return functional.interpolate(value, size=lateral.shape[-2:], mode="bilinear", align_corners=False) + lateral

    def forward(self, value: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("Expected float tensor shaped [N, 3, height, width]")
        level0 = self.stem(value)
        level1 = self.encoder1(level0)
        level2 = self.encoder2(level1)
        level3 = self.encoder3(level2)
        decoded = self.lateral3(level3)
        decoded = self.refine2(self._upsample_add(decoded, self.lateral2(level2)))
        decoded = self.refine1(self._upsample_add(decoded, self.lateral1(level1)))
        decoded = self.refine0(self._upsample_add(decoded, self.lateral0(level0)))
        logits = self.head(decoded)
        center = torch.sigmoid(logits[:, 0:1])
        radius = functional.softplus(logits[:, 1:2]) + 1.0
        artifact = torch.sigmoid(logits[:, 2:3])
        return torch.cat((center, radius, artifact), dim=1)


def save_checkpoint(
    path: Path,
    model: CompactCenterNet,
    *,
    selected_threshold: float,
    dataset_manifest_sha256: str,
    training_revision: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "kind": "trained-pytorch-marker-center-checkpoint",
            "config": asdict(model.config),
            "tensor_contract": asdict(model.contract),
            "selected_threshold": float(selected_threshold),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "training_revision": training_revision,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[CompactCenterNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config_data = dict(payload["config"])
    config_data["channels"] = tuple(config_data["channels"])
    model = CompactCenterNet(ModelConfig(**config_data))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "CompactCenterNet",
    "DepthwiseSeparableBlock",
    "ModelConfig",
    "TensorContract",
    "load_checkpoint",
    "save_checkpoint",
]
