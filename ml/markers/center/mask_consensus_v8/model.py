# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Mask-consensus dense marker-center and artifact-mask network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from ml.markers.center.model import TensorContract


@dataclass(frozen=True)
class MaskConsensusConfig:
    architecture: str = "mask-consensus-residual-unet-v8"
    channels: tuple[int, int, int] = (24, 40, 64)
    decoder_channels: int = 32
    seed: int = 20268117


class ConvRelu(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel: int = 3,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel // 2)
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.ReLU(inplace=False),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.first = ConvRelu(channels, channels, dilation=dilation)
        self.second = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.second(self.first(value)))


class MaskConsensusCenterNet(nn.Module):
    """Predict centers and missing artifacts, then gate centers by all masks."""

    contract = TensorContract()

    def __init__(self, config: MaskConsensusConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskConsensusConfig()
        c0, c1, c2 = self.config.channels
        decoder = self.config.decoder_channels
        self.stem = ConvRelu(4, c0, 5)
        self.full = nn.Sequential(ResidualBlock(c0), ResidualBlock(c0, dilation=2))
        self.down1 = ConvRelu(c0, c1, stride=2)
        self.context1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1, dilation=2))
        self.down2 = ConvRelu(c1, c2, stride=2)
        self.context2 = nn.Sequential(
            ResidualBlock(c2),
            ResidualBlock(c2, dilation=2),
            ResidualBlock(c2, dilation=4),
        )
        self.lateral1 = nn.Conv2d(c1, decoder, 1)
        self.lateral0 = nn.Conv2d(c0, decoder, 1)
        self.decode1 = nn.Sequential(ConvRelu(c2 + decoder, decoder), ResidualBlock(decoder))
        self.decode0 = nn.Sequential(ConvRelu(decoder * 2, decoder), ResidualBlock(decoder))
        self.center_head = nn.Sequential(ConvRelu(decoder, decoder), nn.Conv2d(decoder, 1, 1))
        self.radius_head = nn.Sequential(ConvRelu(decoder, decoder), nn.Conv2d(decoder, 1, 1))
        self.artifact_head = nn.Sequential(ConvRelu(decoder, decoder), nn.Conv2d(decoder, 1, 1))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                    generator=generator,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.center_head[-1].weight)
        nn.init.constant_(self.center_head[-1].bias, -4.0)
        nn.init.zeros_(self.radius_head[-1].weight)
        nn.init.constant_(self.radius_head[-1].bias, 1.0)
        nn.init.zeros_(self.artifact_head[-1].weight)
        nn.init.constant_(self.artifact_head[-1].bias, -4.0)

    @staticmethod
    def _resize(value: Tensor, reference: Tensor) -> Tensor:
        return functional.interpolate(
            value,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, value: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("Expected float tensor shaped [N, 3, height, width]")
        ink = value[:, 0:1]
        text_mask = value[:, 1:2].clamp(0.0, 1.0)
        seed_artifact = value[:, 2:3].clamp(0.0, 1.0)
        seed_exclusion = torch.maximum(text_mask, seed_artifact)
        safe_ink = ink * (1.0 - seed_exclusion)
        level0 = self.full(self.stem(torch.cat((value, safe_ink), dim=1)))
        level1 = self.context1(self.down1(level0))
        level2 = self.context2(self.down2(level1))
        decoded1 = self.decode1(
            torch.cat((self._resize(level2, level1), self.lateral1(level1)), dim=1)
        )
        decoded0 = self.decode0(
            torch.cat((self._resize(decoded1, level0), self.lateral0(level0)), dim=1)
        )
        learned_artifact = torch.sigmoid(self.artifact_head(decoded0))
        artifact = torch.maximum(seed_artifact, learned_artifact)
        exclusion = torch.maximum(text_mask, artifact)
        center = torch.sigmoid(self.center_head(decoded0)) * (1.0 - exclusion)
        radius = functional.softplus(self.radius_head(decoded0)) + 1.0
        return torch.cat((center, radius, artifact), dim=1)


def create_model() -> MaskConsensusCenterNet:
    return MaskConsensusCenterNet()


def save_checkpoint(
    path: Path,
    model: MaskConsensusCenterNet,
    *,
    selected_threshold: float,
    dataset_manifest_sha256: str,
    training_revision: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "kind": "trained-pytorch-mask-consensus-marker-center-checkpoint",
            "config": asdict(model.config),
            "tensor_contract": asdict(model.contract),
            "selected_threshold": float(selected_threshold),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "training_revision": training_revision,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[MaskConsensusCenterNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config_data = dict(payload["config"])
    config_data["channels"] = tuple(config_data["channels"])
    model = MaskConsensusCenterNet(MaskConsensusConfig(**config_data))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "MaskConsensusCenterNet",
    "MaskConsensusConfig",
    "create_model",
    "load_checkpoint",
    "save_checkpoint",
]
