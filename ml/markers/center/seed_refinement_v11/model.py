# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Non-monotonic seed-refinement marker-center and artifact-mask network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from ml.markers.center.model import TensorContract


@dataclass(frozen=True)
class SeedRefinementConfig:
    architecture: str = "marker-center-seed-refinement-v11"
    channels: tuple[int, int, int] = (16, 28, 44)
    decoder_channels: int = 24
    seed: int = 202611411
    fixed_radius_pixels: float = 2.5
    maximum_logit_correction: float = 6.0


class ConvRelu(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=True),
            nn.ReLU(inplace=False),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=True),
            nn.ReLU(inplace=False),
        )
        self.second = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation, bias=True
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.second(self.first(value)))


class RefinementTower(nn.Module):
    def __init__(self, config: SeedRefinementConfig) -> None:
        super().__init__()
        c0, c1, c2 = config.channels
        decoder = config.decoder_channels
        self.stem = ConvRelu(4, c0)
        self.full = nn.Sequential(ResidualBlock(c0), ResidualBlock(c0, dilation=2))
        self.down1 = ConvRelu(c0, c1, stride=2)
        self.context1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1, dilation=2))
        self.down2 = ConvRelu(c1, c2, stride=2)
        self.context2 = nn.Sequential(ResidualBlock(c2), ResidualBlock(c2, dilation=2))
        self.lateral1 = nn.Conv2d(c1, decoder, 1)
        self.lateral0 = nn.Conv2d(c0, decoder, 1)
        self.decode1 = nn.Sequential(ConvRelu(c2 + decoder, decoder), ResidualBlock(decoder))
        self.decode0 = nn.Sequential(ConvRelu(decoder * 2, decoder), ResidualBlock(decoder))

    @staticmethod
    def _resize(value: Tensor, reference: Tensor) -> Tensor:
        return functional.interpolate(
            value, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, value: Tensor) -> Tensor:
        ink = value[:, 0:1]
        text_mask = value[:, 1:2].clamp(0.0, 1.0)
        safe_ink = ink * (1.0 - text_mask)
        level0 = self.full(self.stem(torch.cat((value, safe_ink), dim=1)))
        level1 = self.context1(self.down1(level0))
        level2 = self.context2(self.down2(level1))
        decoded1 = self.decode1(
            torch.cat((self._resize(level2, level1), self.lateral1(level1)), dim=1)
        )
        return self.decode0(
            torch.cat((self._resize(decoded1, level0), self.lateral0(level0)), dim=1)
        )


class SeedRefinementMarkerNet(nn.Module):
    """Refine the artifact seed in both directions before suppressing centers."""

    contract = TensorContract()

    def __init__(self, config: SeedRefinementConfig | None = None) -> None:
        super().__init__()
        self.config = config or SeedRefinementConfig()
        self.center_tower = RefinementTower(self.config)
        self.artifact_tower = RefinementTower(self.config)
        self.center_head = nn.Conv2d(self.config.decoder_channels, 1, 1)
        self.artifact_head = nn.Conv2d(self.config.decoder_channels, 1, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu", generator=generator
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.center_head.weight)
        nn.init.constant_(self.center_head.bias, -4.0)
        nn.init.zeros_(self.artifact_head.weight)
        nn.init.zeros_(self.artifact_head.bias)

    def forward(self, value: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export() and (value.ndim != 4 or value.shape[1] != 3):
            raise ValueError("Expected float tensor shaped [N, 3, height, width]")
        text_mask = value[:, 1:2].clamp(0.0, 1.0)
        seed_artifact = value[:, 2:3].clamp(0.0, 1.0)
        seed_probability = seed_artifact.clamp(0.01, 0.99)
        seed_logit = torch.log(seed_probability) - torch.log1p(-seed_probability)
        correction = (
            torch.tanh(self.artifact_head(self.artifact_tower(value)))
            * self.config.maximum_logit_correction
        )
        artifact = torch.sigmoid(seed_logit + correction)
        learned_center = torch.sigmoid(self.center_head(self.center_tower(value)))
        center = learned_center * (1.0 - torch.maximum(text_mask, artifact.detach()))
        radius = torch.ones_like(center) * self.config.fixed_radius_pixels
        return torch.cat((center, radius, artifact), dim=1)


def create_model() -> SeedRefinementMarkerNet:
    return SeedRefinementMarkerNet()


def save_checkpoint(
    path: Path,
    model: SeedRefinementMarkerNet,
    *,
    selected_threshold: float,
    dataset_manifest_sha256: str,
    training_revision: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "kind": "trained-pytorch-seed-refinement-marker-checkpoint",
            "config": asdict(model.config),
            "tensor_contract": asdict(model.contract),
            "selected_threshold": float(selected_threshold),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "training_revision": training_revision,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[SeedRefinementMarkerNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config_data = dict(payload["config"])
    config_data["channels"] = tuple(config_data["channels"])
    model = SeedRefinementMarkerNet(SeedRefinementConfig(**config_data))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "SeedRefinementConfig",
    "SeedRefinementMarkerNet",
    "create_model",
    "load_checkpoint",
    "save_checkpoint",
]
