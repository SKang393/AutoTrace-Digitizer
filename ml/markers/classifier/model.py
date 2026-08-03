# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Compact trainable marker patch classifier and tensor contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from .dataset import FILL_NAMES, PATCH_SIZE, SHAPE_NAMES


@dataclass(frozen=True)
class TensorContract:
    input_name: str = "marker_patch"
    input_layout: str = "NCHW"
    input_dtype: str = "float32"
    input_channels: tuple[str, ...] = ("ink_probability",)
    input_range: tuple[float, float] = (0.0, 1.0)
    patch_size: tuple[int, int] = (PATCH_SIZE, PATCH_SIZE)
    output_names: tuple[str, ...] = ("shape_logits", "fill_logits", "artifact_logit", "embedding")
    shape_classes: tuple[str, ...] = SHAPE_NAMES
    fill_classes: tuple[str, ...] = FILL_NAMES
    embedding_normalization: str = "l2"
    coordinate_space: str = "model_tensor"


@dataclass(frozen=True)
class ClassifierConfig:
    architecture: str = "compact-spatial-cnn-patch-classifier-v3"
    channels: tuple[int, ...] = (16, 24, 40)
    hidden_size: int = 96
    embedding_size: int = 12
    seed: int = 20260803


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(),
        )


class DepthwiseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, stride=stride, padding=1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Hardswish(),
        )
        self.pointwise = ConvNormAct(input_channels, output_channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(value))


class CompactMarkerClassifier(nn.Module):
    """One encoder with independent shape, fill, artifact, and embedding heads."""

    contract = TensorContract()

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ClassifierConfig()
        c0, c1, c2 = self.config.channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c0, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c0, c0, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c0, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 4 * 4, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.08),
        )
        self.shape_head = nn.Linear(self.config.hidden_size, len(SHAPE_NAMES))
        self.fill_head = nn.Linear(self.config.hidden_size, len(FILL_NAMES))
        self.artifact_head = nn.Linear(self.config.hidden_size, 1)
        self.embedding_head = nn.Linear(self.config.hidden_size, self.config.embedding_size)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu", generator=generator)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not torch.onnx.is_in_onnx_export():
            if value.ndim != 4 or value.shape[1:] != (1, PATCH_SIZE, PATCH_SIZE):
                raise ValueError(f"Expected float tensor shaped [N, 1, {PATCH_SIZE}, {PATCH_SIZE}]")
        features = self.projection(self.encoder(value))
        embedding = functional.normalize(self.embedding_head(features), p=2, dim=1, eps=1e-8)
        return self.shape_head(features), self.fill_head(features), self.artifact_head(features), embedding


def save_checkpoint(
    path: Path,
    model: CompactMarkerClassifier,
    *,
    dataset_manifest_sha256: str,
    shape_temperature: float,
    fill_temperature: float,
    training_revision: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "kind": "trained-pytorch-marker-patch-classifier",
            "config": asdict(model.config),
            "tensor_contract": asdict(model.contract),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "shape_temperature": float(shape_temperature),
            "fill_temperature": float(fill_temperature),
            "training_revision": training_revision,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[CompactMarkerClassifier, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config_data = dict(payload["config"])
    config_data["channels"] = tuple(config_data["channels"])
    model = CompactMarkerClassifier(ClassifierConfig(**config_data))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "ClassifierConfig",
    "CompactMarkerClassifier",
    "DepthwiseBlock",
    "TensorContract",
    "load_checkpoint",
    "save_checkpoint",
]
