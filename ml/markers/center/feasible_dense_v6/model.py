# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh compact dense model for feasible marker contract V6."""

from __future__ import annotations

from ml.markers.center.model import CompactCenterNet, ModelConfig


MODEL_CONFIG = ModelConfig(
    architecture="feasible-dense-expanded-compact-fpn-v6",
    channels=(16, 24, 40, 64),
    decoder_channels=32,
    seed=20266393,
)


def create_model() -> CompactCenterNet:
    return CompactCenterNet(MODEL_CONFIG)


__all__ = ["MODEL_CONFIG", "create_model"]
