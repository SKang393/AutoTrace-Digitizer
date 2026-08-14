# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Preregistered dense three-head marker and artifact network."""

from __future__ import annotations

from ml.markers.center.model import CompactCenterNet, ModelConfig


MODEL_CONFIG = ModelConfig(
    architecture="dense-contract-expanded-compact-fpn-v5",
    channels=(16, 24, 40, 64),
    decoder_channels=32,
    seed=20261393,
)


def create_model() -> CompactCenterNet:
    return CompactCenterNet(MODEL_CONFIG)


__all__ = ["MODEL_CONFIG", "create_model"]
