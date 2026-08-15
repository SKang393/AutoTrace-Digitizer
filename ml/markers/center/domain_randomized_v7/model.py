# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Compact dense model for the marker-center V7 generalization experiment."""

from __future__ import annotations

from ml.markers.center.model import CompactCenterNet, ModelConfig


MODEL_CONFIG = ModelConfig(
    architecture="domain-randomized-expanded-compact-fpn-v7",
    channels=(16, 24, 40, 64),
    decoder_channels=32,
    seed=20267431,
)


def create_model() -> CompactCenterNet:
    return CompactCenterNet(MODEL_CONFIG)


__all__ = ["MODEL_CONFIG", "create_model"]
