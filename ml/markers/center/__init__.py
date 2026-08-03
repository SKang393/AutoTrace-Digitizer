# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic marker-center training and evaluation scaffold."""

from .model import CompactCenterNet, ModelConfig, TensorContract
from .postprocess import Detection, detect_heads

__all__ = [
    "CompactCenterNet",
    "Detection",
    "ModelConfig",
    "TensorContract",
    "detect_heads",
]
