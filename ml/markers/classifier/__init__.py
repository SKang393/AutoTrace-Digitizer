# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic marker patch classification training package."""

from .dataset import ARTIFACT_KINDS, FILL_NAMES, SHAPE_NAMES, PatchSample, build_fixed_dataset
from .model import CompactMarkerClassifier, ClassifierConfig, TensorContract

__all__ = [
    "ARTIFACT_KINDS",
    "FILL_NAMES",
    "SHAPE_NAMES",
    "ClassifierConfig",
    "CompactMarkerClassifier",
    "PatchSample",
    "TensorContract",
    "build_fixed_dataset",
]
