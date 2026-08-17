# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed contracts for marker-center decoupled-head V10."""

from __future__ import annotations

from pathlib import Path


TASK = "marker-center"
REVISION = "marker-center-decoupled-heads-v10"
ROOT = Path("ml/markers/center/decoupled_heads_v10")
TRIGGER_RESULT_PATH = Path("ml/markers/center/mask_consensus_v9/P3_RESULT.json")
TRIGGER_RESULT_SHA256 = "542c6093415c251256ef0cbb3e25ac97251d08b1ca1e259317352117943c6f79"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
ARTIFACT_THRESHOLD = 0.35
MINIMUM_ARTIFACT_PRECISION = 0.90
MINIMUM_ARTIFACT_RECALL = 0.95
MINIMUM_PASSING_THRESHOLD_COUNT = 3
ONNX_PARITY_TOLERANCE = 1e-5
EXPERIMENT_BUDGET = 3
DESIGN_SOURCE_PATHS = (
    Path("ml/markers/center/mask_consensus_v8/dataset.py"),
    ROOT / "dataset.py",
    ROOT / "prepare_split.py",
    ROOT / "protocol.py",
)


__all__ = [
    "ARTIFACT_THRESHOLD",
    "DESIGN_SOURCE_PATHS",
    "EXPERIMENT_BUDGET",
    "MINIMUM_ARTIFACT_PRECISION",
    "MINIMUM_ARTIFACT_RECALL",
    "MINIMUM_PASSING_THRESHOLD_COUNT",
    "ONNX_PARITY_TOLERANCE",
    "REVISION",
    "ROOT",
    "TASK",
    "THRESHOLDS",
    "TRIGGER_RESULT_PATH",
    "TRIGGER_RESULT_SHA256",
]

