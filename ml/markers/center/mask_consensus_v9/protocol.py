# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed contracts for marker-center mask-consensus V9."""

from __future__ import annotations

from pathlib import Path


TASK = "marker-center"
REVISION = "marker-center-mask-consensus-v9"
ROOT = Path("ml/markers/center/mask_consensus_v9")
TRIGGER_RESULT_PATH = Path("ml/markers/center/mask_consensus_v8/P3_RESULT.json")
TRIGGER_RESULT_SHA256 = "fd2dfa1a196e4ddbb63d5099fbc44e734630370e211e7917509286b5ee3204f8"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
ARTIFACT_THRESHOLD = 0.35
MINIMUM_ARTIFACT_PRECISION = 0.90
MINIMUM_ARTIFACT_RECALL = 0.95
MINIMUM_PASSING_THRESHOLD_COUNT = 3
ONNX_PARITY_TOLERANCE = 1e-5
PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE = 1e-6
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
    "PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE",
    "REVISION",
    "ROOT",
    "TASK",
    "THRESHOLDS",
    "TRIGGER_RESULT_PATH",
    "TRIGGER_RESULT_SHA256",
]
