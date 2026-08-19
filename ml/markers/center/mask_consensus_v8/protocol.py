# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed contracts for the marker-center mask-consensus V8 experiment."""

from __future__ import annotations

from pathlib import Path


EVIDENCE_POLICY = "ml/policy/evidence-policy.json"
TASK = "marker-center"
REVISION = "marker-center-mask-consensus-v8"
ROOT = Path("ml/markers/center/mask_consensus_v8")
TRIGGER_RESULT_PATH = Path("ml/markers/center/domain_randomized_v7/P3_RESULT.json")
TRIGGER_RESULT_SHA256 = "b4ac9d69b45971e90f6a91317ffae58e3caf07b16959002e4381fe20dcfe3abc"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
ARTIFACT_THRESHOLD = 0.35
MINIMUM_ARTIFACT_PRECISION = 0.90
MINIMUM_ARTIFACT_RECALL = 0.95
MINIMUM_PASSING_THRESHOLD_COUNT = 3
ONNX_PARITY_TOLERANCE = 1e-5
EXPERIMENT_BUDGET = 3
DESIGN_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "model.py",
    ROOT / "prepare_split.py",
    ROOT / "protocol.py",
)


__all__ = [
    "ARTIFACT_THRESHOLD",
    "DESIGN_SOURCE_PATHS",
    "EVIDENCE_POLICY",
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
