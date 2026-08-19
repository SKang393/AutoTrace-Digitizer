# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed contracts for marker-center seed-refinement V11."""

from __future__ import annotations

from pathlib import Path


EVIDENCE_POLICY = "ml/policy/evidence-policy.json"
TASK = "marker-center"
REVISION = "marker-center-seed-refinement-v11"
ROOT = Path("ml/markers/center/seed_refinement_v11")
TRIGGER_RESULT_PATH = Path("ml/markers/center/decoupled_heads_v10/AGGREGATE_FEASIBILITY.json")
TRIGGER_RESULT_SHA256 = "c0b580c68346124b878521dc6ef46f1e3ed4fe587c29be35be66d5bb8992b62f"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
ARTIFACT_THRESHOLD = 0.35
MINIMUM_ARTIFACT_PRECISION = 0.90
MINIMUM_ARTIFACT_RECALL = 0.95
MINIMUM_PASSING_THRESHOLD_COUNT = 3
ONNX_PARITY_TOLERANCE = 1e-5
EXPERIMENT_BUDGET = 3
DESIGN_SOURCE_PATHS = (
    Path("ml/markers/center/mask_consensus_v8/dataset.py"),
    Path("ml/markers/center/decoupled_heads_v10/AGGREGATE_FEASIBILITY.json"),
    ROOT / "dataset.py",
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
