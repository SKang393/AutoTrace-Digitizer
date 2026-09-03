# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V21 declarations for an isolated focal-loss confidence repair."""

from __future__ import annotations

TASK = "marker-center"
REVISION = "marker-center-focal-confidence-v21"
CANDIDATE_ID = "P1"
ARCHITECTURE = "scale-separated-multiscale-patch-cnn-v16"

V20_RESULT_PATH = "ml/markers/center/tail_coverage_v20/P1_RESULT.json"
V20_RESULT_SHA256 = "2689cefc8acad6f5f11f798a3c4a0c86da92cc964681101f6a6fbb0b1e7e9185"
V20_DIAGNOSTIC_PATH = "ml/markers/center/tail_coverage_v20/diagnostics/V20_DIAGNOSTIC.json"
V20_DIAGNOSTIC_SHA256 = "d9245d673d00e795f7095424250b25cabd1124f880568c944da2966da6cbedda"
V13_MANIFEST_SHA256 = "771141bd5d22d2ffdb56e63f193c16c0775a0f47c02313ef67b06bbce9603126"
EVIDENCE_POLICY_PATH = "ml/policy/evidence-policy.json"
EVIDENCE_POLICY_SHA256 = "4dc18136c284b0b1805d3a3b22a9197ad06e6a41f4e43b4e1d4d9245b97e0aed"
MODEL_LICENSE = "Apache-2.0"

ACCEPTANCE_BAR = {"precision_minimum": 0.95, "recall_minimum": 0.95}
THRESHOLDS = (0.25, 0.40, 0.55, 0.70)
LABEL_POSITIVE_DISTANCE_PX = 3.0
FIXED_CONFIDENCE_THRESHOLD = 0.25
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
CLASSIFICATION_LOSS = "binary_focal_loss_with_logits"
RUNTIME_CONTRACT = {
    "input": ["candidate_count", 3, 33, 33],
    "output": ["candidate_count", 4],
    "radius_minimum": 2.5,
    "radius_maximum": 8.0,
}
GEOMETRY_VETO_GUARD = {
    "diagnostic_marker_geometry_veto_count": 2,
    "diagnostic_unmasked_artifact_veto_count": 1,
    "maximum_allowed_marker_geometry_veto_count": 2,
    "maximum_allowed_unmasked_artifact_veto_count": 1,
    "prohibited_kind": "line_intersection",
}
SEALED_RUN_BUDGET = 1
TRAIN_SCENE_COUNT = 21

# The V20 generator is imported unchanged. These hashes make the frozen input
# dependency explicit without duplicating or rewriting its scene tensors.
V20_PROTOCOL_SHA256 = "13b12549e5107274ae85863907cb9bcfed3038fa393afdb5cb0115bb22fc1f45"
V20_TRAINING_FAMILIES_SHA256 = "845ce4a7a0637d2caf2b4a6cf0bc6e6cb6e8be099318cd108ba8fd99d64c39b9"
V20_TRAIN_RUNNER_SHA256 = "f0bffcea3816613712328d1010d712d2ad7ab4515f24a2c3bbea265d19925868"

__all__ = [
    "ACCEPTANCE_BAR", "ARCHITECTURE", "CANDIDATE_ID", "CLASSIFICATION_LOSS",
    "EVIDENCE_POLICY_PATH", "EVIDENCE_POLICY_SHA256", "FIXED_CONFIDENCE_THRESHOLD",
    "FOCAL_ALPHA", "FOCAL_GAMMA", "GEOMETRY_VETO_GUARD", "LABEL_POSITIVE_DISTANCE_PX",
    "MODEL_LICENSE", "RUNTIME_CONTRACT", "REVISION", "SEALED_RUN_BUDGET", "TASK",
    "THRESHOLDS", "TRAIN_SCENE_COUNT", "V13_MANIFEST_SHA256", "V20_DIAGNOSTIC_PATH",
    "V20_DIAGNOSTIC_SHA256", "V20_PROTOCOL_SHA256", "V20_RESULT_PATH", "V20_RESULT_SHA256",
    "V20_TRAINING_FAMILIES_SHA256", "V20_TRAIN_RUNNER_SHA256",
]
