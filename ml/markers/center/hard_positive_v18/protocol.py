# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V18 train-only hard-positive mining declarations."""

TASK = "marker-center"
REVISION = "marker-center-hard-positive-v18"
CANDIDATE_ID = "P1"
V17_RESULT_PATH = "ml/markers/center/metric_aligned_v17/P1_RESULT.json"
V17_RESULT_SHA256 = "90640ce927581d52d4db1119ac1a3b58b5cb2b0b47dd9a91a7e2ec2fcf98a12c"
V13_MANIFEST_SHA256 = "771141bd5d22d2ffdb56e63f193c16c0775a0f47c02313ef67b06bbce9603126"
EVIDENCE_POLICY_PATH = "ml/policy/evidence-policy.json"
EVIDENCE_POLICY_SHA256 = "4dc18136c284b0b1805d3a3b22a9197ad06e6a41f4e43b4e1d4d9245b97e0aed"
MODEL_LICENSE = "Apache-2.0"
ACCEPTANCE_BAR = {"precision_minimum": 0.95, "recall_minimum": 0.95}
THRESHOLDS = (0.25, 0.40, 0.55, 0.70)
WARMUP_EPOCHS = 12
FINISH_EPOCHS = 24
HARD_POSITIVE_THRESHOLD = 0.25
HARD_POSITIVE_REPEAT_COUNT = 3
EXPECTED_WARMUP_EXAMPLES = 3212
EXPECTED_WARMUP_STEPS = 312
RUNTIME_CONTRACT = {"input": ["candidate_count", 3, 33, 33], "output": ["candidate_count", 4], "radius_minimum": 2.5, "radius_maximum": 8.0}
