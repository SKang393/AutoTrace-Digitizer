# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen declarations for V16 from-scratch synthetic training."""

TASK = "marker-center"
REVISION = "marker-center-scale-classifier-v16"
CANDIDATE_ID = "P1"
EVIDENCE_POLICY_PATH = "ml/policy/evidence-policy.json"
EVIDENCE_POLICY_SHA256 = "4dc18136c284b0b1805d3a3b22a9197ad06e6a41f4e43b4e1d4d9245b97e0aed"
V13_MANIFEST_SHA256 = "771141bd5d22d2ffdb56e63f193c16c0775a0f47c02313ef67b06bbce9603126"
V15_RESULT_PATH = "ml/markers/center/geometry_finetune_v15/P1_RESULT.json"
V15_RESULT_SHA256 = "5fa8853427c6a3f4bb24073a18f0f3b99c31aaa67105ad511d25dbc0a9b56520"
MODEL_LICENSE = "Apache-2.0"
ACCEPTANCE_BAR = {"precision_minimum": 0.95, "recall_minimum": 0.95}
THRESHOLDS = (0.25, 0.40, 0.55, 0.70)
RUNTIME_CONTRACT = {"input": ["candidate_count", 3, 33, 33], "output": ["candidate_count", 4], "radius_minimum": 2.5, "radius_maximum": 8.0}
