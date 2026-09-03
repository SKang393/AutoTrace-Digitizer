# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen, non-consuming V24 feasibility declarations."""

TASK = "marker-center"
REVISION = "marker-center-mask-preserving-v24-feasibility"
TRAINING_REVISION = "marker-center-mask-preserving-v24"
TRAINING_CANDIDATE_ID = "P1"
V21_ONNX_SHA256 = "0b413db48f8e6707ee5ec99afff4cd8ec3d25c6b8a8d9f165bd416deb4578a38"
GENERATOR_AUDIT_PATH = "ml/markers/center/real_range_generator_v1/AUDIT.json"
GENERATOR_AUDIT_SHA256 = "57c35c2f7c270d9dedbb7e7a60c1c46b80109e7717af75086dfc83f1b24b1b7f"
GENERATOR_DEV_SPLIT_SHA256 = "2eb77f1efb641f432c5e8581b95d09c1c6a05b9f9781019f2862adfdeefc8edb"
EVIDENCE_POLICY_PATH = "ml/policy/evidence-policy.json"
EVIDENCE_POLICY_SHA256 = "4dc18136c284b0b1805d3a3b22a9197ad06e6a41f4e43b4e1d4d9245b97e0aed"
ACCEPTANCE_BARS_PATH = "ml/policy/acceptance-bars.json"
ACCEPTANCE_BARS_SHA256 = "aab9f2ab60cf166828f0928b8496f537341870fd457d0408952e22549fc53a56"
CONFIDENCE_THRESHOLD = 0.25
OFFSET_SCALE = 4.0
RADIUS_CLIP_PX = (2.5, 8.0)
RING_RADII_PX = tuple(range(3, 13))
OPTIMIZER_STEPS = 0
TRAINING_PERFORMED = False
CANDIDATE_CONSUMED = False
PROVIDER = "CPUExecutionProvider"
INPUT_SHAPE = ["candidate_count", 3, 33, 33]
OUTPUT_SHAPE = ["candidate_count", 4]
