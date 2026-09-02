# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen declarations for V15 authorization and train/dev evaluation."""

TASK = "marker-center"
REVISION = "marker-center-geometry-finetune-v15"
CANDIDATE_ID = "P1"
EVIDENCE_POLICY_PATH = "ml/policy/evidence-policy.json"
MODEL_LICENSE = "Apache-2.0"
SOURCE_CHECKPOINT_PATH = "ml/markers/center/artifacts/runtime-consistency-v2/P2-run/marker-center-runtime-consistency-p2.pt"
SOURCE_CHECKPOINT_SHA256 = "6b670a6f29454d7f63527f57210aa918540a817fca156a71b96872ff09aa2787"
ACCEPTANCE_BAR = {"precision_minimum": 0.95, "recall_minimum": 0.95}
THRESHOLDS = (0.25, 0.40, 0.55, 0.70)
RUNTIME_RADIUS_CONTRACT = {"minimum_pixels": 2.5, "maximum_pixels": 8.0}
