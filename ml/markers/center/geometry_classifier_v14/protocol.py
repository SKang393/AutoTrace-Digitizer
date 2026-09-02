# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen declarations for the V14 deterministic classifier diagnostic."""

TASK = "marker-center"
REVISION = "marker-center-geometry-classifier-v14"
ISOLATED_CHANGE = "fixed deterministic compactness, isotropy, radial-support, line-evidence, and mask score on V13 proposals"
METRIC = "canonical maximum-cardinality 5px marker precision and recall"
ACCEPTANCE_BAR = {"precision_minimum": 0.95, "recall_minimum": 0.95}
THRESHOLDS = (0.20, 0.35, 0.50, 0.65, 0.80)
BUDGET = {"sealed_runs": 0, "public_gate_evaluations": 0}
EVIDENCE_POLICY_PATH = "ml/policy/evidence-policy.json"
SOURCE_BUNDLE_PATHS = (
    "ml/markers/center/geometry_classifier_v14/protocol.py",
    "ml/markers/center/geometry_classifier_v14/features.py",
    "ml/markers/center/geometry_classifier_v14/diagnose.py",
    "ml/markers/center/proposal_geometry_v13/dataset.py",
    "ml/markers/center/proposal_geometry_v13/geometry.py",
)
EVALUATOR_SOURCE_PATHS = (
    "ml/markers/center/geometry_classifier_v14/diagnose.py",
    "ml/markers/center/geometry_classifier_v14/features.py",
    "ml/markers/center/metrics.py",
)
