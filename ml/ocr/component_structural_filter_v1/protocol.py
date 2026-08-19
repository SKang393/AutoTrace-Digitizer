# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen identity and gates for the OCR structural-filter repair."""

from __future__ import annotations


EVIDENCE_POLICY = "ml/policy/evidence-policy.json"
TASK = "ocr-recognition"
REVISION = "graph-numeric-component-structural-filter-v1"
PUBLIC_REVISION = f"{REVISION}-public-v1"
CANDIDATE_ID = "P1"
SOURCE_REVISION = "graph-numeric-component-geometric-v4"
SOURCE_CANDIDATE_ID = "P3"
SOURCE_ONNX_SHA256 = "941b19a8f700484a039335d25ecc47f41de1976b3b2016653c6beb2a4eb51894"
SOURCE_REPORT_SHA256 = "807db11ffe147f9eadf025bede7403469d064161bb4ffd26349e857eee7588ca"
SOURCE_CHECKPOINT_SHA256 = "955edefd1a8e2bf713f8627cf88af4aa2977c1afdfdda86b5d0d86ae2911ed7e"
SOURCE_CONFIDENCE_THRESHOLD = 0.55
STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO = 0.75

VALIDATION_EXACT_MATCH_MINIMUM = 0.90
VALIDATION_CER_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
MARKER_EXCLUSION_ACCURACY_MINIMUM = 1.0
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 0.0001
SEALED_EXACT_MATCH_MINIMUM = 0.90
SEALED_CER_MAXIMUM = 0.05


__all__ = [name for name in globals() if name.isupper()]
