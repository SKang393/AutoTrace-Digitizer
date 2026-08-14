# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Reuse the frozen V16 aggregate evaluator for the unchanged V17 contract."""

from ml.ocr.margin_robust_layout_proposal_role_v16.pipeline import Runner, evaluate_thresholds

__all__ = ["Runner", "evaluate_thresholds"]
