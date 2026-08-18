# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Checksum-bound evidence and crop extraction for OCR V26."""

from ml.ocr.crop_evidence_role_anchor_v24.pipeline import (
    extract_crop_evidence,
    proposal_crops,
)

__all__ = ["extract_crop_evidence", "proposal_crops"]
