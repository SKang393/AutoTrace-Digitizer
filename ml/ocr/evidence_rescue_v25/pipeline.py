# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Checksum-bound OCR evidence extraction shared with the frozen V24 parent."""

from ml.ocr.crop_evidence_role_anchor_v24.pipeline import (
    extract_crop_evidence,
    proposal_crops,
)

__all__ = ["extract_crop_evidence", "proposal_crops"]
