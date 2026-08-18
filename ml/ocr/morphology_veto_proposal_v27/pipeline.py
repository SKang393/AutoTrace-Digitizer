# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Checksum-bound proposal inputs for OCR V27."""

from ml.ocr.scene_topology_proposal_v26.pipeline import (
    extract_crop_evidence,
    proposal_crops,
)

from .features import structure_features

__all__ = ["extract_crop_evidence", "proposal_crops", "structure_features"]
