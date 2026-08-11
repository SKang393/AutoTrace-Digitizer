# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen public-gate contract for OCR component-region V6."""

from __future__ import annotations

from pathlib import Path


EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/pipeline.py"),
    Path("ml/ocr/component_region_detector_v6/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/sealed_gate.py"),
    Path("ml/markers/gate_seal.py"),
)

GATE_CONFIG = {
    "evaluation_limit": 1,
    "exact_region_count_every_fixture": True,
    "false_region_count": 0,
    "missed_region_count": 0,
    "duplicate_region_count": 0,
    "prohibited_structure_hits": 0,
    "onnx_parity_maximum_absolute_error": 1e-5,
    "provider": "CPUExecutionProvider",
}

