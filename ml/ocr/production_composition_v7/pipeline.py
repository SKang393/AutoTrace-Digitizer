# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Reuse the exact V6 scientific composition without behavior changes."""

from ml.ocr.production_composition_v6.pipeline import DirectRunner, DirectTensorEvidence, evaluate_scenes

__all__ = ["DirectRunner", "DirectTensorEvidence", "evaluate_scenes"]
