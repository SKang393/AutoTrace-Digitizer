# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Focused contract checks for the aggregate-only V38 diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.ocr.dice_loss_detector_v38.diagnostics.diagnose import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_SHA256,
    _pixel_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "DIAGNOSTIC.json"


def test_pixel_metric_helper_is_deterministic() -> None:
    probability = np.asarray([[0.1, 0.8], [0.9, 0.2]], dtype=np.float32)
    truth = np.asarray([[False, True], [True, False]])
    result = _pixel_metrics(probability, truth, 0.5)
    assert result["true_positive_pixels"] == 2
    assert result["false_positive_pixels"] == 0
    assert result["false_negative_pixels"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_report_is_aggregate_only_and_actionable() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == "graphreader.ocr-dice-loss-detector-v38-diagnostic.v1"
    assert report["evidence"] == {
        "case_level_output": False,
        "private_or_article_images": False,
        "sealed_or_public_reads": 0,
        "scene_count": 5,
        "split": "dev",
        "synthetic_only": True,
        "truth_region_count": 86,
    }
    assert report["fixed_hashes"]["v38_onnx"]["sha256"] == EXPECTED_MODEL_SHA256
    assert report["fixed_hashes"]["v38_checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert report["interpretation"]["isolated_responsible_stage"] == "full_box_pixel_segmentation"
    assert report["interpretation"]["next_revision_startable"] is True
    assert len(report["pixel_segmentation"]["threshold_sweep"]) == 19
    assert len(report["postprocessing"]["threshold_and_morphology_sweep"]) == 38
    assert len(report["by_dimension"]) == 5


def test_report_does_not_expose_case_or_pixel_payloads() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    def walk(value: object) -> list[str]:
        if isinstance(value, dict):
            result: list[str] = []
            for key, item in value.items():
                result.append(str(key).lower())
                result.extend(walk(item))
            return result
        if isinstance(value, list):
            result = []
            for item in value:
                result.extend(walk(item))
            return result
        return []

    keys = walk(report)
    assert "scene_id" not in keys
    assert "truths" not in keys
    assert "raster" not in keys
    assert "pixels" not in keys
    assert "predictions" not in keys
