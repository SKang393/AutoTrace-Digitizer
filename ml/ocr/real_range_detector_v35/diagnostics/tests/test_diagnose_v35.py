# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Contract checks for the aggregate-only V35 diagnostic."""

import json
from pathlib import Path


def test_saved_report_is_aggregate_only() -> None:
    report_path = Path(__file__).parents[1] / "DIAGNOSTIC.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evidence"]["split"] == "dev"
    assert report["evidence"]["synthetic_only"] is True
    assert report["evidence"]["sealed_or_public_reads"] == 0
    assert "scene_id" not in json.dumps(report)
    assert report["baseline_fixed_pipeline"]["precision"] == 0.20512820512820512
    assert report["baseline_fixed_pipeline"]["recall"] == 0.46511627906976744


def test_diagnostic_identifies_segmentation() -> None:
    report = json.loads((Path(__file__).parents[1] / "DIAGNOSTIC.json").read_text(encoding="utf-8"))
    assert report["isolated_responsible_stage"] == "pixel_segmentation"
    assert report["tiling_overlap_mapping"]["covered_pixel_fraction"] == 1.0
