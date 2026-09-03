# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Contract checks for the aggregate-only V36 diagnostic."""

import json
from pathlib import Path


def test_saved_report_is_aggregate_only() -> None:
    report_path = Path(__file__).parents[1] / "DIAGNOSTIC.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    encoded = json.dumps(report).lower()
    assert report["evidence"]["split"] == "dev"
    assert report["evidence"]["synthetic_only"] is True
    assert report["evidence"]["sealed_or_public_reads"] == 0
    assert report["evidence"]["case_level_output"] is False
    assert "scene_id" not in encoded
    assert "pixel_output" not in encoded
    assert report["fixed_hashes"]["checkpoint_onnx"]["sha256"]


def test_report_contains_stage_separation_and_fixed_tiling() -> None:
    report = json.loads((Path(__file__).parents[1] / "DIAGNOSTIC.json").read_text(encoding="utf-8"))
    assert report["core_pixel_segmentation"]["threshold_sweep"]
    assert report["predicted_core_proposals"]["threshold_sweep"]
    assert report["expanded_proposals"]["threshold_sweep"]
    assert report["ground_truth_core_expansion_oracle"]["truth_regions"] == 86
    assert report["tiling_overlap_mapping"]["covered_pixel_fraction"] == 1.0
    assert report["isolated_responsible_stage"] == "core_pixel_segmentation"
    assert report["next_revision_startable"] is True
    assert report["ground_truth_core_expansion_oracle"]["recall"] == 1.0
    assert "not the responsible stage" in report["interpretation"]["core_expansion"]
