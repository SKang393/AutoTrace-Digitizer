# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

from ml.markers.center.train_family_v19.diagnostics.diagnose_v19 import summarize


ROOT = Path(__file__).parents[5]
ONNX = ROOT / "artifacts/goal22-worktrees/marker-v19/ml/markers/center/train_family_v19/artifacts/P1-run/marker-center-train-family-v19-p1.onnx"


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_v19_diagnostic_reproduces_retired_dev_totals_without_case_ids() -> None:
    result = summarize(ONNX)
    selected = result["thresholds"]["0.25"]
    assert selected["true_positives"] == 76
    assert selected["false_positives"] == 0
    assert selected["false_negatives"] == 20
    assert result["miss_stage"]["total_misses"] == 20
    assert result["miss_stage"]["counts"]["confidence_below_threshold"] == 19
    assert result["miss_stage"]["counts"]["marker_geometry_veto"] == 1
    assert result["scope"]["scene_ids_emitted"] is False
    assert result["scope"]["case_level_details_emitted"] is False
    strings = list(_strings(result))
    assert not any(value.startswith("dev-") for value in strings)
    assert not any(key in result for key in ("scene_ids", "case_ids", "truth_rows", "predictions"))


def test_v19_diagnostic_proves_proposal_coverage_and_dimensions() -> None:
    result = summarize(ONNX)
    assert result["inference"]["truth_count"] == 96
    assert result["dimensions"]["family"]["geometry_wide_dev"]["truth_with_geometry_proposal_3px"] == 32
    assert result["dimensions"]["family"]["geometry_mixed_dev"]["truth_with_geometry_proposal_3px"] == 32
    assert result["dimensions"]["family"]["geometry_intersection_dev"]["truth_with_geometry_proposal_3px"] == 32
    assert sum(item["truth_count"] for item in result["dimensions"]["marker_radius_px"].values()) == 96
    assert sum(item["truth_count"] for item in result["dimensions"]["marker_geometry"].values()) == 96
    assert result["next_revision"]["startable"] is True


def test_tracked_aggregate_has_expected_artifact_binding() -> None:
    path = Path(__file__).with_name("V19_DIAGNOSTIC.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["schema"] == "graphreader.marker-center-train-family-v19-diagnostic.v1"
    assert result["artifacts"]["onnx"]["sha256"] == "4d479cd6d7ecb910bd4ebfa4c0063c2d611609489fdd45f90601b0ee581f8ec3"
    assert result["inference"]["truth_count"] == 96
    assert result["miss_stage"]["total_misses"] == 20
