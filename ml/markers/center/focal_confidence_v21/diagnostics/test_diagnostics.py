# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

from ml.markers.center.focal_confidence_v21.diagnostics.diagnose_v21 import summarize


ROOT = Path(__file__).parents[5]
ARTIFACTS = ROOT / "artifacts/goal22-worktrees/marker-v21/ml/markers/center/focal_confidence_v21/artifacts/P1-run"


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


def test_v21_diagnostic_reproduces_retired_dev_totals_without_case_ids() -> None:
    result = summarize(
        ARTIFACTS / "marker-center-focal-confidence-v21-p1.onnx",
        ARTIFACTS / "marker-center-focal-confidence-v21-p1.pt",
    )
    selected = result["thresholds"]["0.25"]
    assert selected["true_positives"] == 88
    assert selected["false_positives"] == 0
    assert selected["false_negatives"] == 8
    assert result["miss_stage"]["total_misses"] == 8
    assert result["scope"]["scene_ids_emitted"] is False
    assert result["scope"]["case_level_details_emitted"] is False
    strings = list(_strings(result))
    assert not any(value.startswith("dev-") for value in strings)
    assert not any(key in result for key in ("scene_ids", "case_ids", "truth_rows", "predictions"))


def test_v21_diagnostic_covers_all_requested_stage_dimensions() -> None:
    result = summarize(
        ARTIFACTS / "marker-center-focal-confidence-v21-p1.onnx",
        ARTIFACTS / "marker-center-focal-confidence-v21-p1.pt",
    )
    assert result["inference"]["truth_count"] == 96
    assert result["inference"]["proposal_funnel"] == {"truth_count": 96, "raw_3px": 96, "geometry_3px": 96, "geometry_5px": 96}
    assert result["miss_stage"]["counts"] == {
        "confidence_below_threshold": 1,
        "marker_geometry_veto": 5,
        "unmasked_artifact_veto": 2,
    }
    assert sum(row["truth_count"] for row in result["dimensions"]["family"].values()) == 96
    assert sum(row["truth_count"] for row in result["dimensions"]["marker_shape"].values()) == 96
    assert sum(row["truth_count"] for row in result["dimensions"]["marker_radius_px"].values()) == 96
    assert set(result["threshold_sensitivity"]) == {"0.0", "0.05", "0.1", "0.15", "0.25", "0.4", "0.55", "0.7"}
    assert result["next_revision"]["startable"] is True


def test_tracked_aggregate_is_utf8_lf_and_bound_to_v21_payloads() -> None:
    path = Path(__file__).with_name("V21_DIAGNOSTIC.json")
    data = path.read_bytes()
    assert b"\r" not in data
    result = json.loads(data.decode("utf-8"))
    assert result["schema"] == "graphreader.marker-center-focal-confidence-v21-diagnostic.v1"
    assert result["artifacts"]["onnx"]["sha256"] == "0b413db48f8e6707ee5ec99afff4cd8ec3d25c6b8a8d9f165bd416deb4578a38"
    assert result["artifacts"]["checkpoint"]["sha256"] == "ba9722ebd3091c91749c175607a480500d3b651e6719d19b69c7e48cee4ef6c9"
    assert result["miss_stage"]["total_misses"] == 8
