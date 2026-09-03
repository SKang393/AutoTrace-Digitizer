# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

from ml.markers.center.tail_coverage_v20.diagnostics.diagnose_v20 import summarize


ROOT = Path(__file__).parents[5]
ARTIFACTS = ROOT / "artifacts/goal22-worktrees/marker-v20/ml/markers/center/tail_coverage_v20/artifacts/P1-run"


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


def test_v20_diagnostic_reproduces_retired_dev_totals_without_case_ids() -> None:
    result = summarize(ARTIFACTS / "marker-center-tail-coverage-v20-p1.onnx", ARTIFACTS / "marker-center-tail-coverage-v20-p1.pt")
    selected = result["thresholds"]["0.25"]
    assert selected["true_positives"] == 83
    assert selected["false_positives"] == 0
    assert selected["false_negatives"] == 13
    assert result["miss_stage"]["total_misses"] == 13
    assert result["scope"]["scene_ids_emitted"] is False
    assert result["scope"]["case_level_details_emitted"] is False
    strings = list(_strings(result))
    assert not any(value.startswith("dev-") for value in strings)
    assert not any(key in result for key in ("scene_ids", "case_ids", "truth_rows", "predictions"))


def test_v20_diagnostic_covers_proposals_shape_radius_and_stage_residual() -> None:
    result = summarize(ARTIFACTS / "marker-center-tail-coverage-v20-p1.onnx", ARTIFACTS / "marker-center-tail-coverage-v20-p1.pt")
    assert result["inference"]["truth_count"] == 96
    assert result["inference"]["proposal_funnel"] == {"truth_count": 96, "raw_3px": 96, "geometry_3px": 96, "geometry_5px": 96}
    assert sum(row["truth_count"] for row in result["dimensions"]["marker_shape"].values()) == 96
    assert sum(row["truth_count"] for row in result["dimensions"]["marker_radius_px"].values()) == 96
    assert result["next_revision"]["startable"] is True


def test_tracked_aggregate_is_utf8_lf_and_bound_to_v20_payloads() -> None:
    path = Path(__file__).with_name("V20_DIAGNOSTIC.json")
    data = path.read_bytes()
    assert b"\r" not in data
    result = json.loads(data.decode("utf-8"))
    assert result["schema"] == "graphreader.marker-center-tail-coverage-v20-diagnostic.v1"
    assert result["artifacts"]["onnx"]["sha256"] == "862ad1f4c3be53714ab9ccd2e745783f778c0f194846f1dfe85f9fbaf5a4701b"
    assert result["artifacts"]["checkpoint"]["sha256"] == "f2caa70861e28276f9ea5f692020057831752f6f3d8d6f866b3ed27a891cb0df"
    assert result["miss_stage"]["total_misses"] == 13
