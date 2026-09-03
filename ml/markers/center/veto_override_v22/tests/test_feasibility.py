# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

from ml.markers.center.veto_override_v22.diagnose_v22 import summarize


ROOT = Path(__file__).parents[5]
ARTIFACTS = ROOT / "artifacts/goal22-worktrees/marker-v21/ml/markers/center/focal_confidence_v21/artifacts/P1-run"


def test_v22_sweep_is_aggregate_only_and_has_no_startable_floor() -> None:
    result = summarize(
        ARTIFACTS / "marker-center-focal-confidence-v21-p1.onnx",
    )
    assert result["status"] == "failed_feasibility_no_candidate"
    assert result["passing_floors"] == []
    assert result["feasibility"]["startable"] is False
    assert result["candidate"] is None
    assert result["scope"]["fixed_dev_manifest_sha256"] == "771141bd5d22d2ffdb56e63f193c16c0775a0f47c02313ef67b06bbce9603126"
    assert result["scope"]["synthetic_only"] is True
    assert result["scope"]["training_performed"] is False
    assert result["scope"]["optimizer_steps"] == 0
    assert result["scope"]["scene_ids_emitted"] is False
    assert result["scope"]["case_level_details_emitted"] is False


def test_v22_floor_metrics_and_bindings_are_fixed() -> None:
    result = summarize(
        ARTIFACTS / "marker-center-focal-confidence-v21-p1.onnx",
    )
    floors = result["override_floors"]
    assert set(floors) == {"0.9", "0.95", "0.99", "0.995", "0.999"}
    assert floors["0.9"]["true_positives"] == 89
    assert floors["0.9"]["false_positives"] == 0
    assert floors["0.9"]["false_negatives"] == 7
    assert floors["0.9"]["bypassed_candidates"] == 1
    assert floors["0.9"]["prohibited_structure_hits"] == 0
    assert floors["0.9"]["clears_both_bars"] is False
    for floor in ("0.95", "0.99", "0.995", "0.999"):
        assert floors[floor]["bypassed_candidates"] == 0
        assert floors[floor]["false_negatives"] == 8
        assert floors[floor]["prohibited_structure_hits"] == 0
    assert result["binding"]["v21_result_sha256"] == "a78710ee13da02ec45c26a216524a90b261b0fbc17017491d525a59fe9ecdacb"
    assert result["binding"]["v21_diagnostic_sha256"] == "41a7aea0432f11891c5d1c0c641ff637ccea93e073d007ea37044bf89c4b5785"
    assert result["binding"]["v21_onnx_sha256"] == "0b413db48f8e6707ee5ec99afff4cd8ec3d25c6b8a8d9f165bd416deb4578a38"


def test_tracked_diagnostic_is_utf8_lf_and_contains_no_case_detail() -> None:
    path = Path(__file__).parents[1] / "V22_FEASIBILITY_DIAGNOSTIC.json"
    data = path.read_bytes()
    assert b"\r" not in data
    result = json.loads(data.decode("utf-8"))
    assert result["schema"] == "graphreader.marker-center-veto-override-v22-feasibility.v1"
    strings = json.dumps(result, sort_keys=True)
    assert "scene_ids" not in strings or result["scope"]["scene_ids_emitted"] is False
    assert "truth_rows" not in strings
    assert result["candidate"] is None
