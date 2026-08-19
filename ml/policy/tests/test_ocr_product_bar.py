# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json

from ml.policy.rescore_ocr_product_bar import PROTOCOL_PATH, RESULT_PATH, rescore


def test_recorded_aggregates_match_tracked_product_bar_result() -> None:
    assert rescore() == json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_product_bar_uses_only_requested_selected_threshold_gates() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert set(protocol["acceptance_bar"]) == {
        "scene_exact_rate_minimum",
        "character_error_rate_maximum",
        "role_accuracy_minimum",
        "prohibited_structure_hits_maximum",
    }
    assert protocol["operating_point"] == {
        "selected_threshold_only": True,
        "threshold_sensitivity_is_descriptive_only": True,
        "consecutive_threshold_rule": False,
        "margin_gate": False,
        "robustness_gate": False,
    }


def test_v30_is_candidate_approved_and_v31_is_dev_corroboration() -> None:
    result = rescore()
    by_revision = {item["revision"]: item for item in result["candidates"]}

    assert result["status"] == "pass"
    assert result["synthetic_candidate_approval"] is True
    assert result["production_approval"] is False
    assert by_revision["graph-text-unanimous-structure-veto-v30"]["product_bar_passed"] is True
    assert by_revision["graph-text-unanimous-structure-veto-v30"]["evidence_split"] == "sealed"
    assert by_revision["graph-text-robust-quorum-recall-v31"]["product_bar_passed"] is True
    assert by_revision["graph-text-robust-quorum-recall-v31"]["evidence_split"] == "dev"


def test_rescore_reads_no_archives_or_case_level_material() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(protocol).lower()

    assert ".zip" not in serialized
    assert "fixture_archive_path" not in serialized
    assert protocol["budget"] == {
        "model_training_runs": 0,
        "model_inference_runs": 0,
        "sealed_split_reads": 0,
        "case_level_reads": 0,
    }
