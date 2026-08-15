# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V18."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.recognition_confirmed_proposal_role_v18.evaluate_candidate import RUNNER_SOURCE_PATHS
from ml.ocr.recognition_confirmed_proposal_role_v18.pipeline import decode_with_confidence
from ml.ocr.recognition_confirmed_proposal_role_v18.protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    EXPERIMENT_BUDGET,
    FEASIBILITY_PATH,
    FEASIBILITY_SHA256,
    RECOGNITION_CONFIDENCE_THRESHOLD,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    SPLITS,
    protocol_configuration,
)
from ml.ocr.recognition_confirmed_proposal_role_v18.sealed_gate import EVALUATOR_SOURCE_PATHS


ROOT = Path(__file__).resolve().parents[4]
V18_ROOT = ROOT / "ml/ocr/recognition_confirmed_proposal_role_v18"


def test_protocol_is_canonical_fresh_and_execution_blocked() -> None:
    expected = protocol_configuration()
    assert (V18_ROOT / "PROTOCOL.json").read_bytes() == canonical_json_bytes(expected)
    assert expected["state"] == "design_preregistered_before_stored_split_materialization"
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 1
    assert expected["candidate_ids"] == ["P1"]
    assert expected["currently_preregistered_candidate"] is None
    assert expected["execution_authorized"] is False
    assert expected["optimizer_steps"] == 0
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    assert expected["marker_creation_gate_required_before_approval"] is True


def test_exact_payloads_not_predecessor_fixtures_are_bound() -> None:
    expected = protocol_configuration()
    assert sha256_file(ROOT / FEASIBILITY_PATH) == FEASIBILITY_SHA256
    if (ROOT / DETECTOR_PATH).is_file():
        assert sha256_file(ROOT / DETECTOR_PATH) == DETECTOR_SHA256
    if (ROOT / RECOGNIZER_PATH).is_file():
        assert sha256_file(ROOT / RECOGNIZER_PATH) == RECOGNIZER_SHA256
    assert expected["feasibility_evidence"]["case_level_details_used"] is False
    assert expected["feasibility_evidence"]["fixture_bytes_or_case_identity_reused_by_v18"] is False
    assert expected["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert expected["split_policy"]["v17_fixture_bytes_scene_truth_or_case_identity_reused"] is False
    assert "no Chandler" in expected["data_scope"]


def test_fresh_stored_splits_and_gate_remain_closed() -> None:
    assert [item.scene_count for item in SPLITS] == [192, 256]
    assert len({item.seed_offset for item in SPLITS}) == 2
    assert len({item.renderer_family for item in SPLITS}) == 2
    assert len({item.degradation_family for item in SPLITS}) == 2
    selection = json.loads((V18_ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    seal = json.loads((V18_ROOT / "SEALED_PUBLIC_TEST_SEAL.json").read_text(encoding="utf-8"))
    gate = json.loads((V18_ROOT / "gates/sealed-public-v1.json").read_text(encoding="utf-8"))
    config = json.loads((V18_ROOT / "evaluation/p1.json").read_text(encoding="utf-8"))
    assert selection["validation"]["scene_count"] == 192
    assert selection["validation"]["truth_region_count"] == 1536
    assert selection["sealed_public"]["scene_count"] == 256
    assert selection["sealed_public"]["truth_region_count"] == 2048
    for scope in ("validation", "sealed_public"):
        item = selection[scope]
        assert sha256_file(ROOT / item["fixture_archive_path"]) == item["fixture_archive_sha256"]
        assert sha256_file(ROOT / item["private_manifest_path"]) == item["private_manifest_sha256"]
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert gate["evaluation_limit"] == 1
    assert config["optimizer_steps"] == 0
    assert config["public_gate_archive_opened"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["marker_creation_evaluated"] is False
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS) == gate["expected_evaluator_source_bundle_sha256"]
    candidate_report = V18_ROOT / "artifacts/P1-run/candidate-report.json"
    if candidate_report.is_file():
        assert sha256_file(candidate_report) == sha256_file(V18_ROOT / "P1_RESULT.json")
    assert not (V18_ROOT / "artifacts/public-gate-report.json").exists()


def test_mean_collapsed_ctc_confidence_is_fixed_and_rejects_blank() -> None:
    alphabet = "ab"
    output = np.asarray([[
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.80, 0.15],
        [0.05, 0.05, 0.90],
    ]], dtype=np.float32)
    prediction = decode_with_confidence(output, alphabet)[0]
    assert prediction.text == "ab"
    assert np.isclose(prediction.confidence, 0.90)
    blank = decode_with_confidence(np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32), alphabet)[0]
    assert blank.text == ""
    assert blank.confidence == 0.0
    assert RECOGNITION_CONFIDENCE_THRESHOLD == 0.60


def test_canonical_ledger_records_consumed_failed_p1_and_keeps_public_closed() -> None:
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(
        item for item in ledger["revisions"]
        if item["revision"] == protocol_configuration()["revision"]
    )
    assert entry["status"] == "exhausted_failed_selection"
    assert entry["experiment_budget"] == 1
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["marker_creation_evaluated"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    result_path = V18_ROOT / "P1_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert sha256_file(result_path) == entry["p1_result_sha256"]
    assert result["status"] == "failed_selection"
    assert result["selection_gate_passed"] is False
    assert result["metrics"]["exact_scene_count"] == 189
    assert result["metrics"]["false_positives"] == 1
    assert result["metrics"]["false_negatives"] == 3
    assert result["metrics"]["duplicate_region_count"] == 0
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
