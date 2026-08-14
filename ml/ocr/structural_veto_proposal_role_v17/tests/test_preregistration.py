# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed design checks for OCR structural-veto V17."""

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.structural_veto_proposal_role_v17.protocol import (
    BASE_CHECKPOINT_SHA256,
    EXPERIMENT_BUDGET,
    SPLITS,
    THRESHOLDS,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
V17_ROOT = ROOT / "ml/ocr/structural_veto_proposal_role_v17"


def test_protocol_is_canonical_fresh_and_execution_blocked() -> None:
    expected = protocol_configuration()
    assert (V17_ROOT / "PROTOCOL.json").read_bytes() == canonical_json_bytes(expected)
    assert expected["state"] == "design_preregistered_before_stored_split_materialization"
    assert expected["currently_preregistered_candidate"] is None
    assert expected["execution_authorized"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    trigger = expected["trigger_evidence"]
    assert trigger["result_sha256"] == TRIGGER_RESULT_SHA256
    assert trigger["case_level_details_used"] is False
    assert trigger["fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert trigger["consumed_v16_candidate_or_gate_rerun_authorized"] is False


def test_trigger_and_base_checkpoint_are_exact_and_distinct() -> None:
    trigger_path = ROOT / protocol_configuration()["trigger_evidence"]["result_path"]
    base_path = ROOT / protocol_configuration()["base_checkpoint"]["path"]
    assert sha256_file(trigger_path) == TRIGGER_RESULT_SHA256
    if base_path.is_file():
        assert sha256_file(base_path) == BASE_CHECKPOINT_SHA256
    base = protocol_configuration()["base_checkpoint"]
    assert base["all_base_parameters_frozen"] is True
    assert base["role_order_and_argmax_preserved"] is True
    veto = protocol_configuration()["veto_branch"]
    assert veto["role_logits_modified"] is False


def test_fresh_splits_and_zero_error_threshold_margin_are_fixed() -> None:
    assert [item.scene_count for item in SPLITS] == [720, 216, 288]
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    expected = protocol_configuration()
    assert expected["selection_thresholds"] == list(THRESHOLDS)
    assert expected["selection_gates"]["minimum_consecutive_passing_thresholds"] == 3
    assert expected["selection_gates"]["false_regions"] == 0
    assert expected["selection_gates"]["missed_regions"] == 0
    assert expected["selection_gates"]["prohibited_structure_hits"] == 0
    assert expected["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert expected["split_policy"]["v16_fixture_bytes_scene_truth_or_case_identity_reused"] is False
    assert expected["training"]["validation_or_public_pixels_used"] is False
    assert "no Chandler" in expected["data_scope"]


def test_canonical_ledger_records_design_only_without_authorization() -> None:
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol_configuration()["revision"])
    assert entry["status"] == "design_preregistered"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["split_materialized"] is False
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False

