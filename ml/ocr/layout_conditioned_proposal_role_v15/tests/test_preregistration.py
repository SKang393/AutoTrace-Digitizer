# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed design preregistration checks for OCR V15."""

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.layout_conditioned_proposal_role_v15.protocol import (
    BASE_GEOMETRY_FEATURE_COUNT,
    ENCODED_WIDTH,
    EXPERIMENT_BUDGET,
    GEOMETRY_FEATURE_COUNT,
    PLOT_GEOMETRY_FEATURE_COUNT,
    PLOT_GEOMETRY_ORDER,
    SPLITS,
    protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
V15_ROOT = ROOT / "ml/ocr/layout_conditioned_proposal_role_v15"


def test_protocol_is_canonical_fail_closed_and_aggregate_only() -> None:
    expected = protocol_configuration()
    assert (V15_ROOT / "PROTOCOL.json").read_bytes() == canonical_json_bytes(expected)
    assert expected["state"] == "design_preregistered_before_stored_split_materialization"
    assert expected["currently_preregistered_candidate"] is None
    assert expected["execution_authorized"] is False
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    trigger = expected["trigger_evidence"]
    assert trigger["report_sha256"] == "89b66d006166a8a2efb770c029e0a3f9dc76a5ed0425a2ee16a9f0900da262a4"
    assert trigger["evidence_scope_used_for_v15_design"] == "aggregate metrics only"
    assert trigger["v14_validation_fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert trigger["case_level_details_emitted"] is False
    assert trigger["consumed_v14_candidate_rerun_authorized"] is False


def test_plot_geometry_contract_is_exact_and_production_bound() -> None:
    expected = protocol_configuration()
    assert BASE_GEOMETRY_FEATURE_COUNT == 16
    assert PLOT_GEOMETRY_FEATURE_COUNT == len(PLOT_GEOMETRY_ORDER) == 8
    assert GEOMETRY_FEATURE_COUNT == 24
    assert ENCODED_WIDTH == 152
    assert expected["input"] == ["proposal_count", 2, 32, 152]
    proposal = expected["proposal_algorithm"]
    assert proposal["plot_geometry_source"] == "verified-axis-stage-plot-bounds-v1"
    assert proposal["plot_geometry_order"] == list(PLOT_GEOMETRY_ORDER)
    assert proposal["component_grouping_unchanged_from_production"] is True


def test_split_registrations_are_fresh_disjoint_and_unmaterialized() -> None:
    assert [item.scene_count for item in SPLITS] == [512, 160, 224]
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    protocol = protocol_configuration()
    assert protocol["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert protocol["split_policy"]["v14_validation_fixture_bytes_scene_truth_or_case_identity_reused"] is False
    assert protocol["split_policy"]["public_case_level_failure_analysis_permitted"] is False
    assert "no Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]
    for relative in (
        "SELECTION_MANIFEST.json", "SEALED_PUBLIC_TEST_SEAL.json",
        "gates/sealed-public-v1.json", "training/p1.json",
    ):
        assert not (V15_ROOT / relative).exists()


def test_canonical_ledger_registers_design_but_authorizes_nothing() -> None:
    ledger = json.loads(
        (ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item.get("revision") == "graph-text-layout-conditioned-proposal-role-v15"
    )
    assert entry["status"] == "design_preregistered"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["protocol_sha256"] == sha256_file(V15_ROOT / "PROTOCOL.json")
    assert entry["split_materialized"] is False
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
