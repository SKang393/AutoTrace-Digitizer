# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V12."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import json

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.full_scene_proposal_role_v12.model import FullSceneProposalRoleNet
from ml.ocr.full_scene_proposal_role_v12.protocol import (
    ENCODED_WIDTH, EXPERIMENT_BUDGET, ROLE_ORDER, SPLITS, protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = ROOT / "ml/ocr/full_scene_proposal_role_v12/PROTOCOL.json"
LEDGER_PATH = ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def test_protocol_file_is_canonical_and_fail_closed() -> None:
    expected = protocol_configuration()
    assert PROTOCOL_PATH.read_bytes() == canonical_json_bytes(expected)
    assert expected["execution_authorized"] is False
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    assert expected["trigger_evidence"]["evidence_scope_used_for_v12_design"] == "aggregate metrics only"
    assert expected["trigger_evidence"]["v3_fixture_bytes_or_scene_truth_used"] is False
    assert expected["trigger_evidence"]["consumed_gate_rerun_authorized"] is False


def test_split_families_are_disjoint_before_pixels_exist() -> None:
    assert len({item.seed_offset for item in SPLITS}) == len(SPLITS)
    assert len({item.renderer_family for item in SPLITS}) == len(SPLITS)
    assert len({item.degradation_family for item in SPLITS}) == len(SPLITS)
    expected = protocol_configuration()
    assert expected["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert expected["split_policy"]["v3_fixture_bytes_or_scene_truth_reused"] is False
    assert expected["split_policy"]["validation_or_public_pixels_used_for_training"] is False
    assert "no Chandler" in expected["data_scope"]
    assert "Generalization" in expected["data_scope"]


def test_model_contract_is_exact_finite_and_geometry_gated() -> None:
    model = FullSceneProposalRoleNet().eval()
    value = torch.zeros((3, 2, 32, ENCODED_WIDTH), dtype=torch.float32)
    with torch.inference_mode():
        result = model(value)
    assert result.shape == (3, 2 + len(ROLE_ORDER))
    assert torch.isfinite(result).all()
    assert hasattr(model, "visual_gate")
    with pytest.raises(ValueError, match="2,32,144"):
        model(torch.zeros((1, 2, 32, ENCODED_WIDTH - 1), dtype=torch.float32))


def test_split_and_candidate_records_are_frozen_but_execution_is_closed() -> None:
    root = ROOT / "ml/ocr/full_scene_proposal_role_v12"
    for relative in ("SELECTION_MANIFEST.json", "SEALED_PUBLIC_TEST_SEAL.json", "gates/sealed-public-v1.json", "training/p1.json"):
        assert (root / relative).is_file()
    selection = json.loads((root / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    seal = json.loads((root / "SEALED_PUBLIC_TEST_SEAL.json").read_text(encoding="utf-8"))
    config = json.loads((root / "training/p1.json").read_text(encoding="utf-8"))
    assert selection["sealed_public_truth_available_to_candidate"] is False
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert config["public_gate_archive_opened"] is False
    assert (root / "P1_RESULT.json").is_file()
    assert not (root / "artifacts/P1-run/public-report.json").exists()
    for record in (selection, seal, config):
        assert record["production_approval"] is False
        assert record["release_eligible"] is False


def test_canonical_ledger_consumes_failed_p1_and_authorizes_nothing() -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == "graph-text-full-scene-proposal-role-v12")
    assert entry["status"] == "candidate_1_failed_selection"
    assert entry["protocol_sha256"] == sha256_file(PROTOCOL_PATH)
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    result_path = ROOT / entry["p1_result_path"]
    assert entry["p1_result_sha256"] == sha256_file(result_path)
    assert entry["p1_selection_exact_scene_count"] == 119
    assert entry["p1_selection_scene_count"] == 120
    assert entry["p1_selection_false_positives"] == 1
    assert entry["p1_selection_false_negatives"] == 0
    assert entry["p1_selection_role_accuracy"] == 1.0
    assert entry["p1_onnx_parity_passed"] is False
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
