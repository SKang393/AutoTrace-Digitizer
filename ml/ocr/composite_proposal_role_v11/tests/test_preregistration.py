# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.composite_proposal_role_v11.model import CompositeProposalRoleNet
from ml.ocr.composite_proposal_role_v11.protocol import (
    ENCODED_WIDTH,
    EXPERIMENT_BUDGET,
    REVISION,
    ROLE_ORDER,
    SPLITS,
    TASK,
    protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/PROTOCOL.json"
LEDGER_PATH = ROOT / "ml/markers/training-budgets/production-repair-v1.json"
SELECTION_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/SEALED_PUBLIC_TEST_SEAL.json"
GATE_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/gates/sealed-public-v1.json"
CONFIG_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/training/p1.json"
P2_CONFIG_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/training/p2.json"
P1_RESULT_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/P1_RESULT.json"
P2_RESULT_PATH = ROOT / "ml/ocr/composite_proposal_role_v11/P2_RESULT.json"


def test_protocol_file_is_canonical_and_fail_closed() -> None:
    expected = protocol_configuration()
    assert PROTOCOL_PATH.read_bytes() == canonical_json_bytes(expected)
    assert expected["execution_authorized"] is False
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    assert expected["trigger_evidence"]["scene_level_or_fixture_byte_access_for_v11"] is False
    assert expected["trigger_evidence"]["consumed_gate_rerun_authorized"] is False


def test_split_families_and_scopes_are_disjoint_before_pixels_exist() -> None:
    assert len({item.seed_offset for item in SPLITS}) == len(SPLITS)
    assert len({item.renderer_family for item in SPLITS}) == len(SPLITS)
    assert len({item.degradation_family for item in SPLITS}) == len(SPLITS)
    protocol = protocol_configuration()
    assert protocol["split_policy"] == {
        "train_validation_public_family_ids_disjoint": True,
        "sealed_public_truth_hidden_until_one_time_gate": True,
        "predecessor_fixture_bytes_reused": False,
        "v2_scene_bytes_or_truth_reused": False,
        "validation_or_public_pixels_used_for_training": False,
    }
    assert "Generalization" in protocol["data_scope"]
    assert "no Chandler" in protocol["data_scope"]


def test_model_contract_is_exact_and_finite() -> None:
    model = CompositeProposalRoleNet().eval()
    value = torch.zeros((3, 2, 32, ENCODED_WIDTH), dtype=torch.float32)
    with torch.inference_mode():
        result = model(value)
    assert result.shape == (3, 2 + len(ROLE_ORDER))
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match="2,32,144"):
        model(torch.zeros((1, 2, 32, ENCODED_WIDTH - 1), dtype=torch.float32))


def test_canonical_ledger_consumes_p1_and_p2_then_authorizes_one_public_gate() -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == REVISION)
    assert entry["task"] == TASK
    assert entry["status"] == "candidate_2_selected_public_gate_pending"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P3"]
    assert entry["protocol_sha256"] == sha256_file(PROTOCOL_PATH)
    assert entry["selection_manifest_sha256"] == sha256_file(SELECTION_PATH)
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(SEAL_PATH)
    assert entry["public_gate_config_sha256"] == sha256_file(GATE_PATH)
    assert entry["candidate_config_paths"] == {
        "P1": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "P2": P2_CONFIG_PATH.relative_to(ROOT).as_posix(),
    }
    assert entry["candidate_config_sha256"] == {
        "P1": sha256_file(CONFIG_PATH),
        "P2": sha256_file(P2_CONFIG_PATH),
    }
    assert entry["p1_result_sha256"] == sha256_file(P1_RESULT_PATH)
    assert entry["p1_selection_exact_scene_count"] == entry["p1_selection_scene_count"] == 96
    assert entry["p1_selection_false_positives"] == entry["p1_selection_false_negatives"] == 0
    assert entry["p1_onnx_parity_passed"] is False
    assert entry["p1_public_gate_archive_opened"] is False
    assert entry["p2_result_sha256"] == sha256_file(P2_RESULT_PATH)
    assert entry["p2_selection_exact_scene_count"] == entry["p2_selection_scene_count"] == 96
    assert entry["p2_selection_false_positives"] == entry["p2_selection_false_negatives"] == 0
    assert entry["p2_onnx_parity_passed"] is True
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_authorized_candidate_id"] == "P2"
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_frozen_records_remain_fail_closed_and_exclude_private_data() -> None:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert selection["training_evidence"]["validation_or_public_pixels_used"] is False
    assert selection["training_evidence"]["v2_bytes_used"] is False
    assert selection["sealed_public_truth_available_to_candidate"] is False
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert gate["evaluation_limit"] == 1
    assert config["candidate_id"] == "P1"
    assert config["public_gate_archive_opened"] is False
    for record in (selection, seal, gate, config):
        assert record["production_approval"] is False
        assert record["release_eligible"] is False
    for record in (selection, seal, config):
        assert record["chandler_included"] is False
        assert record["generalization_label_included"] is False
        assert record["private_or_article_images"] is False
        assert record["v2_bytes_used"] is False


def test_p2_repairs_only_absolute_export_parity_without_training() -> None:
    config = json.loads(P2_CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(P1_RESULT_PATH.read_text(encoding="utf-8"))
    assert config["candidate_id"] == "P2"
    assert config["optimizer_steps"] == 0
    assert config["weights_changed"] is False
    assert config["output_scale"] == 0.5
    assert config["p1_checkpoint_sha256"] == result["checkpoint_sha256"]
    assert config["p1_report_sha256"] == result["report_sha256"]
    assert config["p1_result_sha256"] == sha256_file(P1_RESULT_PATH)
    assert result["selection_metrics"]["exact_scene_count"] == result["selection_metrics"]["scene_count"] == 96
    assert result["onnx_parity_passed"] is False
    assert result["public_gate_archive_opened"] is False


def test_p2_selected_result_is_still_not_production_approved() -> None:
    result = json.loads(P2_RESULT_PATH.read_text(encoding="utf-8"))
    assert result["status"] == "selected_public_gate_pending"
    assert result["optimizer_steps"] == 0
    assert result["weights_changed"] is False
    assert result["onnx_parity_passed"] is True
    assert result["selection_metrics"]["exact_scene_count"] == result["selection_metrics"]["scene_count"] == 96
    assert result["selection_metrics"]["false_positives"] == result["selection_metrics"]["false_negatives"] == 0
    assert result["public_gate_evaluations"] == 0
    assert result["public_gate_archive_opened"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
