# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.markers import training_budget
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


def test_canonical_ledger_records_design_without_execution_authority() -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == REVISION)
    assert entry["task"] == TASK
    assert entry["status"] == "design_preregistered_split_pending"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["protocol_sha256"] == sha256_file(PROTOCOL_PATH)
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_training_refuses_before_split_and_candidate_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_budget, "require_committed_sources", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="design_preregistered_split_pending"):
        training_budget.require_training_budget(ROOT, task=TASK, revision=REVISION)
