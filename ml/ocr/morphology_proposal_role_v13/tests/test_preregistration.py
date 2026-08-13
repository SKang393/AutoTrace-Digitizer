# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V13."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.morphology_proposal_role_v13.dataset import render_scene, proposals
from ml.ocr.morphology_proposal_role_v13.model import MorphologyProposalRoleNet
from ml.ocr.morphology_proposal_role_v13.protocol import (
    ENCODED_WIDTH, EXPERIMENT_BUDGET, ROLE_ORDER, SPLITS, protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = ROOT / "ml/ocr/morphology_proposal_role_v13/PROTOCOL.json"
V13_ROOT = ROOT / "ml/ocr/morphology_proposal_role_v13"


def test_protocol_is_canonical_fail_closed_and_aggregate_only() -> None:
    expected = protocol_configuration()
    assert PROTOCOL_PATH.read_bytes() == canonical_json_bytes(expected)
    assert expected["execution_authorized"] is False
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    evidence = expected["trigger_evidence"]
    assert evidence["evidence_scope_used_for_v13_design"] == "aggregate metrics only"
    assert evidence["v12_public_fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert evidence["consumed_gate_rerun_authorized"] is False


def test_split_registrations_are_fresh_and_disjoint() -> None:
    assert [item.scene_count for item in SPLITS] == [600, 160, 224]
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    protocol = protocol_configuration()
    assert protocol["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert protocol["split_policy"]["public_case_level_failure_analysis_permitted"] is False
    assert "no Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]


def test_model_contract_is_exact_finite_and_anisotropic() -> None:
    model = MorphologyProposalRoleNet().eval()
    value = torch.zeros((3, 2, 32, ENCODED_WIDTH), dtype=torch.float32)
    with torch.inference_mode():
        result = model(value)
    assert result.shape == (3, 2 + len(ROLE_ORDER))
    assert torch.isfinite(result).all()
    assert hasattr(model, "horizontal_morphology")
    assert hasattr(model, "vertical_morphology")
    assert hasattr(model, "mixture_gate")
    with pytest.raises(ValueError, match="2,32,144"):
        model(torch.zeros((1, 2, 32, ENCODED_WIDTH - 1), dtype=torch.float32))


@pytest.mark.parametrize("split,index", [
    ("train", 0), ("train", 197), ("train", 599),
    ("validation", 0), ("validation", 159),
    ("sealed_public", 0), ("sealed_public", 223),
])
def test_preregistered_source_renderer_has_one_proposal_per_truth(split: str, index: int) -> None:
    scene = render_scene(split, index)
    candidates = proposals(scene.raster)
    assert len(scene.truths) == len(ROLE_ORDER)
    for truth in scene.truths:
        assert sum(box_iou(candidate.box, truth.box) >= 0.5 for candidate in candidates) == 1


def test_design_checkpoint_contains_no_split_or_candidate_artifacts() -> None:
    for relative in (
        "SELECTION_MANIFEST.json", "SEALED_PUBLIC_TEST_SEAL.json",
        "gates/sealed-public-v1.json", "training/p1.json", "P1_RESULT.json",
    ):
        assert not (V13_ROOT / relative).exists()
    assert not (V13_ROOT / "artifacts").exists()


def test_no_v13_budget_ledger_entry_exists_before_runner_and_split_freeze() -> None:
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    assert not any(item.get("revision") == "graph-text-morphology-proposal-role-v13" for item in ledger["revisions"])
