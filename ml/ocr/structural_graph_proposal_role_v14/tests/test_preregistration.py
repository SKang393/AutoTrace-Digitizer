# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V14."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.structural_graph_proposal_role_v14.dataset import render_scene, proposals
from ml.ocr.structural_graph_proposal_role_v14.model import StructuralGraphProposalRoleNet
from ml.ocr.structural_graph_proposal_role_v14.protocol import (
    ENCODED_WIDTH, EXPERIMENT_BUDGET, ROLE_ORDER, SPLITS, protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
V14_ROOT = ROOT / "ml/ocr/structural_graph_proposal_role_v14"
PROTOCOL_PATH = V14_ROOT / "PROTOCOL.json"


def test_protocol_is_canonical_fail_closed_and_aggregate_only() -> None:
    expected = protocol_configuration()
    assert PROTOCOL_PATH.read_bytes() == canonical_json_bytes(expected)
    assert expected["execution_authorized"] is False
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    evidence = expected["trigger_evidence"]
    assert evidence["evidence_scope_used_for_v14_design"] == "aggregate metrics only"
    assert evidence["v13_public_fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert evidence["case_level_details_emitted"] is False
    assert evidence["consumed_gate_rerun_authorized"] is False


def test_split_registrations_are_fresh_and_disjoint() -> None:
    assert [item.scene_count for item in SPLITS] == [480, 144, 208]
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    protocol = protocol_configuration()
    assert protocol["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert protocol["split_policy"]["public_case_level_failure_analysis_permitted"] is False
    assert "no Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]
    assert protocol["architecture"] == "dual-context-topology-spectrum-residual-proposal-role-cnn-v1"


def test_model_contract_is_exact_finite_and_topology_aware() -> None:
    model = StructuralGraphProposalRoleNet().eval()
    value = torch.zeros((3, 2, 32, ENCODED_WIDTH), dtype=torch.float32)
    with torch.inference_mode():
        result = model(value)
    assert result.shape == (3, 2 + len(ROLE_ORDER))
    assert torch.isfinite(result).all()
    assert hasattr(model, "topology")
    assert hasattr(model, "occupancy_spectrum")
    assert hasattr(model, "proposal_residual")
    assert not hasattr(model, "horizontal_morphology")
    with pytest.raises(ValueError, match="2,32,144"):
        model(torch.zeros((1, 2, 32, ENCODED_WIDTH - 1), dtype=torch.float32))


@pytest.mark.parametrize("split,index", [
    ("train", 0), ("train", 173), ("train", 479),
    ("validation", 0), ("validation", 143),
    ("sealed_public", 0), ("sealed_public", 207),
])
def test_source_renderer_has_one_production_proposal_per_truth(split: str, index: int) -> None:
    scene = render_scene(split, index)
    candidates = proposals(scene.raster)
    assert len(scene.truths) == len(ROLE_ORDER)
    for truth in scene.truths:
        assert sum(box_iou(candidate.box, truth.box) >= 0.5 for candidate in candidates) == 1


def test_execution_records_do_not_exist_before_freeze() -> None:
    for relative in (
        "SELECTION_MANIFEST.json", "SEALED_PUBLIC_TEST_SEAL.json",
        "gates/sealed-public-v1.json", "training/p1.json", "P1_RESULT.json",
    ):
        assert not (V14_ROOT / relative).exists()
    assert not (V14_ROOT / "artifacts").exists()


def test_v13_consumed_public_identity_is_trigger_only() -> None:
    protocol = protocol_configuration()
    trigger = protocol["trigger_evidence"]
    assert trigger["report_sha256"] == "fe0a0d63d0c00c35c463e45c11a87f97aa81bb625d0744c8c2307537b6d6d2ff"
    assert trigger["scene_count"] == 224
    assert trigger["exact_scene_count"] == 223
    assert trigger["false_regions"] == trigger["prohibited_structure_hits"] == 1
    assert trigger["missed_regions"] == trigger["duplicate_regions"] == 0
    assert trigger["role_accuracy"] == 1.0
