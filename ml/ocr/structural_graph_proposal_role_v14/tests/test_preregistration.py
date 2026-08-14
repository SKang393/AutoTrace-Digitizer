# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V14."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.structural_graph_proposal_role_v14.dataset import render_scene, proposals
from ml.ocr.structural_graph_proposal_role_v14.model import StructuralGraphProposalRoleNet
from ml.ocr.structural_graph_proposal_role_v14.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.structural_graph_proposal_role_v14.train_p1 import RUNNER_SOURCE_PATHS
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


def test_split_and_candidate_records_are_frozen_and_fail_closed() -> None:
    for relative in (
        "SELECTION_MANIFEST.json", "SEALED_PUBLIC_TEST_SEAL.json",
        "gates/sealed-public-v1.json", "training/p1.json",
    ):
        assert (V14_ROOT / relative).is_file()
    selection = json.loads((V14_ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    seal = json.loads((V14_ROOT / "SEALED_PUBLIC_TEST_SEAL.json").read_text(encoding="utf-8"))
    config = json.loads((V14_ROOT / "training/p1.json").read_text(encoding="utf-8"))
    assert selection["sealed_public_truth_available_to_candidate"] is False
    assert selection["training_tensor_shape"] == [29518, 2, 32, 144]
    assert selection["train"]["split_fingerprint"] == "29fd2e0274aaaddd3a95b228fd46e06ddcdc2570139538b11a9b878b9e3ba263"
    assert selection["validation"]["split_fingerprint"] == "f1744da5c00ffe6c8f74e3a084c10e93926f05d25ddaebd8f0a9f0eac7593b00"
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert seal["fixture_archive_sha256"] == "57d8dbf2435f1cff415a1ef7641db59de0556c416532f11f6790164e3db66ebf"
    assert seal["split_fingerprint"] == "dbdad5ab4634be014a920350ab39d078609b63f6e827f2fd6df7956f9dd0cadd"
    assert config["public_gate_archive_opened"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)
    assert not (V14_ROOT / "P1_RESULT.json").exists()
    for record in (selection, seal, config):
        assert record["production_approval"] is False
        assert record["release_eligible"] is False


def test_frozen_budget_ledger_still_refuses_p1_execution() -> None:
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == "graph-text-structural-graph-proposal-role-v14")
    assert entry["status"] == "split_frozen_execution_pending"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["selection_manifest_sha256"] == sha256_file(V14_ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(V14_ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(V14_ROOT / "training/p1.json")
    assert entry["p1_expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)
    assert entry["expected_public_evaluator_source_bundle_sha256"] == source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS)
    assert GATE_CONFIG["evaluation_limit"] == 1
    assert GATE_CONFIG["case_level_failure_analysis_permitted"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False


def test_v13_consumed_public_identity_is_trigger_only() -> None:
    protocol = protocol_configuration()
    trigger = protocol["trigger_evidence"]
    assert trigger["report_sha256"] == "fe0a0d63d0c00c35c463e45c11a87f97aa81bb625d0744c8c2307537b6d6d2ff"
    assert trigger["scene_count"] == 224
    assert trigger["exact_scene_count"] == 223
    assert trigger["false_regions"] == trigger["prohibited_structure_hits"] == 1
    assert trigger["missed_regions"] == trigger["duplicate_regions"] == 0
    assert trigger["role_accuracy"] == 1.0
