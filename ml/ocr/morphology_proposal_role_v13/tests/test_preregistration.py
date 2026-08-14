# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V13."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.morphology_proposal_role_v13.dataset import render_scene, proposals
from ml.ocr.morphology_proposal_role_v13.model import MorphologyProposalRoleNet
from ml.ocr.morphology_proposal_role_v13.sealed_gate import GATE_CONFIG
from ml.ocr.morphology_proposal_role_v13.train_p1 import RUNNER_SOURCE_PATHS
from ml.ocr.morphology_proposal_role_v13.train_p2 import RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS
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


def test_split_and_candidate_records_are_frozen_and_fail_closed() -> None:
    for relative in (
        "SELECTION_MANIFEST.json", "SEALED_PUBLIC_TEST_SEAL.json",
        "gates/sealed-public-v1.json", "training/p1.json", "training/p2.json",
    ):
        assert (V13_ROOT / relative).is_file()
    selection = json.loads((V13_ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    seal = json.loads((V13_ROOT / "SEALED_PUBLIC_TEST_SEAL.json").read_text(encoding="utf-8"))
    config = json.loads((V13_ROOT / "training/p1.json").read_text(encoding="utf-8"))
    assert selection["sealed_public_truth_available_to_candidate"] is False
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert config["public_gate_archive_opened"] is False
    result = json.loads((V13_ROOT / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_selection"
    assert result["consumed"] is True
    assert result["selection_metrics"]["exact_scene_count"] == 159
    assert result["selection_metrics"]["scene_count"] == 160
    assert result["selection_metrics"]["false_positives"] == 1
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["prohibited_structure_hits"] == 1
    assert result["selection_metrics"]["role_accuracy"] == 1.0
    assert result["onnx_parity_passed"] is False
    assert result["onnx_parity_maximum_absolute_error"] == 1.52587890625e-05
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    p2_result = json.loads((V13_ROOT / "P2_RESULT.json").read_text(encoding="utf-8"))
    assert p2_result["status"] == "selected"
    assert p2_result["consumed"] is True
    assert p2_result["selection_metrics"]["exact_scene_count"] == 160
    assert p2_result["selection_metrics"]["scene_count"] == 160
    assert p2_result["selection_metrics"]["false_positives"] == 0
    assert p2_result["selection_metrics"]["false_negatives"] == 0
    assert p2_result["selection_metrics"]["duplicate_region_count"] == 0
    assert p2_result["selection_metrics"]["prohibited_structure_hits"] == 0
    assert p2_result["selection_metrics"]["role_accuracy"] == 1.0
    assert min(p2_result["selection_metrics"]["per_role_accuracy"].values()) == 1.0
    assert p2_result["onnx_parity_passed"] is True
    assert p2_result["onnx_parity_maximum_absolute_error"] == 7.62939453125e-06
    assert p2_result["public_gate_archive_opened"] is False
    assert p2_result["public_gate_evaluations"] == 0
    assert not (V13_ROOT / "artifacts/P1-sealed-public").exists()
    for record in (selection, seal, config):
        assert record["production_approval"] is False
        assert record["release_eligible"] is False


def test_runners_are_fail_closed_before_split_and_budget_freeze() -> None:
    assert len(RUNNER_SOURCE_PATHS) == len(set(RUNNER_SOURCE_PATHS))
    assert GATE_CONFIG["evaluation_limit"] == 1
    assert GATE_CONFIG["case_level_failure_analysis_permitted"] is False
    assert GATE_CONFIG["false_regions"] == GATE_CONFIG["missed_regions"] == 0
    assert GATE_CONFIG["direct_fixture_byte_execution_required"] is True


def test_v13_budget_ledger_authorizes_only_exact_p2() -> None:
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == "graph-text-morphology-proposal-role-v13")
    assert entry["status"] == "exhausted_failed_public_gate"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["retired_unregistered_candidate_ids"] == ["P3"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert "single V13 public gate is consumed" in entry["execution_blocker"]
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(V13_ROOT / "training/p1.json")
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(V13_ROOT / "training/p2.json")
    assert entry["p1_result_sha256"] == sha256_file(V13_ROOT / "P1_RESULT.json")
    assert entry["p1_selection_exact_scene_count"] == 159
    assert entry["p1_selection_scene_count"] == 160
    assert entry["p1_selection_false_positives"] == 1
    assert entry["p1_selection_false_negatives"] == 0
    assert entry["p1_selection_prohibited_structure_hits"] == 1
    assert entry["p1_selection_role_accuracy"] == 1.0
    assert entry["p1_onnx_parity_passed"] is False
    p2 = json.loads((V13_ROOT / "training/p2.json").read_text(encoding="utf-8"))
    assert p2["p1_result_sha256"] == sha256_file(V13_ROOT / "P1_RESULT.json")
    assert p2["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, P2_RUNNER_SOURCE_PATHS)
    assert p2["optimizer_steps"] == 0
    assert p2["weights_changed"] is False
    assert p2["output_scale"] == 0.5
    assert p2["effective_output_logit_scale"] == 0.05
    assert p2["validation_or_public_pixels_used_for_design"] is False
    assert p2["public_gate_archive_opened"] is False
    assert entry["p2_source_checkpoint_sha256"] == p2["p1_checkpoint_sha256"]
    assert entry["p2_expected_runner_source_bundle_sha256"] == p2["expected_runner_source_bundle_sha256"]
    assert entry["p2_result_sha256"] == sha256_file(V13_ROOT / "P2_RESULT.json")
    assert entry["p2_selection_exact_scene_count"] == 160
    assert entry["p2_selection_scene_count"] == 160
    assert entry["p2_selection_false_positives"] == 0
    assert entry["p2_selection_false_negatives"] == 0
    assert entry["p2_selection_duplicate_count"] == 0
    assert entry["p2_selection_prohibited_structure_hits"] == 0
    assert entry["p2_selection_role_accuracy"] == 1.0
    assert entry["p2_onnx_parity_passed"] is True
    assert entry["selection_manifest_sha256"] == sha256_file(V13_ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(V13_ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] is None
    assert entry["public_gate_authorized_on_selection_pass"] is True
    assert entry["public_gate_evaluations"] == 1
    assert entry["public_gate_archive_opened"] is True
    assert entry["public_gate_status"] == "fail"
    assert entry["public_gate_report_sha256"] == "fe0a0d63d0c00c35c463e45c11a87f97aa81bb625d0744c8c2307537b6d6d2ff"
    assert entry["public_gate_scene_count"] == 224
    assert entry["public_gate_exact_scene_count"] == 223
    assert entry["public_gate_true_positives"] == 1792
    assert entry["public_gate_false_positives"] == 1
    assert entry["public_gate_false_negatives"] == 0
    assert entry["public_gate_duplicate_count"] == 0
    assert entry["public_gate_prohibited_structure_hits"] == 1
    assert entry["public_gate_role_accuracy"] == 1.0
    assert entry["public_gate_minimum_per_role_accuracy"] == 1.0
    assert entry["public_gate_case_level_details_emitted"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
