# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR V14."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.structural_graph_proposal_role_v14.dataset import render_scene, proposals
from ml.ocr.structural_graph_proposal_role_v14.model import StructuralGraphProposalRoleNet
from ml.ocr.structural_graph_proposal_role_v14.model_p2 import (
    COLUMN_BIN_BOUNDS, StructuralGraphProposalRoleP2Net,
)
from ml.ocr.structural_graph_proposal_role_v14.model_p3 import (
    OutputScaledCandidate, StructuralGraphProposalRoleP3Net,
)
from ml.ocr.structural_graph_proposal_role_v14.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.structural_graph_proposal_role_v14.train_p1 import RUNNER_SOURCE_PATHS
from ml.ocr.structural_graph_proposal_role_v14.train_p2 import (
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS, _export as export_p2,
)
from ml.ocr.structural_graph_proposal_role_v14.train_p3 import (
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS, _export as export_p3,
)
from ml.ocr.structural_graph_proposal_role_v14.protocol import (
    ENCODED_WIDTH, EXPERIMENT_BUDGET, ROLE_ORDER, SPLITS, protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
V14_ROOT = ROOT / "ml/ocr/structural_graph_proposal_role_v14"
PROTOCOL_PATH = V14_ROOT / "PROTOCOL.json"


def _assert_historical_source_binding(expected: str, current: str) -> None:
    assert len(expected) == 64 and set(expected) <= set("0123456789abcdef")
    assert expected != current


def test_protocol_is_canonical_fail_closed_and_aggregate_only() -> None:
    expected = protocol_configuration()
    historical = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert historical["split_policy"].pop("public_case_level_failure_analysis_permitted") is False
    assert expected.pop("evidence_policy") == "ml/policy/evidence-policy.json"
    assert canonical_json_bytes(historical) == canonical_json_bytes(expected)
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


def test_p2_static_occupancy_preserves_p1_outputs_and_state_contract() -> None:
    assert COLUMN_BIN_BOUNDS == tuple(
        (index * 128 // 18, ((index + 1) * 128 + 17) // 18)
        for index in range(18)
    )
    p1 = StructuralGraphProposalRoleNet().eval()
    p2 = StructuralGraphProposalRoleP2Net().eval()
    assert tuple(p1.state_dict()) == tuple(p2.state_dict())
    generator = torch.Generator().manual_seed(20262042)
    value = torch.rand((7, 2, 32, ENCODED_WIDTH), generator=generator)
    with torch.inference_mode():
        p1_output = p1(value)
        p2_output = p2(value)
    assert torch.max(torch.abs(p1_output - p2_output)).item() <= 1e-6
    assert torch.equal(torch.argmax(p1_output, dim=1), torch.argmax(p2_output, dim=1))


def test_p2_fixed_occupancy_exports_dynamic_cpu_onnx(tmp_path: Path) -> None:
    model = StructuralGraphProposalRoleP2Net().eval()
    source = torch.rand((5, 2, 32, ENCODED_WIDTH), generator=torch.Generator().manual_seed(20262043))
    path = tmp_path / "p2-export.onnx"
    export_p2(model, source, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    values = np.zeros((3, 2, 32, ENCODED_WIDTH), dtype=np.float32)
    output = np.asarray(session.run(None, {"region_proposals": values})[0], dtype=np.float32)
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert output.shape == (3, 10)
    assert np.isfinite(output).all()
    with torch.inference_mode():
        expected = model(torch.from_numpy(values)).numpy()
    assert float(np.max(np.abs(expected - output))) <= 1e-5


def test_p3_zero_initialized_role_residual_preserves_p2_and_exports_scaled_cpu_onnx(tmp_path: Path) -> None:
    p2 = StructuralGraphProposalRoleP2Net().eval()
    p3 = StructuralGraphProposalRoleP3Net().eval()
    source = torch.rand((5, 2, 32, ENCODED_WIDTH), generator=torch.Generator().manual_seed(20262045))
    with torch.inference_mode():
        p2_output = p2(source)
        p3_output = p3(source)
    assert torch.equal(p2_output, p3_output)
    assert torch.count_nonzero(p3.role_geometry_residual[2].weight).item() == 0
    wrapped = OutputScaledCandidate(p3, 0.5).eval()
    path = tmp_path / "p3-export.onnx"
    export_p3(wrapped, source, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    values = np.zeros((3, 2, 32, ENCODED_WIDTH), dtype=np.float32)
    output = np.asarray(session.run(None, {"region_proposals": values})[0], dtype=np.float32)
    with torch.inference_mode():
        expected = wrapped(torch.from_numpy(values)).numpy()
    assert output.shape == (3, 10)
    assert np.isfinite(output).all()
    assert float(np.max(np.abs(expected - output))) <= 1e-5


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
    _assert_historical_source_binding(
        config["expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS),
    )
    result = json.loads((V14_ROOT / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_runner"
    assert result["failure_phase"] == "export"
    assert result["optimizer_steps"] == 1616
    assert result["onnx_created"] is False
    assert result["selection_metrics_available"] is False
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    p2_result = json.loads((V14_ROOT / "P2_RESULT.json").read_text(encoding="utf-8"))
    assert p2_result["status"] == "failed_selection"
    assert p2_result["optimizer_steps"] == 0
    assert p2_result["weights_changed"] is False
    assert p2_result["selection_metrics"]["true_positives"] == 1152
    assert p2_result["selection_metrics"]["false_positives"] == 2
    assert p2_result["selection_metrics"]["false_negatives"] == 0
    assert p2_result["selection_metrics"]["role_accuracy"] == 0.9444444444444444
    assert p2_result["onnx_parity_passed"] is False
    assert p2_result["public_gate_archive_opened"] is False
    assert p2_result["public_gate_evaluations"] == 0
    p3_result = json.loads((V14_ROOT / "P3_RESULT.json").read_text(encoding="utf-8"))
    assert p3_result["status"] == "failed_selection"
    assert p3_result["optimizer_steps"] == 808
    assert p3_result["weights_changed"] is True
    assert p3_result["selection_metrics"]["true_positives"] == 1150
    assert p3_result["selection_metrics"]["false_positives"] == 0
    assert p3_result["selection_metrics"]["false_negatives"] == 2
    assert p3_result["selection_metrics"]["prohibited_structure_hits"] == 0
    assert p3_result["selection_metrics"]["per_role_accuracy"]["PhaseHeading"] == 0.7638888888888888
    assert p3_result["onnx_parity_passed"] is True
    assert p3_result["public_gate_archive_opened"] is False
    assert p3_result["public_gate_evaluations"] == 0
    for record in (selection, seal, config):
        assert record["production_approval"] is False
        assert record["release_eligible"] is False


def test_frozen_budget_ledger_exhausts_p3_and_keeps_public_gate_closed() -> None:
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == "graph-text-structural-graph-proposal-role-v14")
    assert entry["status"] == "exhausted"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert "V14 is exhausted" in entry["execution_blocker"]
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] is None
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["selection_manifest_sha256"] == sha256_file(V14_ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(V14_ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(V14_ROOT / "training/p1.json")
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(V14_ROOT / "training/p2.json")
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(V14_ROOT / "training/p3.json")
    _assert_historical_source_binding(
        entry["p1_expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS),
    )
    assert entry["p1_result_sha256"] == sha256_file(V14_ROOT / "P1_RESULT.json")
    assert entry["p1_selection_report_sha256"] == "6019b0612cd968248bd1c7379dfbf4f527c0b79787c97e0775c4d35129eb6c45"
    assert entry["p1_checkpoint_sha256"] == "e09c250e567a00b7c79e4995235a9ac3aa5ecb84fdf5f3eda44254efda36ce28"
    assert entry["p1_optimizer_steps"] == 1616
    assert entry["p1_failure_phase"] == "export"
    assert entry["p1_training_opened_seal_sha256"] == "d36e462f13e22959d0fd90fc17be6a757dac941fa8f0e7a7683975526469dd8c"
    assert entry["p1_training_result_seal_sha256"] == "fce5aaf7ba68223a2d9175a2f77a83c24e0ee7ce245667abc19923b1a94d6748"
    _assert_historical_source_binding(
        entry["p2_expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P2_RUNNER_SOURCE_PATHS),
    )
    assert entry["p2_result_sha256"] == sha256_file(V14_ROOT / "P2_RESULT.json")
    assert entry["p2_selection_report_sha256"] == "66db4c1819bbb62ec5a1d900285f612f4e57aba692630f10c67cad3a62dfc6c2"
    assert entry["p2_onnx_sha256"] == "2e429dc0c58bf587385bc20a03fa542bbe5e39f41972d6921ff96d6f970b5232"
    assert entry["p2_selection_exact_scene_count"] == 81
    assert entry["p2_selection_false_positives"] == 2
    assert entry["p2_selection_false_negatives"] == 0
    assert entry["p2_selection_role_accuracy"] == 0.9444444444444444
    assert entry["p2_phase_heading_accuracy"] == 0.7847222222222222
    assert entry["p2_y_tick_accuracy"] == 0.7777777777777778
    assert entry["p2_pytorch_equivalence_passed"] is False
    assert entry["p2_onnx_parity_passed"] is False
    _assert_historical_source_binding(
        entry["p3_expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P3_RUNNER_SOURCE_PATHS),
    )
    assert entry["p3_result_sha256"] == sha256_file(V14_ROOT / "P3_RESULT.json")
    assert entry["p3_selection_report_sha256"] == "89b66d006166a8a2efb770c029e0a3f9dc76a5ed0425a2ee16a9f0900da262a4"
    assert entry["p3_checkpoint_sha256"] == "0d7bdfce92f8ad37f4483f60eb99f50f1fd13c08797713ba210ae523ae172bb1"
    assert entry["p3_onnx_sha256"] == "6fc8ec245262eb99aaa2c6fd315be97353dea569fc336aeb248b2a808f9014d6"
    assert entry["p3_optimizer_steps"] == 808
    assert entry["p3_weights_changed"] is True
    assert entry["p3_selection_exact_scene_count"] == 89
    assert entry["p3_selection_true_positives"] == 1150
    assert entry["p3_selection_false_positives"] == 0
    assert entry["p3_selection_false_negatives"] == 2
    assert entry["p3_selection_prohibited_structure_hits"] == 0
    assert entry["p3_phase_heading_accuracy"] == 0.7638888888888888
    assert entry["p3_y_tick_accuracy"] == 0.8680555555555556
    assert entry["p3_onnx_parity_passed"] is True
    assert entry["p3_training_opened_seal_sha256"] == "d7776678316e681e588f0cbd51e46ef983f5a67ee36eeb545824d7b8195cf6a5"
    assert entry["p3_training_result_seal_sha256"] == "7c01f522acc478e764914d97f161079c9e2682195ddc186282971625b0548c23"
    p3 = json.loads((V14_ROOT / "training/p3.json").read_text(encoding="utf-8"))
    _assert_historical_source_binding(
        p3["expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P3_RUNNER_SOURCE_PATHS),
    )
    assert p3["p2_aggregate_trigger_only"] is True
    assert p3["p2_validation_case_detail_or_pixels_used_for_design"] is False
    assert p3["expected_optimizer_steps"] == 808
    assert p3["output_scale"] == 0.5
    p2 = json.loads((V14_ROOT / "training/p2.json").read_text(encoding="utf-8"))
    _assert_historical_source_binding(
        p2["expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P2_RUNNER_SOURCE_PATHS),
    )
    assert p2["optimizer_steps"] == 0
    assert p2["weights_changed"] is False
    assert p2["p1_checkpoint_sha256"] == entry["p1_checkpoint_sha256"]
    _assert_historical_source_binding(
        entry["expected_public_evaluator_source_bundle_sha256"],
        source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS),
    )
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
