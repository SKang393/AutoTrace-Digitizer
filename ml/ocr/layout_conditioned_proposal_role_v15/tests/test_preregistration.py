# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed design preregistration checks for OCR V15."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.layout_conditioned_proposal_role_v15.dataset import (
    encode_proposal,
    proposals,
    render_scene,
)
from ml.ocr.layout_conditioned_proposal_role_v15.model import LayoutConditionedProposalRoleNet
from ml.ocr.layout_conditioned_proposal_role_v15.model_p3 import AnchorScaledCandidate
from ml.ocr.layout_conditioned_proposal_role_v15.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.layout_conditioned_proposal_role_v15.train_p1 import RUNNER_SOURCE_PATHS, _export
from ml.ocr.layout_conditioned_proposal_role_v15.train_p2 import (
    P1_RESULT_PATH,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _hard_negative_indices,
)
from ml.ocr.layout_conditioned_proposal_role_v15.train_p3 import (
    P2_RESULT_PATH,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
)
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


def _assert_historical_source_binding(expected: str, current: str) -> None:
    assert len(expected) == 64 and set(expected) <= set("0123456789abcdef")
    assert expected != current


def test_protocol_is_canonical_fail_closed_and_aggregate_only() -> None:
    expected = protocol_configuration()
    historical = json.loads((V15_ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
    assert historical["split_policy"].pop("public_case_level_failure_analysis_permitted") is False
    assert expected.pop("evidence_policy") == "ml/policy/evidence-policy.json"
    assert canonical_json_bytes(historical) == canonical_json_bytes(expected)
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


def test_layout_encoder_and_model_contract_are_exact_and_finite() -> None:
    scene = render_scene("train", 0)
    candidate = proposals(scene.raster)[0]
    value = encode_proposal(scene.raster, candidate, scene.plot)
    assert value.shape == (2, 32, 152)
    assert np.isfinite(value).all()
    relative = value[0, 0, 144:]
    assert relative.shape == (8,)
    assert np.any(relative != 0.0)
    model = LayoutConditionedProposalRoleNet().eval()
    with torch.inference_mode():
        result = model(torch.from_numpy(np.stack((value, value))))
    assert result.shape == (2, 10)
    assert torch.isfinite(result).all()
    assert hasattr(model, "layout_role")
    with pytest.raises(ValueError, match="2,32,152"):
        model(torch.zeros((1, 2, 32, 151), dtype=torch.float32))


def test_layout_model_exports_dynamic_cpu_onnx(tmp_path: Path) -> None:
    model = LayoutConditionedProposalRoleNet().eval()
    source = torch.rand((5, 2, 32, 152), generator=torch.Generator().manual_seed(20262052))
    path = tmp_path / "v15-p1-export.onnx"
    _export(model, source, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    values = np.zeros((3, 2, 32, 152), dtype=np.float32)
    output = np.asarray(session.run(None, {"region_proposals": values})[0], dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(values)).numpy()
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert output.shape == (3, 10)
    assert np.isfinite(output).all()
    assert float(np.max(np.abs(expected - output))) <= 1e-5


def test_runner_and_public_evaluator_sources_are_bound_before_freeze() -> None:
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)
    assert source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS)
    assert GATE_CONFIG["evaluation_limit"] == 1
    assert GATE_CONFIG["case_level_failure_analysis_permitted"] is False
    assert GATE_CONFIG["provider"] == "CPUExecutionProvider"


@pytest.mark.parametrize("split,index", [
    ("train", 0), ("train", 257), ("train", 511),
    ("validation", 0), ("validation", 159),
    ("sealed_public", 0), ("sealed_public", 223),
])
def test_source_renderer_has_one_production_proposal_per_truth(split: str, index: int) -> None:
    scene = render_scene(split, index)
    candidates = proposals(scene.raster)
    assert len(scene.truths) == 8
    for truth in scene.truths:
        assert sum(box_iou(candidate.box, truth.box) >= 0.5 for candidate in candidates) == 1


def test_split_registrations_are_fresh_disjoint_and_frozen_fail_closed() -> None:
    assert [item.scene_count for item in SPLITS] == [512, 160, 224]
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    protocol = protocol_configuration()
    assert protocol["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert protocol["split_policy"]["v14_validation_fixture_bytes_scene_truth_or_case_identity_reused"] is False
    assert "no Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]
    selection_path = V15_ROOT / "SELECTION_MANIFEST.json"
    seal_path = V15_ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    gate_path = V15_ROOT / "gates/sealed-public-v1.json"
    candidate_path = V15_ROOT / "training/p1.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert selection["train"]["split_fingerprint"] == "e0aee54d8ccccb6f29737276eae628079bf739bbe5f7c303973006f994a2d35c"
    assert selection["validation"]["split_fingerprint"] == "fc03e37194b2aeff37a3828f5d62d551db602fc5be730b7193807190123fc7d1"
    assert selection["training_tensor_shape"] == [24272, 2, 32, 152]
    assert selection["sealed_public_truth_available_to_candidate"] is False
    assert seal["split_fingerprint"] == "47a19f9f0f1bec2ca4eead39a0cc673eec6df891d007bbd5e96e9079c30ddc40"
    assert seal["fixture_archive_sha256"] == "f2c6783ed0269aa794448b61b09cfca767a82e274247c98bd14d0ab3fcfcf722"
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert gate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert gate["evaluation_limit"] == 1
    assert candidate["candidate_id"] == "P1"
    assert candidate["selection_manifest_sha256"] == sha256_file(selection_path)
    assert candidate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert candidate["public_gate_archive_opened"] is False
    assert candidate["public_gate_evaluations"] == 0
    for value in (selection, seal, gate, candidate):
        assert value["production_approval"] is False
        assert value["release_eligible"] is False


def test_p1_result_and_p2_sources_are_checksum_bound() -> None:
    result_path = ROOT / P1_RESULT_PATH
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config_path = V15_ROOT / "training/p2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_selection"
    assert result["consumed"] is True
    assert result["selection_metrics"]["false_negatives"] == 1
    assert result["selection_metrics"]["false_positives"] == 0
    assert result["threshold_0_88_aggregate"]["false_negatives"] == 0
    assert result["threshold_0_88_aggregate"]["false_positives"] == 3
    assert result["public_gate_archive_opened"] is False
    assert config["p1_result_sha256"] == sha256_file(result_path)
    _assert_historical_source_binding(
        config["expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P2_RUNNER_SOURCE_PATHS),
    )
    assert config["p1_aggregate_metrics_only_used_for_design"] is True
    assert config["p1_validation_case_detail_or_pixels_used_for_design"] is False
    assert config["expected_optimizer_steps"] == 128
    assert config["public_gate_archive_opened"] is False


def test_p2_hard_negative_selector_uses_highest_training_scores_only() -> None:
    class Scores(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            score = values[:, 0, 0, 0]
            return torch.stack((-score, score), dim=1)

    values = torch.zeros((6, 2, 32, 152), dtype=torch.float32)
    values[:, 0, 0, 0] = torch.tensor([0.1, 0.7, 0.4, 0.9, 0.3, 0.8])
    labels = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.int64)
    selected = _hard_negative_indices(Scores(), values, labels, 2, 3)
    assert selected.tolist() == [5, 1]


def test_p2_result_and_p3_sources_are_checksum_bound() -> None:
    result_path = ROOT / P2_RESULT_PATH
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config_path = V15_ROOT / "training/p3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_selection"
    assert result["consumed"] is True
    assert result["selection_metrics"]["exact_scene_count"] == 160
    assert result["selection_metrics"]["false_positives"] == 0
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["role_accuracy"] == 1.0
    assert result["onnx_parity_maximum_absolute_error"] == 1.1444091796875e-05
    assert result["onnx_parity_passed"] is False
    assert result["public_gate_archive_opened"] is False
    assert config["p2_result_sha256"] == sha256_file(result_path)
    assert config["p2_report_sha256"] == result["candidate_report_sha256"]
    assert config["p2_checkpoint_sha256"] == result["checkpoint_sha256"]
    assert config["p2_onnx_sha256"] == result["onnx_sha256"]
    _assert_historical_source_binding(
        config["expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P3_RUNNER_SOURCE_PATHS),
    )
    assert config["expected_optimizer_steps"] == 0
    assert config["weights_changed"] is False
    assert config["p2_aggregate_metrics_only_used_for_design"] is True
    assert config["p2_validation_case_detail_or_pixels_used_for_design"] is False
    assert config["public_gate_archive_opened"] is False


def test_p3_anchor_calibration_preserves_acceptance_and_roles() -> None:
    class FixedLogits(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values

    logits = torch.tensor([
        [0.0, -2.0, 8.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        [0.0, 0.0, -2.0, 9.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        [0.0, 0.6632942174102642, -4.0, -3.0, 7.0, 1.0, 0.0, -1.0, -2.0, -3.0],
        [0.0, 2.0, -6.0, -5.0, -4.0, 8.0, 1.0, 0.0, -1.0, -2.0],
    ], dtype=torch.float32)
    candidate = AnchorScaledCandidate(FixedLogits(), scale=0.8, anchor=0.66).eval()
    with torch.inference_mode():
        calibrated = candidate(logits)
    base_accepted = torch.softmax(logits[:, :2], dim=1)[:, 1] >= 0.66
    calibrated_accepted = torch.softmax(calibrated[:, :2], dim=1)[:, 1] >= 0.66
    assert torch.equal(base_accepted, calibrated_accepted)
    assert torch.equal(logits[:, 2:].argmax(dim=1), calibrated[:, 2:].argmax(dim=1))
    assert all(parameter.requires_grad is False for parameter in candidate.parameters())


def test_p3_result_is_selected_and_public_gate_remains_closed() -> None:
    result_path = V15_ROOT / "P3_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "selected"
    assert result["consumed"] is True
    assert result["selection_gate_passed"] is True
    assert result["optimizer_steps"] == 0
    assert result["weights_changed"] is False
    assert result["selected_threshold"] == 0.66
    assert result["selection_metrics"]["exact_scene_count"] == 160
    assert result["selection_metrics"]["true_positives"] == 1280
    assert result["selection_metrics"]["false_positives"] == 0
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["duplicate_region_count"] == 0
    assert result["selection_metrics"]["prohibited_structure_hits"] == 0
    assert min(result["selection_metrics"]["per_role_accuracy"].values()) == 1.0
    assert result["anchor_acceptance_mismatch_count"] == 0
    assert result["role_argmax_mismatch_count"] == 0
    assert result["onnx_parity_passed"] is True
    assert result["onnx_parity_maximum_absolute_error"] == 5.7220458984375e-06
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_authorized"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["production_approval"] is False
    assert result["release_eligible"] is False


def test_public_gate_result_fails_closed_without_case_details() -> None:
    result_path = V15_ROOT / "PUBLIC_GATE_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "fail"
    assert result["rerun_allowed"] is False
    assert result["public_gate_evaluations"] == 1
    assert result["direct_execution_inference_calls"] == 224
    assert result["public_scene_count"] == 224
    assert result["public_exact_scene_count"] == 217
    assert result["public_true_positives"] == 1791
    assert result["public_false_positives"] == 1
    assert result["public_false_negatives"] == 1
    assert result["public_duplicate_region_count"] == 0
    assert result["public_prohibited_structure_hits"] == 1
    assert result["production_approval"] is False
    assert result["release_eligible"] is False


def test_canonical_ledger_exhausts_v15_after_failed_public_gate() -> None:
    ledger = json.loads(
        (ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item.get("revision") == "graph-text-layout-conditioned-proposal-role-v15"
    )
    assert entry["status"] == "exhausted_failed_public_gate"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["protocol_sha256"] == sha256_file(V15_ROOT / "PROTOCOL.json")
    assert entry["split_generator_source_bundle_sha256"] == "502d3fefa949acc1b755871005fcf66824ba07ee04b1c9515d9d9874e62ff3e5"
    _assert_historical_source_binding(
        entry["p1_expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS),
    )
    _assert_historical_source_binding(
        entry["expected_public_evaluator_source_bundle_sha256"],
        source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS),
    )
    assert entry["selection_manifest_sha256"] == sha256_file(V15_ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(
        V15_ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    )
    assert entry["public_gate_config_sha256"] == sha256_file(
        V15_ROOT / "gates/sealed-public-v1.json"
    )
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(V15_ROOT / "training/p1.json")
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(V15_ROOT / "training/p2.json")
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(V15_ROOT / "training/p3.json")
    assert entry["p1_result_sha256"] == sha256_file(V15_ROOT / "P1_RESULT.json")
    _assert_historical_source_binding(
        entry["p2_expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P2_RUNNER_SOURCE_PATHS),
    )
    assert entry["p2_result_sha256"] == sha256_file(V15_ROOT / "P2_RESULT.json")
    _assert_historical_source_binding(
        entry["p3_expected_runner_source_bundle_sha256"],
        source_bundle_sha256(ROOT, P3_RUNNER_SOURCE_PATHS),
    )
    assert entry["p3_expected_optimizer_steps"] == 0
    assert entry["p3_output_scale"] == 0.8
    assert entry["p3_anchor_threshold"] == 0.66
    assert entry["p3_result_sha256"] == sha256_file(V15_ROOT / "P3_RESULT.json")
    assert entry["p3_selection_report_sha256"] == "1c89f43258a561e77421244deb5a3f093cb5ce68bd567dc7320ebd86ada0cc12"
    assert entry["p3_onnx_sha256"] == "ff145e14a6f84170e1d727dbd89f57316cb7ee2eaccb6c7f3df5013271d7c948"
    assert entry["p3_selection_exact_scene_count"] == 160
    assert entry["p3_selection_true_positives"] == 1280
    assert entry["p3_selection_false_positives"] == 0
    assert entry["p3_selection_false_negatives"] == 0
    assert entry["p3_anchor_acceptance_mismatch_count"] == 0
    assert entry["p3_role_argmax_mismatch_count"] == 0
    assert entry["p3_onnx_parity_passed"] is True
    assert entry["split_materialized"] is True
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert "cannot rerun" in entry["execution_blocker"]
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] is None
    assert entry["public_gate_evaluations"] == 1
    assert entry["public_gate_archive_opened"] is True
    assert entry["public_gate_result_sha256"] == sha256_file(V15_ROOT / "PUBLIC_GATE_RESULT.json")
    assert entry["public_gate_report_sha256"] == "8bd7170db115f6fccbfc9b998bd5f6fce0d8ae001469b692fa07e8392068553d"
    assert entry["public_gate_status"] == "fail"
    assert entry["public_gate_exact_scene_count"] == 217
    assert entry["public_gate_false_positives"] == 1
    assert entry["public_gate_false_negatives"] == 1
    assert entry["public_gate_prohibited_structure_hits"] == 1
    assert entry["public_gate_case_level_details_emitted"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
