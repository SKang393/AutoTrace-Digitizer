# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration tests for OCR V26."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    source_bundle_sha256,
)
from ml.ocr.scene_topology_proposal_v26.dataset import proposal_summary, render_scene
from ml.ocr.scene_topology_proposal_v26.model import (
    FrozenRoleAxialTopologyProposalNet,
)
from ml.ocr.scene_topology_proposal_v26.model_p2 import (
    FrozenP1BoundedMarginProposalNet,
)
from ml.ocr.scene_topology_proposal_v26.model_p3 import (
    FrozenP2FinalTailProposalNet,
)
from ml.ocr.scene_topology_proposal_v26.sealed_gate import (
    EVALUATOR_SOURCE_PATHS,
    GATE_CONFIG,
    _public_window,
)
from ml.ocr.scene_topology_proposal_v26.protocol import (
    CANDIDATE_LIMIT,
    FEATURE_COUNT,
    ROLE_ORDER,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.scene_topology_proposal_v26.train_p1 import (
    RUNNER_SOURCE_PATHS,
    _proposal_objective,
)
from ml.ocr.scene_topology_proposal_v26.train_p2 import (
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _proposal_objective as _p2_proposal_objective,
)
from ml.ocr.scene_topology_proposal_v26.train_p3 import (
    CANONICAL_OUTPUT as P3_CANONICAL_OUTPUT,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
    _cpu_session_basic,
    _proposal_objective as _p3_proposal_objective,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_fresh_bounded_and_fail_closed() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (REPO_ROOT / "ml/ocr/scene_topology_proposal_v26/PROTOCOL.json").read_text(
            encoding="utf-8"
        )
    )
    assert tracked == json.loads(json.dumps(protocol))
    assert protocol["candidate_budget"]["candidate_limit"] == CANDIDATE_LIMIT == 3
    assert protocol["candidate_budget"]["optimizer_steps_maximum"] == 2304
    assert protocol["fixture_identity_frozen"] is False
    assert protocol["training_authorized"] is False
    assert protocol["public_execution_authorized"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["private_validation_authorized"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert "Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]


def test_v25_trigger_is_exact_aggregate_only_terminal_record() -> None:
    path = REPO_ROOT / TRIGGER_RESULT_PATH
    assert sha256_file(path) == TRIGGER_RESULT_SHA256
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["selection_metrics"]
    assert result["candidate_id"] == "P3"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["case_level_details_emitted"] is False
    assert metrics["exact_scene_count"] == 112
    assert metrics["true_positives"] == 1017
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 7
    assert metrics["duplicate_region_count"] == 0
    assert metrics["prohibited_structure_hits"] == 1
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert "cases" not in result and "predictions" not in result


def test_canonical_budget_records_p1_p2_failures_and_authorizes_only_p3() -> None:
    ledger = json.loads(
        (REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item for item in ledger["revisions"]
        if item["task"] == "ocr-detection-recognition"
        and item["revision"] == "graph-text-scene-topology-proposal-v26"
    )
    assert entry["status"] == "candidate_3_preregistered"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P3"]
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["split_materialized"] is True
    split_seal_path = REPO_ROOT / entry["split_seal_path"]
    assert entry["split_seal_sha256"] == sha256_file(split_seal_path)
    split_seal = json.loads(split_seal_path.read_text(encoding="utf-8"))
    assert split_seal["source_commit"] == entry["split_source_commit"]
    assert split_seal["source_bundle_sha256"] == entry["split_source_bundle_sha256"]
    assert split_seal["cross_split_source_overlap_counts"] == {
        "train_sealed_public": 0,
        "train_validation": 0,
        "validation_sealed_public": 0,
    }
    for name in ("train", "validation", "sealed_public"):
        registered = split_seal["splits"][name]
        assert entry["split_archive_sha256"][name] == registered["archive_sha256"]
        assert entry["split_manifest_sha256"][name] == registered["manifest_sha256"]
        assert entry["split_fingerprints"][name] == registered["split_fingerprint"]
        assert entry["split_proposal_counts"][name] == (
            registered["proposal_summary"]["proposal_count"]
        )
    public_config_path = REPO_ROOT / entry["public_gate_config_path"]
    assert entry["public_gate_config_sha256"] == sha256_file(public_config_path)
    public_config = json.loads(public_config_path.read_text(encoding="utf-8"))
    assert public_config["split_seal_sha256"] == entry["split_seal_sha256"]
    assert public_config["expected_evaluator_source_bundle_sha256"] == (
        source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    )
    assert public_config["expected_gate_config_sha256"] == sha256_bytes(
        canonical_json_bytes(dict(GATE_CONFIG))
    )
    assert public_config["public_execution_authorized"] is False
    assert public_config["public_evaluations"] == 0
    assert public_config["public_archive_opened"] is False
    candidate_config_path = REPO_ROOT / entry["candidate_config_paths"]["P1"]
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(
        candidate_config_path
    )
    candidate_config = json.loads(candidate_config_path.read_text(encoding="utf-8"))
    assert candidate_config["candidate_id"] == "P1"
    assert candidate_config["expected_optimizer_steps"] == 2304
    assert candidate_config["expected_runner_source_bundle_sha256"] == (
        source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    )
    assert entry["candidate_runner_source_bundle_sha256"]["P1"] == (
        candidate_config["expected_runner_source_bundle_sha256"]
    )
    assert candidate_config["public_execution_authorized"] is False
    assert candidate_config["private_or_article_images"] is False
    assert candidate_config["chandler_included"] is False
    result_path = REPO_ROOT / entry["candidate_result_paths"]["P1"]
    assert entry["candidate_result_sha256"]["P1"] == sha256_file(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["optimizer_steps"] == 2304
    assert result["selection_metrics"]["true_positives"] == 1024
    assert result["selection_metrics"]["false_positives"] == 5
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["prohibited_structure_hits"] == 5
    assert result["onnx_parity_maximum_absolute_error"] > 1e-5
    assert result["passing_threshold_window"] == []
    assert result["case_level_details_emitted"] is False
    assert "cases" not in result and "predictions" not in result
    assert entry["candidate_report_sha256"]["P1"] == result["report_sha256"]
    assert entry["candidate_checkpoint_sha256"]["P1"] == result["checkpoint_sha256"]
    assert entry["candidate_onnx_sha256"]["P1"] == result["onnx_sha256"]
    p2_config_path = REPO_ROOT / entry["candidate_config_paths"]["P2"]
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(p2_config_path)
    p2_config = json.loads(p2_config_path.read_text(encoding="utf-8"))
    assert p2_config["candidate_id"] == "P2"
    assert p2_config["expected_optimizer_steps"] == 1152
    assert p2_config["trainable_scope"] == "proposal_head_only"
    assert p2_config["p1_case_detail_or_pixels_used"] is False
    assert p2_config["expected_runner_source_bundle_sha256"] == (
        source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS)
    )
    assert entry["candidate_runner_source_bundle_sha256"]["P2"] == (
        p2_config["expected_runner_source_bundle_sha256"]
    )
    p2_result_path = REPO_ROOT / entry["candidate_result_paths"]["P2"]
    assert entry["candidate_result_sha256"]["P2"] == sha256_file(p2_result_path)
    p2_result = json.loads(p2_result_path.read_text(encoding="utf-8"))
    assert p2_result["candidate_consumed"] is True
    assert p2_result["status"] == "failed_selection"
    assert p2_result["optimizer_steps"] == 1152
    assert p2_result["selection_metrics"]["true_positives"] == 1024
    assert p2_result["selection_metrics"]["false_positives"] == 1
    assert p2_result["selection_metrics"]["false_negatives"] == 0
    assert p2_result["selection_metrics"]["duplicate_region_count"] == 0
    assert p2_result["selection_metrics"]["prohibited_structure_hits"] == 1
    assert p2_result["selection_metrics"]["exact_scene_count"] == 122
    assert p2_result["onnx_parity_maximum_absolute_error"] > 1e-5
    assert p2_result["parent_role_maximum_absolute_error"] == 0.0
    assert p2_result["passing_threshold_window"] == []
    assert p2_result["case_level_details_emitted"] is False
    assert "cases" not in p2_result and "predictions" not in p2_result
    assert entry["candidate_report_sha256"]["P2"] == p2_result["report_sha256"]
    assert entry["candidate_checkpoint_sha256"]["P2"] == (
        p2_result["checkpoint_sha256"]
    )
    assert entry["candidate_onnx_sha256"]["P2"] == p2_result["onnx_sha256"]
    assert entry["candidate_training_opened_seal_sha256"]["P2"] == (
        p2_result["training_opened_seal_sha256"]
    )
    assert entry["candidate_training_result_seal_sha256"]["P2"] == (
        p2_result["training_result_seal_sha256"]
    )
    p3_config_path = REPO_ROOT / entry["candidate_config_paths"]["P3"]
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(p3_config_path)
    p3_config = json.loads(p3_config_path.read_text(encoding="utf-8"))
    assert p3_config["candidate_id"] == "P3"
    assert p3_config["expected_optimizer_steps"] == 1152
    assert p3_config["trainable_scope"] == "proposal_head_final_linear_only"
    assert p3_config["candidate_onnx_graph_optimization_level"] == (
        "ORT_ENABLE_BASIC"
    )
    assert p3_config["p2_case_detail_or_pixels_used"] is False
    assert p3_config["expected_runner_source_bundle_sha256"] == (
        source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS)
    )
    assert entry["candidate_runner_source_bundle_sha256"]["P3"] == (
        p3_config["expected_runner_source_bundle_sha256"]
    )
    assert not (REPO_ROOT / P3_CANONICAL_OUTPUT).exists()
    assert "P3" not in entry["candidate_result_paths"]
    assert entry["selection_evaluations"] == 2
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P3"
    assert "may execute once" in entry["execution_authorization"]
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["manifest_created"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    protocol_path = REPO_ROOT / entry["protocol_path"]
    assert entry["protocol_sha256"] == sha256_file(protocol_path)


def test_split_families_and_seed_offsets_are_disjoint() -> None:
    registrations = [
        split_registration(name) for name in ("train", "validation", "sealed_public")
    ]
    assert [item.scene_count for item in registrations] == [384, 128, 192]
    assert len({item.seed_offset for item in registrations}) == 3
    renderer_sets = [set(item.renderer_families) for item in registrations]
    degradation_sets = [set(item.degradation_families) for item in registrations]
    assert not any(
        renderer_sets[left] & renderer_sets[right]
        for left in range(3) for right in range(left + 1, 3)
    )
    assert not any(
        degradation_sets[left] & degradation_sets[right]
        for left in range(3) for right in range(left + 1, 3)
    )


def test_renderer_produces_fresh_complete_proposal_scenes() -> None:
    scenes = tuple(
        render_scene(split, 0) for split in ("train", "validation", "sealed_public")
    )
    assert len({scene.raster.tobytes() for scene in scenes}) == 3
    for scene in scenes:
        summary = proposal_summary((scene,))
        assert summary["positive_proposal_count"] == len(ROLE_ORDER)
        assert summary["negative_proposal_count"] > 0
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert scene.raster.shape == (320, 640)
        assert scene.scene_id.startswith("scene-topology-v26-")


def test_model_exports_dynamic_cpu_and_preserves_parent_roles(tmp_path: Path) -> None:
    torch.manual_seed(2601)
    model = FrozenRoleAxialTopologyProposalNet().eval()
    evidence = torch.linspace(-1.0, 1.0, 5 * FEATURE_COUNT).reshape(
        1, 5, FEATURE_COUNT,
    )
    crops = torch.linspace(-1.0, 1.0, 5 * 2 * 32 * 128).reshape(1, 5, 2, 32, 128)
    path = tmp_path / "scene-topology-v26.onnx"
    torch.onnx.export(
        model,
        (evidence, crops),
        path,
        input_names=["proposal_evidence", "proposal_crops"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for count in (3, 5):
        values = evidence[:, :count].numpy().astype(np.float32)
        crop_values = crops[:, :count].numpy().astype(np.float32)
        with torch.inference_mode():
            torch_values = torch.from_numpy(values)
            torch_crops = torch.from_numpy(crop_values)
            expected = model(torch_values, torch_crops).numpy()
            parent = model.role_parent(torch_values, torch_crops).numpy()
        actual = np.asarray(session.run(None, {
            "proposal_evidence": values,
            "proposal_crops": crop_values,
        })[0], dtype=np.float32)
        assert actual.shape == expected.shape == (1, count, 2 + len(ROLE_ORDER))
        assert float(np.max(np.abs(expected - actual))) <= 1e-5
        assert np.array_equal(expected[:, :, 2:], parent[:, :, 2:])


def test_objective_penalizes_both_error_sides_and_scene_overlap() -> None:
    config = protocol_configuration()["candidate_p1"]
    targets = torch.tensor([1, 0])
    weights = torch.ones(2)
    good_logits = torch.tensor([[-3.0, 3.0], [3.0, -3.0]])
    bad_logits = torch.zeros_like(good_logits)
    good_total, good = _proposal_objective(good_logits, targets, weights, config)
    bad_total, bad = _proposal_objective(bad_logits, targets, weights, config)
    assert float(good["positive_floor"]) == 0.0
    assert float(good["negative_ceiling"]) == 0.0
    assert float(good["scene_separation"]) == 0.0
    assert float(bad["positive_floor"]) > 0.0
    assert float(bad["negative_ceiling"]) > 0.0
    assert float(bad["scene_separation"]) > 0.0
    assert float(bad_total) > float(good_total)


def test_p2_freezes_p1_features_and_targets_bounded_hard_negative_margins() -> None:
    parent = FrozenRoleAxialTopologyProposalNet(seed=2608182601)
    model = FrozenP1BoundedMarginProposalNet(seed=2608182602)
    model.load_p1_state_dict(parent.state_dict())
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.startswith("proposal_head.") for name in trainable_names)
    model.train()
    assert model.proposal_head.training is True
    assert model.role_parent.training is False
    assert model.crop_stem.training is False
    assert model.crop_projection.training is False
    assert model.evidence_projection.training is False

    config = json.loads(
        (REPO_ROOT / "ml/ocr/scene_topology_proposal_v26/training/p2.json").read_text(
            encoding="utf-8"
        )
    )
    targets = torch.tensor([1, 0, 0, 0])
    weights = torch.ones(2)
    bounded = torch.tensor([
        [-0.75, 0.75],
        [0.75, -0.75],
        [0.75, -0.75],
        [0.75, -0.75],
    ])
    unsafe = bounded.clone()
    unsafe[1] = torch.tensor([-1.0, 1.0])
    bounded_total, bounded_parts = _p2_proposal_objective(
        bounded, targets, weights, config,
    )
    unsafe_total, unsafe_parts = _p2_proposal_objective(
        unsafe, targets, weights, config,
    )
    assert float(bounded_parts["positive_floor"]) == 0.0
    assert float(bounded_parts["negative_ceiling"]) == 0.0
    assert float(bounded_parts["hard_negative"]) == 0.0
    assert float(unsafe_parts["negative_ceiling"]) > 0.0
    assert float(unsafe_parts["hard_negative"]) > 0.0
    assert float(unsafe_total) > float(bounded_total)


def test_p3_freezes_p2_except_final_layer_and_uses_basic_cpu_ort(tmp_path: Path) -> None:
    parent = FrozenP1BoundedMarginProposalNet(seed=2608182602)
    model = FrozenP2FinalTailProposalNet(seed=2608182603)
    model.load_p2_state_dict(parent.state_dict())
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    assert trainable_names == ["proposal_head.5.bias", "proposal_head.5.weight"]
    model.train()
    assert model.proposal_head[5].training is True
    assert model.proposal_head[2].training is False
    assert model.role_parent.training is False
    assert model.crop_stem.training is False
    assert model.crop_projection.training is False
    assert model.evidence_projection.training is False

    config = json.loads(
        (REPO_ROOT / "ml/ocr/scene_topology_proposal_v26/training/p3.json").read_text(
            encoding="utf-8"
        )
    )
    targets = torch.tensor([1, 0, 0, 0])
    weights = torch.ones(2)
    bounded = torch.tensor([
        [-1.5, 1.5],
        [2.0, -2.0],
        [2.0, -2.0],
        [2.0, -2.0],
    ])
    unsafe = bounded.clone()
    unsafe[1] = torch.tensor([-1.0, 1.0])
    bounded_total, bounded_parts = _p3_proposal_objective(
        bounded, targets, weights, config,
    )
    unsafe_total, unsafe_parts = _p3_proposal_objective(
        unsafe, targets, weights, config,
    )
    assert float(bounded_parts["positive_floor"]) == 0.0
    assert float(bounded_parts["negative_ceiling"]) == 0.0
    assert float(bounded_parts["hard_negative"]) == 0.0
    assert float(bounded_parts["scene_separation"]) == 0.0
    assert float(unsafe_parts["negative_ceiling"]) > 0.0
    assert float(unsafe_parts["hard_negative"]) > 0.0
    assert float(unsafe_parts["scene_separation"]) > 0.0
    assert float(unsafe_total) > float(bounded_total)

    model.eval()
    evidence = torch.linspace(-1.0, 1.0, 5 * FEATURE_COUNT).reshape(
        1, 5, FEATURE_COUNT,
    )
    crops = torch.linspace(-1.0, 1.0, 5 * 2 * 32 * 128).reshape(
        1, 5, 2, 32, 128,
    )
    path = tmp_path / "scene-topology-v26-p3.onnx"
    torch.onnx.export(
        model,
        (evidence, crops),
        path,
        input_names=["proposal_evidence", "proposal_crops"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )
    session = _cpu_session_basic(path)
    assert session.get_session_options().graph_optimization_level == (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    values = evidence.numpy().astype(np.float32)
    crop_values = crops.numpy().astype(np.float32)
    with torch.inference_mode():
        expected = model(evidence, crops).numpy()
    actual = np.asarray(session.run(None, {
        "proposal_evidence": values,
        "proposal_crops": crop_values,
    })[0], dtype=np.float32)
    assert float(np.max(np.abs(expected - actual))) <= 1e-5


def test_public_gate_requires_a_selected_three_threshold_window() -> None:
    assert GATE_CONFIG["evaluation_limit"] == 1
    assert GATE_CONFIG["case_level_failure_analysis_permitted"] is False
    selected = {
        "selected_threshold": 0.45,
        "passing_threshold_window": [0.35, 0.45, 0.55],
    }
    assert _public_window(selected) == (0.35, 0.45, 0.55)
    try:
        _public_window({"selected_threshold": 0.45, "passing_threshold_window": []})
    except RuntimeError as error:
        assert "no preregistered robust threshold window" in str(error)
    else:
        raise AssertionError("V26 public gate accepted a candidate without a robust window")
