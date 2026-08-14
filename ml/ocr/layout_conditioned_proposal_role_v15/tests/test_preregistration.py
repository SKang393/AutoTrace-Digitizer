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
from ml.ocr.layout_conditioned_proposal_role_v15.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.layout_conditioned_proposal_role_v15.train_p1 import RUNNER_SOURCE_PATHS, _export
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


def test_protocol_is_canonical_fail_closed_and_aggregate_only() -> None:
    expected = protocol_configuration()
    assert (V15_ROOT / "PROTOCOL.json").read_bytes() == canonical_json_bytes(expected)
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
    assert protocol["split_policy"]["public_case_level_failure_analysis_permitted"] is False
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


def test_canonical_ledger_authorizes_only_single_use_p1() -> None:
    ledger = json.loads(
        (ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item.get("revision") == "graph-text-layout-conditioned-proposal-role-v15"
    )
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["protocol_sha256"] == sha256_file(V15_ROOT / "PROTOCOL.json")
    assert entry["split_generator_source_bundle_sha256"] == "502d3fefa949acc1b755871005fcf66824ba07ee04b1c9515d9d9874e62ff3e5"
    assert entry["p1_expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)
    assert entry["expected_public_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert entry["selection_manifest_sha256"] == sha256_file(V15_ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(
        V15_ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    )
    assert entry["public_gate_config_sha256"] == sha256_file(
        V15_ROOT / "gates/sealed-public-v1.json"
    )
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(V15_ROOT / "training/p1.json")
    assert entry["split_materialized"] is True
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
