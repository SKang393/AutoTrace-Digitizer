# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed design checks for OCR structural-veto V17."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.margin_robust_layout_proposal_role_v16.train_p1 import _export
from ml.ocr.structural_veto_proposal_role_v17.dataset import encode_proposal, proposals, render_scene
from ml.ocr.structural_veto_proposal_role_v17.model import StructuralVetoProposalRoleNet
from ml.ocr.structural_veto_proposal_role_v17.sealed_gate import EVALUATOR_SOURCE_PATHS
from ml.ocr.structural_veto_proposal_role_v17.train_p1 import RUNNER_SOURCE_PATHS
from ml.ocr.structural_veto_proposal_role_v17.protocol import (
    BASE_CHECKPOINT_SHA256,
    EXPERIMENT_BUDGET,
    SPLITS,
    THRESHOLDS,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
V17_ROOT = ROOT / "ml/ocr/structural_veto_proposal_role_v17"


def test_protocol_is_canonical_fresh_and_execution_blocked() -> None:
    expected = protocol_configuration()
    assert (V17_ROOT / "PROTOCOL.json").read_bytes() == canonical_json_bytes(expected)
    assert expected["state"] == "design_preregistered_before_stored_split_materialization"
    assert expected["currently_preregistered_candidate"] is None
    assert expected["execution_authorized"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    trigger = expected["trigger_evidence"]
    assert trigger["result_sha256"] == TRIGGER_RESULT_SHA256
    assert trigger["case_level_details_used"] is False
    assert trigger["fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert trigger["consumed_v16_candidate_or_gate_rerun_authorized"] is False


def test_trigger_and_base_checkpoint_are_exact_and_distinct() -> None:
    trigger_path = ROOT / protocol_configuration()["trigger_evidence"]["result_path"]
    base_path = ROOT / protocol_configuration()["base_checkpoint"]["path"]
    assert sha256_file(trigger_path) == TRIGGER_RESULT_SHA256
    if base_path.is_file():
        assert sha256_file(base_path) == BASE_CHECKPOINT_SHA256
    base = protocol_configuration()["base_checkpoint"]
    assert base["all_base_parameters_frozen"] is True
    assert base["role_order_and_argmax_preserved"] is True
    veto = protocol_configuration()["veto_branch"]
    assert veto["role_logits_modified"] is False


def test_fresh_splits_and_zero_error_threshold_margin_are_fixed() -> None:
    assert [item.scene_count for item in SPLITS] == [720, 216, 288]
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    expected = protocol_configuration()
    assert expected["selection_thresholds"] == list(THRESHOLDS)
    assert expected["selection_gates"]["minimum_consecutive_passing_thresholds"] == 3
    assert expected["selection_gates"]["false_regions"] == 0
    assert expected["selection_gates"]["missed_regions"] == 0
    assert expected["selection_gates"]["prohibited_structure_hits"] == 0
    assert expected["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert expected["split_policy"]["v16_fixture_bytes_scene_truth_or_case_identity_reused"] is False
    assert expected["training"]["validation_or_public_pixels_used"] is False
    assert "no Chandler" in expected["data_scope"]


def test_materialized_split_is_source_bound_and_execution_blocked() -> None:
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol_configuration()["revision"])
    selection_path = V17_ROOT / "SELECTION_MANIFEST.json"
    seal_path = V17_ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    gate_path = V17_ROOT / "gates/sealed-public-v1.json"
    config_path = V17_ROOT / "training/p1.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert entry["status"] == "split_materialized_candidate1_preregistered_execution_blocked"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["split_materialized"] is True
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    assert sha256_file(selection_path) == entry["selection_manifest_sha256"]
    assert sha256_file(seal_path) == entry["sealed_public_test_seal_sha256"]
    assert sha256_file(gate_path) == entry["public_gate_config_sha256"]
    assert sha256_file(config_path) == entry["candidate_config_sha256"]["P1"]
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS) == gate["expected_evaluator_source_bundle_sha256"]
    assert selection["train"]["scene_count"] == 720
    assert selection["validation"]["scene_count"] == 216
    assert seal["scene_count"] == 288
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert config["expected_optimizer_steps"] == 1608
    assert config["public_gate_archive_opened"] is False
    assert sha256_file(ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert sha256_file(ROOT / seal["private_manifest_path"]) == seal["private_manifest_sha256"]
    assert not (V17_ROOT / "artifacts/P1-run").exists()
    assert not (
        ROOT / "ml/markers/training-seals/ocr-detection/graph-text-structural-veto-proposal-role-v17"
    ).exists()


def test_fresh_renderer_has_one_production_proposal_per_truth() -> None:
    for split, index in (("train", 0), ("validation", 215), ("sealed_public", 287)):
        scene = render_scene(split, index)
        candidates = proposals(scene.raster)
        assert scene.scene_id.startswith(f"structural-veto-proposal-role-v17-{split}-")
        assert len(scene.truths) == 8
        for truth in scene.truths:
            assert sum(box_iou(candidate.box, truth.box) >= 0.5 for candidate in candidates) == 1


def test_veto_model_modifies_only_positive_proposal_logit_and_exports(tmp_path: Path) -> None:
    scene = render_scene("train", 0)
    values = np.stack([
        encode_proposal(scene.raster, candidate, scene.plot)
        for candidate in proposals(scene.raster)[:3]
    ]).astype(np.float32)
    model = StructuralVetoProposalRoleNet().eval()
    tensor = torch.from_numpy(values)
    with torch.inference_mode():
        base = model.base(tensor) * model.base_output_scale
        output = model(tensor)
    assert output.shape == (3, 10)
    assert torch.equal(output[:, :1], base[:, :1])
    assert torch.equal(output[:, 2:], base[:, 2:])
    assert torch.all(output[:, 1] <= base[:, 1])
    assert all(parameter.requires_grad is False for parameter in model.base.parameters())
    assert all(parameter.requires_grad is True for parameter in model.trainable_parameters())
    path = tmp_path / "v17-p1.onnx"
    _export(model, tensor, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {"region_proposals": values})[0], dtype=np.float32)
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert float(np.max(np.abs(output.numpy() - actual))) <= 1e-5
