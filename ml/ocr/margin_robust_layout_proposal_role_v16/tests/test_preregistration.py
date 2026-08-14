# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed design preregistration checks for OCR V16."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.layout_conditioned_proposal_role_v15.dataset import render_scene as render_v15_scene
from ml.ocr.margin_robust_layout_proposal_role_v16.dataset import encode_proposal, proposals, render_scene
from ml.ocr.margin_robust_layout_proposal_role_v16.model import MarginRobustLayoutProposalRoleNet
from ml.ocr.margin_robust_layout_proposal_role_v16.model_p2 import CalibratedMarginCandidate
from ml.ocr.margin_robust_layout_proposal_role_v16.model_p3 import OutputScaledMarginCandidate
from ml.ocr.margin_robust_layout_proposal_role_v16.protocol import (
    ENCODED_WIDTH,
    EXPERIMENT_BUDGET,
    ROBUST_THRESHOLD_RUN_LENGTH,
    SPLITS,
    THRESHOLDS,
    protocol_configuration,
)
from ml.ocr.margin_robust_layout_proposal_role_v16.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.margin_robust_layout_proposal_role_v16.train_p1 import (
    RUNNER_SOURCE_PATHS,
    _export,
    _select_robust_window,
)
from ml.ocr.margin_robust_layout_proposal_role_v16.train_p2 import (
    P1_RESULT_PATH,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
)
from ml.ocr.margin_robust_layout_proposal_role_v16.train_p3 import (
    P2_RESULT_PATH,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
    _hard_negative_indices,
)


ROOT = Path(__file__).resolve().parents[4]
V16_ROOT = ROOT / "ml/ocr/margin_robust_layout_proposal_role_v16"


def _metrics(*, passing: bool) -> dict[str, object]:
    return {
        "scene_count": 2,
        "truth_region_count": 16,
        "exact_scene_count": 2 if passing else 1,
        "true_positives": 16 if passing else 15,
        "false_positives": 0,
        "false_negatives": 0 if passing else 1,
        "duplicate_region_count": 0,
        "prohibited_structure_hits": 0,
        "role_accuracy": 1.0,
        "per_role_accuracy": {role: 1.0 for role in protocol_configuration()["output_contract"]["role_order"]},
    }


def test_protocol_is_canonical_fail_closed_and_aggregate_only() -> None:
    expected = protocol_configuration()
    assert (V16_ROOT / "PROTOCOL.json").read_bytes() == canonical_json_bytes(expected)
    assert expected["state"] == "design_preregistered_before_stored_split_materialization"
    assert expected["currently_preregistered_candidate"] is None
    assert expected["execution_authorized"] is False
    assert expected["production_approval"] is False
    assert expected["release_eligible"] is False
    assert expected["experiment_budget"] == EXPERIMENT_BUDGET == 3
    trigger = expected["trigger_evidence"]
    assert trigger["report_sha256"] == "8bd7170db115f6fccbfc9b998bd5f6fce0d8ae001469b692fa07e8392068553d"
    assert trigger["evidence_scope_used_for_v16_design"] == "aggregate metrics only"
    assert trigger["v15_public_fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert trigger["case_level_details_emitted"] is False
    assert trigger["consumed_v15_candidate_or_gate_rerun_authorized"] is False


def test_fresh_split_registrations_and_threshold_margin_are_frozen() -> None:
    assert [item.scene_count for item in SPLITS] == [640, 192, 256]
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    expected = protocol_configuration()
    assert expected["training"]["proposal_margin"] == 1.2
    assert expected["training"]["proposal_margin_loss_weight"] == 0.5
    assert expected["selection_thresholds"] == list(THRESHOLDS)
    assert expected["selection_gates"]["minimum_consecutive_passing_thresholds"] == 3
    assert expected["split_policy"]["predecessor_fixture_bytes_reused"] is False
    assert expected["split_policy"]["v15_public_fixture_bytes_scene_truth_or_case_identity_reused"] is False
    assert expected["split_policy"]["public_case_level_failure_analysis_permitted"] is False
    assert "no Chandler" in expected["data_scope"]


@pytest.mark.parametrize("split,index", [
    ("train", 0), ("train", 639),
    ("validation", 0), ("validation", 191),
    ("sealed_public", 0), ("sealed_public", 255),
])
def test_source_renderer_has_one_production_proposal_per_truth(split: str, index: int) -> None:
    scene = render_scene(split, index)
    candidates = proposals(scene.raster)
    assert len(scene.truths) == 8
    assert scene.scene_id.startswith("margin-robust-layout-proposal-role-v16-")
    for truth in scene.truths:
        assert sum(box_iou(candidate.box, truth.box) >= 0.5 for candidate in candidates) == 1


def test_v16_fixture_bytes_and_id_are_fresh_from_v15() -> None:
    current = render_scene("train", 0)
    prior = render_v15_scene("train", 0)
    assert current.scene_id != prior.scene_id
    assert current.renderer_family != prior.renderer_family
    assert current.degradation_family != prior.degradation_family
    assert not np.array_equal(current.raster, prior.raster)


def test_margin_model_uses_layout_for_proposal_logits_and_exports_cpu_onnx(tmp_path: Path) -> None:
    scene = render_scene("train", 0)
    candidate = proposals(scene.raster)[0]
    value = encode_proposal(scene.raster, candidate, scene.plot)
    assert value.shape == (2, 32, ENCODED_WIDTH) == (2, 32, 152)
    altered = value.copy()
    altered[:, :, 144:] += 0.25
    model = MarginRobustLayoutProposalRoleNet().eval()
    with torch.inference_mode():
        outputs = model(torch.from_numpy(np.stack((value, altered))))
    assert outputs.shape == (2, 10)
    assert torch.isfinite(outputs).all()
    assert not torch.equal(outputs[0, :2], outputs[1, :2])
    path = tmp_path / "v16-p1.onnx"
    _export(model, torch.from_numpy(np.stack((value, altered))), path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {"region_proposals": np.stack((value, altered))})[0])
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert actual.shape == (2, 10)
    assert float(np.max(np.abs(outputs.numpy() - actual))) <= 1e-5


def test_robust_selection_requires_three_adjacent_passing_thresholds() -> None:
    comparisons = [
        {"threshold": threshold, "metrics": _metrics(passing=index in {1, 2, 3})}
        for index, threshold in enumerate(THRESHOLDS)
    ]
    selected = _select_robust_window(comparisons)
    assert selected is not None
    assert selected[0]["threshold"] == THRESHOLDS[2]
    assert selected[1] == THRESHOLDS[1:4]
    comparisons[3] = {"threshold": THRESHOLDS[3], "metrics": _metrics(passing=False)}
    assert _select_robust_window(comparisons) is None


def test_runner_public_sources_and_ledger_are_fail_closed() -> None:
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)
    assert source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS)
    assert GATE_CONFIG["evaluation_limit"] == 1
    assert GATE_CONFIG["minimum_consecutive_thresholds"] == ROBUST_THRESHOLD_RUN_LENGTH
    assert GATE_CONFIG["case_level_failure_analysis_permitted"] is False
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol_configuration()["revision"])
    assert entry["status"] == "candidate_3_preregistered"
    assert entry["split_materialized"] is True
    assert entry["selection_manifest_sha256"] == "06253f5a0a7318fde69093027d99c4ef1cacf876e9906cc6c33fc0a9d15be72f"
    assert entry["sealed_public_fixture_archive_sha256"] == "663b9a0c1600ca65c04c2acf85a021a057777a44f7e930ce82fc6beb4b7a97c1"
    assert entry["sealed_public_private_manifest_sha256"] == "cd6dc64bff9fc3fb8d1ba6a6185e142b661fdd10fc215bf9064dc7ef438163e9"
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P3"
    assert entry["public_gate_authorized"] is False
    assert entry["preregistered_candidate_ids"] == ["P3"]
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_materialized_split_and_p1_configuration_remain_closed() -> None:
    selection_path = V16_ROOT / "SELECTION_MANIFEST.json"
    seal_path = V16_ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    gate_path = V16_ROOT / "gates/sealed-public-v1.json"
    config_path = V16_ROOT / "training/p1.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert selection["train"]["scene_count"] == 640
    assert selection["train"]["split_fingerprint"] == "816a09f397aa984b6e13ec26133f104dd04087c00a800d55230523db52ea8f70"
    assert selection["validation"]["scene_count"] == 192
    assert selection["validation"]["split_fingerprint"] == "f9cabe4af3428ec0d5dddd3eed52d95cd9e4c6dc5edd2514523338b4ffa1951c"
    assert selection["training_evidence"]["v15_public_fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert seal["scene_count"] == 256
    assert seal["split_fingerprint"] == "e1ec6d2272e83126474f0a09cc5f8c566444dab68fb816899cca2d932ee6dbde"
    assert seal["fixture_archive_sha256"] == "663b9a0c1600ca65c04c2acf85a021a057777a44f7e930ce82fc6beb4b7a97c1"
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert gate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS)
    assert config["candidate_id"] == "P1"
    assert config["selection_manifest_sha256"] == sha256_file(selection_path)
    assert config["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)
    assert config["minimum_consecutive_passing_thresholds"] == 3
    assert config["public_gate_archive_opened"] is False
    assert config["public_gate_evaluations"] == 0
    for value in (selection, seal, gate, config):
        assert value["production_approval"] is False
        assert value["release_eligible"] is False


def test_p1_result_and_p2_calibration_are_checksum_bound_and_closed() -> None:
    result_path = ROOT / P1_RESULT_PATH
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config_path = V16_ROOT / "training/p2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert result_path.read_bytes() == canonical_json_bytes(result)
    assert result["status"] == "failed_selection"
    assert result["consumed"] is True
    assert result["selection_metrics"]["true_positives"] == 1536
    assert result["selection_metrics"]["false_positives"] == 5
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["passing_threshold_window"] == []
    assert result["onnx_parity_passed"] is True
    assert result["public_gate_archive_opened"] is False
    assert config_path.read_bytes() == canonical_json_bytes(config)
    assert config["candidate_id"] == "P2"
    assert config["p1_result_sha256"] == sha256_file(result_path)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        ROOT, P2_RUNNER_SOURCE_PATHS
    )
    assert config["expected_optimizer_steps"] == 0
    assert config["positive_logit_bias"] == -2.0
    assert config["p1_aggregate_metrics_only_used_for_design"] is True
    assert config["p1_validation_case_detail_or_pixels_used_for_design"] is False
    assert config["public_gate_archive_opened"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False

    base = MarginRobustLayoutProposalRoleNet().eval()
    calibrated = CalibratedMarginCandidate(base, positive_logit_bias=-2.0).eval()
    values = torch.rand((3, 2, 32, 152), generator=torch.Generator().manual_seed(20262180))
    with torch.inference_mode():
        before = base(values)
        after = calibrated(values)
    assert torch.equal(before[:, :1], after[:, :1])
    assert torch.equal(before[:, 2:], after[:, 2:])
    assert torch.allclose(before[:, 1] - 2.0, after[:, 1], atol=0.0, rtol=0.0)


def test_p2_result_and_p3_training_repair_are_checksum_bound_and_closed() -> None:
    result_path = ROOT / P2_RESULT_PATH
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config_path = V16_ROOT / "training/p3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert result_path.read_bytes() == canonical_json_bytes(result)
    assert result["status"] == "failed_selection"
    assert result["consumed"] is True
    assert result["selection_metrics"]["true_positives"] == 1536
    assert result["selection_metrics"]["false_positives"] == 4
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["passing_threshold_window"] == []
    assert result["onnx_parity_passed"] is False
    assert result["public_gate_archive_opened"] is False
    assert config_path.read_bytes() == canonical_json_bytes(config)
    assert config["candidate_id"] == "P3"
    assert config["p1_result_sha256"] == sha256_file(ROOT / P1_RESULT_PATH)
    assert config["p2_result_sha256"] == sha256_file(result_path)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        ROOT, P3_RUNNER_SOURCE_PATHS
    )
    assert config["expected_hard_negative_count"] == 10240
    assert config["expected_optimizer_steps"] == 480
    assert config["hard_negative_multiplier"] == 2
    assert config["output_scale"] == 0.5
    assert config["p1_p2_aggregate_metrics_only_used_for_design"] is True
    assert config["p1_p2_validation_case_detail_or_pixels_used_for_design"] is False
    assert config["validation_or_public_pixels_used"] is False
    assert config["public_gate_archive_opened"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False

    base = MarginRobustLayoutProposalRoleNet().eval()
    scaled = OutputScaledMarginCandidate(base, output_scale=0.5).eval()
    values = torch.rand((5, 2, 32, 152), generator=torch.Generator().manual_seed(20262181))
    with torch.inference_mode():
        before = base(values)
        after = scaled(values)
    assert torch.equal(torch.argmax(before[:, :2], dim=1), torch.argmax(after[:, :2], dim=1))
    assert torch.equal(torch.argmax(before[:, 2:], dim=1), torch.argmax(after[:, 2:], dim=1))
    assert torch.allclose(before * 0.5, after, atol=0.0, rtol=0.0)


def test_p3_hard_negative_mining_is_deterministic_and_training_only() -> None:
    class ScoreModel(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            score = value[:, 0, 0, 0]
            zeros = torch.zeros((len(value), 8), dtype=value.dtype)
            return torch.cat((-score[:, None], score[:, None], zeros), dim=1)

    values = torch.zeros((8, 2, 32, 152), dtype=torch.float32)
    values[:, 0, 0, 0] = torch.tensor((0.0, 0.1, 0.2, 0.3, 0.9, 0.5, 0.7, 0.8))
    labels = torch.tensor((1, 1, 0, 0, 0, 0, 0, 0), dtype=torch.int64)
    selected = _hard_negative_indices(
        ScoreModel(), values, labels, multiplier=2, batch_size=3,
    )
    assert selected.tolist() == [4, 7, 6, 5]
