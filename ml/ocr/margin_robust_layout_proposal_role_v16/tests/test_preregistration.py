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

from ml.markers.gate_seal import canonical_json_bytes, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import box_iou
from ml.ocr.layout_conditioned_proposal_role_v15.dataset import render_scene as render_v15_scene
from ml.ocr.margin_robust_layout_proposal_role_v16.dataset import encode_proposal, proposals, render_scene
from ml.ocr.margin_robust_layout_proposal_role_v16.model import MarginRobustLayoutProposalRoleNet
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
    assert entry["status"] == "design_preregistered_before_split_materialization"
    assert entry["execution_authorized"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == []
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
