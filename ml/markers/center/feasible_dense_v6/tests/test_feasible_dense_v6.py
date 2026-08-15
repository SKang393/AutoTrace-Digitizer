# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration tests for feasible dense marker V6."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import ml.markers.center.feasible_dense_v6.dataset as dataset_module
import ml.markers.center.feasible_dense_v6.public_gate as public_gate_module
import ml.markers.center.feasible_dense_v6.public_gate_v2 as public_gate_v2_module

from ml.markers.center.feasible_dense_v6.dataset import (
    HEIGHT,
    PROHIBITED_KINDS,
    PUBLIC_SCENE_COUNT,
    REQUIRED_DISJOINT_CLEARANCE,
    TRAIN_SCENE_COUNT,
    VALIDATION_SCENE_COUNT,
    WIDTH,
    validate_scene_feasibility,
)
from ml.markers.center.feasible_dense_v6.model import create_model
from ml.markers.center.feasible_dense_v6.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    GATE_CONFIGURATION,
)
from ml.markers.center.feasible_dense_v6.train_p1 import RUNNER_SOURCE_PATHS, THRESHOLDS
from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/feasible_dense_v6"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_archives_and_feasibility_metadata_are_exact() -> None:
    selection = _json(ROOT / "SELECTION_MANIFEST.json")
    freeze = _json(ROOT / "SPLIT_FREEZE_REPORT.json")
    split_seal = _json(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    dataset = _json(ROOT / "PUBLIC_DATASET_MANIFEST.json")

    assert selection["train"]["count"] == TRAIN_SCENE_COUNT == 192
    assert selection["validation"]["count"] == VALIDATION_SCENE_COUNT == 48
    assert selection["sealed_public"]["count"] == PUBLIC_SCENE_COUNT == 64
    assert len({selection[name]["renderer_family"] for name in ("train", "validation", "sealed_public")}) == 3
    assert len({selection[name]["degradation_family"] for name in ("train", "validation", "sealed_public")}) == 3

    for name in ("train", "validation"):
        archive_path = REPO_ROOT / selection[name]["archive_path"]
        assert sha256_file(archive_path) == selection[name]["archive_sha256"]
        feasibility = selection[name]["feasibility"]
        assert feasibility["truth_hard_acceptance_overlap_count"] == 0
        assert feasibility["truth_mask_conflict_count"] == 0
        assert feasibility["hard_point_missing_from_artifact_truth_count"] == 0
        assert feasibility["minimum_truth_to_hard_negative_distance_px"] > REQUIRED_DISJOINT_CLEARANCE

    public_archive_path = REPO_ROOT / split_seal["fixture_archive_path"]
    assert sha256_file(public_archive_path) == split_seal["fixture_archive_sha256"]
    assert sha256_file(ROOT / "PUBLIC_DATASET_MANIFEST.json") == split_seal["dataset_manifest_sha256"]
    assert split_seal["public_gate_archive_opened"] is False
    assert split_seal["public_gate_evaluations"] == 0
    assert freeze["sealed_public_feasibility"]["truth_hard_acceptance_overlap_count"] == 0
    assert freeze["sealed_public_feasibility"]["minimum_truth_to_hard_negative_distance_px"] > REQUIRED_DISJOINT_CLEARANCE
    assert len(dataset["fixtures"]) == PUBLIC_SCENE_COUNT
    assert len({item["fixture_id"] for item in dataset["fixtures"]}) == PUBLIC_SCENE_COUNT


def test_fresh_visible_scenes_satisfy_dense_contract_and_feasibility() -> None:
    for split in ("train", "validation"):
        scene = dataset_module._draw_scene(split, 0)
        validate_scene_feasibility(scene)
        assert scene.tensor.shape == (3, HEIGHT, WIDTH)
        assert scene.center_target.shape == (1, HEIGHT, WIDTH)
        assert scene.radius_target.shape == (1, HEIGHT, WIDTH)
        assert scene.artifact_target.shape == (1, HEIGHT, WIDTH)
        assert np.isfinite(scene.tensor).all()
        assert set(kind for kind, _, _ in scene.hard_negatives) == set(PROHIBITED_KINDS)
        assert scene.scene_id.startswith(f"marker-feasible-dense-v6-{split}-")


def test_infeasible_truth_and_prohibited_region_is_rejected() -> None:
    scene = dataset_module._draw_scene("validation", 0)
    _, hard_x, hard_y = scene.hard_negatives[0]
    conflicted = replace(
        scene,
        centers=((hard_x, hard_y, scene.centers[0][2]),) + scene.centers[1:],
    )
    with pytest.raises(RuntimeError, match="acceptance regions overlap"):
        validate_scene_feasibility(conflicted)


def test_model_preserves_frozen_dense_three_head_contract() -> None:
    model = create_model().eval()
    value = torch.zeros((1, 3, HEIGHT, WIDTH), dtype=torch.float32)
    with torch.inference_mode():
        output = model(value)
    assert output.shape == value.shape
    assert torch.all((output[:, 0] >= 0) & (output[:, 0] <= 1))
    assert torch.all(output[:, 1] >= 1)
    assert torch.all((output[:, 2] >= 0) & (output[:, 2] <= 1))
    assert tuple(model.contract.input_channels) == ("ink_probability", "text_mask", "artifact_mask")
    assert tuple(model.contract.output_channels) == (
        "center_probability",
        "radius_pixels",
        "artifact_probability",
    )


def test_source_protocol_and_public_gate_bindings_are_exact() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    config = _json(ROOT / "training/p1.json")
    gate = _json(ROOT / "gates/sealed-public-v1.json")
    gate_v2 = _json(ROOT / "gates/sealed-public-v2.json")
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, public_gate_module.EVALUATOR_SOURCE_PATHS) == gate[
        "expected_evaluator_source_bundle_sha256"
    ]
    assert sha256_bytes(canonical_json_bytes(GATE_CONFIGURATION)) == gate["expected_gate_config_sha256"]
    assert sha256_file(ROOT / "training/p1.json") == protocol["candidate_config_sha256"]
    assert sha256_file(ROOT / "gates/sealed-public-v1.json") == protocol["public_gate_config_sha256"]
    assert source_bundle_sha256(REPO_ROOT, public_gate_v2_module.EVALUATOR_SOURCE_PATHS) == gate_v2[
        "expected_evaluator_source_bundle_sha256"
    ]
    assert sha256_bytes(canonical_json_bytes(GATE_CONFIGURATION)) == gate_v2["expected_gate_config_sha256"]
    assert sha256_file(ROOT / "gates/sealed-public-v2.json") == protocol["public_gate_v2_config_sha256"]
    assert gate_v2["task"] == "marker-center"
    assert gate_v2["revision"] == public_gate_v2_module.REVISION
    assert gate_v2["expected_candidate_hash_keys"] == ["candidate_report_sha256", "onnx_sha256"]
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert config["selection_thresholds"] == list(THRESHOLDS)
    assert protocol["execution_authorized"] is False
    assert protocol["p1_selection_gate_passed"] is True
    assert protocol["p1_result_sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert protocol["public_gate_authorized"] is False
    assert protocol["public_gate_v1_attempt_consumed"] is True
    assert protocol["public_gate_v1_rerun_authorized"] is False
    assert protocol["public_gate_v2_preregistered"] is True
    assert protocol["public_gate_archive_opened"] is False
    assert protocol["public_gate_evaluations"] == 0


def test_canonical_budget_records_selected_p1_and_sealed_public_gate() -> None:
    ledger = _json(LEDGER_PATH)
    entry = next(item for item in ledger["revisions"] if item["revision"] == "marker-center-feasible-dense-v6")
    assert entry["status"] == "candidate_1_selected_public_v2_preregistered"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["p1_expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    result = _json(ROOT / "P1_RESULT.json")
    assert sha256_file(ROOT / "P1_RESULT.json") == entry["p1_result_sha256"]
    assert result["selection_gate_passed"] is True
    assert result["selection_exact_scene_count"] == result["selection_scene_count"] == 48
    assert result["selection_true_positives"] == 432
    assert result["selection_false_positives"] == 0
    assert result["selection_false_negatives"] == 0
    assert result["selection_duplicate_count"] == 0
    assert sum(result["prohibited_structure_hits"].values()) == 0
    assert result["onnx_parity_passed"] is True
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    attempt = _json(ROOT / "PUBLIC_GATE_V1_ATTEMPT.json")
    assert sha256_file(ROOT / "PUBLIC_GATE_V1_ATTEMPT.json") == entry["public_gate_v1_attempt_sha256"]
    assert attempt["status"] == "failed_preseal_configuration_identity"
    assert attempt["public_archive_parsed"] is False
    assert attempt["gate_opened_seal_created"] is False
    assert attempt["public_gate_evaluations"] == 0
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] is None
    assert entry["public_gate_v1_rerun_authorized"] is False
    assert entry["public_gate_v2_preregistered"] is True
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["manifest_created"] is False
    assert entry["model_store_promoted"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_public_runner_refuses_nonpassing_candidate_before_archive_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate-report.json"
    candidate_path.write_bytes(canonical_json_bytes({"candidate_id": "P1", "selection_gate_passed": False}))
    monkeypatch.setattr(
        public_gate_module,
        "read_archive",
        lambda *_args, **_kwargs: pytest.fail("public archive was read before authorization"),
    )
    with pytest.raises(RuntimeError, match="passing visible-selection candidate"):
        public_gate_module.run(candidate_path, tmp_path / "public-report.json")
    assert not (tmp_path / "public-report.json").exists()


def test_public_v2_runner_refuses_before_seal_or_archive_when_unauthorized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate-report.json"
    candidate_path.write_bytes(canonical_json_bytes({"candidate_id": "P1", "selection_gate_passed": True}))
    monkeypatch.setattr(
        public_gate_v2_module,
        "_run_opened_gate",
        lambda *_args, **_kwargs: pytest.fail("public V2 evaluator ran before authorization"),
    )
    with pytest.raises(RuntimeError, match="not separately authorized"):
        public_gate_v2_module.run(candidate_path, tmp_path / "public-v2-report.json")
    assert not (tmp_path / "public-v2-report.json").exists()
