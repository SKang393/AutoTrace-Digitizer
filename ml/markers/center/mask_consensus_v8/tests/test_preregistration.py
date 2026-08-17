# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration tests for marker-center V8."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import torch

import ml.markers.center.mask_consensus_v8.dataset as dataset_module
from ml.markers.center.mask_consensus_v8.dataset import (
    DEGRADATION_FAMILIES,
    HEIGHT,
    PROHIBITED_KINDS,
    PUBLIC_SCENE_COUNT,
    RENDERER_FAMILIES,
    TRAIN_SCENE_COUNT,
    VALIDATION_SCENE_COUNT,
    WIDTH,
)
from ml.markers.center.mask_consensus_v8.model import create_model
from ml.markers.center.mask_consensus_v8.protocol import DESIGN_SOURCE_PATHS, THRESHOLDS
from ml.markers.center.mask_consensus_v8.public_gate import EVALUATOR_SOURCE_PATHS
from ml.markers.center.mask_consensus_v8.train_p1 import RUNNER_SOURCE_PATHS as P1_RUNNER_SOURCE_PATHS
from ml.markers.center.mask_consensus_v8.train_p2 import RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS
from ml.markers.center.mask_consensus_v8.train_p3 import (
    ARTIFACT_POSITIVE_WEIGHT,
    EXPECTED_OPTIMIZER_STEPS,
    FIXED_RADIUS_PIXELS,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
    FixedRadiusInferenceModel,
    _photometric_batch,
)
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/mask_consensus_v8"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_exhausts_p3_after_fail_closed_parity_preflight() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    assert protocol["state"] == "exhausted_failed_runner_before_training"
    assert protocol["experiment_budget"] == 3
    assert protocol["preregistered_candidate_ids"] == []
    assert protocol["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert protocol["remaining_unregistered_candidate_ids"] == []
    assert protocol["selection_gates"]["selection_thresholds"] == list(THRESHOLDS)
    assert protocol["split_materialized"] is True
    assert protocol["public_gate_evaluator_preregistered"] is True
    assert protocol["preregistration_commit"] == "4e20674d0d7a15896005a066c2054753dbf5d7dd"
    assert protocol["preregistration_tree"] == "0e51075b8cd082b9ce48e0232fa008fee9e9627a"
    assert protocol["p2_preregistration_commit"] == "483dce39bae5c5285edc85469939f585e3618d4b"
    assert protocol["p2_preregistration_tree"] == "60ba3da228ee369dcc2b1b27c3b6e2306b2acd63"
    assert protocol["p3_preregistration_commit"] == "ac22709bae98c91d7480b93f455f4c6fa66da2cc"
    assert protocol["p3_preregistration_tree"] == "952a54df56044671d0439d4dfb5c713184c102f5"
    assert protocol["authorized_candidate_id"] is None
    assert protocol["execution_authorized"] is False
    assert protocol["p1_status"] == "failed_selection_consumed"
    assert protocol["p1_selection_exact_scene_count"] == 122
    assert protocol["p1_selection_false_positives"] == 6
    assert protocol["p1_selection_false_negatives"] == 23
    assert protocol["p1_case_detail_or_pixels_inspected"] is False
    assert protocol["p2_status"] == "failed_selection_consumed"
    assert protocol["p2_selection_exact_scene_count"] == 122
    assert protocol["p2_selection_false_positives"] == 6
    assert protocol["p2_selection_false_negatives"] == 23
    assert protocol["p2_case_detail_or_pixels_inspected"] is False
    assert protocol["p3_status"] == "failed_runner_consumed"
    assert protocol["p3_optimizer_steps"] == 0
    assert protocol["p3_failure_phase"] == "p2_parity_localization"
    assert protocol["p3_expected_parity_by_output_channel"] == [
        0.0000033080577850341797,
        0.00001621246337890625,
        0.000008970499038696289,
    ]
    assert protocol["p3_observed_parity_by_output_channel"] == [
        0.000003874301910400391,
        0.00001621246337890625,
        0.000008970499038696289,
    ]
    assert protocol["historical_gate_and_training_seal_file_count_after_p3"] == 286
    assert (
        protocol["historical_gate_and_training_seal_aggregate_sha256_after_p3"]
        == "d77220d7d1ad27b3930b4c8d516ea753e054fc42192804a60d9fe28d80f71fc4"
    )
    assert protocol["p3_case_detail_or_pixels_inspected"] is False
    assert protocol["public_gate_authorized"] is False
    assert protocol["public_gate_archive_opened"] is False
    assert protocol["public_gate_evaluations"] == 0
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_frozen_archives_and_source_bindings_match_exact_bytes() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    selection = _json(ROOT / "SELECTION_MANIFEST.json")
    freeze = _json(ROOT / "SPLIT_FREEZE_REPORT.json")
    public_seal = _json(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    p1_config = _json(ROOT / "training/p1.json")
    p2_config = _json(ROOT / "training/p2.json")
    p3_config = _json(ROOT / "training/p3.json")
    ledger = _json(LEDGER_PATH)
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol["revision"])
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SPLIT_FREEZE_REPORT.json") == protocol["split_freeze_report_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert sha256_file(ROOT / "PUBLIC_DATASET_MANIFEST.json") == protocol["public_dataset_manifest_sha256"]
    assert sha256_file(ROOT / "P1_RESULT.json") == protocol["p1_result_sha256"]
    assert sha256_file(ROOT / "P2_RESULT.json") == protocol["p2_result_sha256"]
    assert sha256_file(ROOT / "P3_RESULT.json") == protocol["p3_result_sha256"]
    assert sha256_file(ROOT / "training/p3.json") == protocol["candidate_config_sha256"]
    assert source_bundle_sha256(REPO_ROOT, DESIGN_SOURCE_PATHS) == protocol["design_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P1_RUNNER_SOURCE_PATHS) == p1_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS) == p2_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS) == p3_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS) == protocol["public_gate_evaluator_source_bundle_sha256"]
    for name in ("train", "validation", "sealed_public"):
        assert sha256_file(REPO_ROOT / selection[name]["archive_path"]) == selection[name]["archive_sha256"]
        assert selection[name]["feasibility"]["truth_hard_acceptance_overlap_count"] == 0
        assert selection[name]["feasibility"]["truth_mask_conflict_count"] == 0
        assert selection[name]["feasibility"]["hard_point_missing_from_artifact_truth_count"] == 0
    assert freeze["cross_split_source_overlap_count"] == 0
    assert freeze["model_execution_count_at_freeze"] == 0
    assert freeze["optimizer_step_count_at_freeze"] == 0
    assert public_seal["public_gate_archive_opened"] is False
    assert public_seal["public_gate_evaluations"] == 0
    assert entry["status"] == "exhausted_failed_runner"
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["preregistration_commit"] == "4e20674d0d7a15896005a066c2054753dbf5d7dd"
    assert entry["preregistration_tree"] == "0e51075b8cd082b9ce48e0232fa008fee9e9627a"
    assert entry["p2_preregistration_commit"] == "483dce39bae5c5285edc85469939f585e3618d4b"
    assert entry["p2_preregistration_tree"] == "60ba3da228ee369dcc2b1b27c3b6e2306b2acd63"
    assert entry["p3_preregistration_commit"] == "ac22709bae98c91d7480b93f455f4c6fa66da2cc"
    assert entry["p3_preregistration_tree"] == "952a54df56044671d0439d4dfb5c713184c102f5"
    assert entry["authorized_candidate_id"] is None
    assert entry["execution_authorized"] is False
    assert entry["public_gate_authorized"] is False


def test_p1_authorization_bound_sources_remain_unchanged_after_consumption() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    preregistration_commit = str(protocol["preregistration_commit"])
    tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", preregistration_commit],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    assert tree == protocol["preregistration_tree"]
    bound_paths = {
        Path("ml/markers/center/mask_consensus_v8/training/p1.json"),
        *P1_RUNNER_SOURCE_PATHS,
        *EVALUATOR_SOURCE_PATHS,
    }
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", preregistration_commit, "--", *(path.as_posix() for path in sorted(bound_paths))],
        cwd=REPO_ROOT,
        check=False,
    )
    assert unchanged.returncode == 0


def test_p2_authorization_binds_the_committed_preregistration_sources() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    commit = str(protocol["p2_preregistration_commit"])
    tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", commit],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    assert tree == protocol["p2_preregistration_tree"]
    bound_paths = {
        Path("ml/markers/center/mask_consensus_v8/P1_RESULT.json"),
        Path("ml/markers/center/mask_consensus_v8/training/p2.json"),
        *P2_RUNNER_SOURCE_PATHS,
    }
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", *(path.as_posix() for path in sorted(bound_paths))],
        cwd=REPO_ROOT,
        check=False,
    )
    assert unchanged.returncode == 0


def test_p3_authorization_binds_the_committed_preregistration_sources() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    commit = str(protocol["p3_preregistration_commit"])
    tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", commit],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    assert tree == protocol["p3_preregistration_tree"]
    bound_paths = {
        Path("ml/markers/center/mask_consensus_v8/P2_RESULT.json"),
        Path("ml/markers/center/mask_consensus_v8/training/p3.json"),
        *P3_RUNNER_SOURCE_PATHS,
    }
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", *(path.as_posix() for path in sorted(bound_paths))],
        cwd=REPO_ROOT,
        check=False,
    )
    assert unchanged.returncode == 0


def test_split_counts_and_family_names_are_preregistered_and_disjoint() -> None:
    assert TRAIN_SCENE_COUNT == 512
    assert VALIDATION_SCENE_COUNT == 128
    assert PUBLIC_SCENE_COUNT == 160
    renderer_sets = {name: set(values) for name, values in RENDERER_FAMILIES.items()}
    degradation_sets = {name: set(values) for name, values in DEGRADATION_FAMILIES.items()}
    for left, right in (("train", "validation"), ("train", "sealed_public"), ("validation", "sealed_public")):
        assert not (renderer_sets[left] & renderer_sets[right])
        assert not (degradation_sets[left] & degradation_sets[right])


def test_visible_scene_contract_and_all_prohibited_kinds() -> None:
    for split in ("train", "validation"):
        scene = dataset_module._draw_scene(split, 0)
        assert scene.tensor.shape == (3, HEIGHT, WIDTH)
        assert scene.center_target.shape == (1, HEIGHT, WIDTH)
        assert scene.radius_target.shape == (1, HEIGHT, WIDTH)
        assert scene.artifact_target.shape == (1, HEIGHT, WIDTH)
        assert np.isfinite(scene.tensor).all()
        assert set(kind for kind, _, _ in scene.hard_negatives) == set(PROHIBITED_KINDS)
        assert scene.scene_id.startswith(f"marker-mask-consensus-v8-{split}-")


def test_model_preserves_dense_contract_and_exact_input_masks_gate_centers() -> None:
    model = create_model().eval()
    value = torch.zeros((1, 3, HEIGHT, WIDTH), dtype=torch.float32)
    value[:, 1, 20, 30] = 1.0
    value[:, 2, 40, 50] = 1.0
    with torch.inference_mode():
        output = model(value)
    assert output.shape == value.shape
    assert output[0, 0, 20, 30] == 0
    assert output[0, 0, 40, 50] == 0
    assert output[0, 2, 40, 50] == 1
    assert torch.all((output[:, 0] >= 0) & (output[:, 0] <= 1))
    assert torch.all(output[:, 1] >= 1)
    assert torch.all((output[:, 2] >= 0) & (output[:, 2] <= 1))


def test_p2_is_aggregate_only_zero_optimizer_artifact_threshold_calibration() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    config = _json(ROOT / "training/p2.json")
    result = _json(ROOT / "P1_RESULT.json")
    assert result["status"] == "failed_selection_consumed"
    assert result["aggregate_only_evidence"] is True
    assert result["case_detail_or_pixels_inspected"] is False
    assert result["public_gate_evaluations"] == 0
    assert config["expected_optimizer_steps"] == 0
    assert config["artifact_threshold"] == 0.45
    assert config["p1_result_sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["public_gate_archive_opened"] is False
    assert config["public_gate_evaluations"] == 0
    assert protocol["p2_artifact_threshold"] == 0.45
    assert protocol["p2_expected_optimizer_steps"] == 0
    assert protocol["p2_status"] == "failed_selection_consumed"


def test_p3_is_aggregate_only_bounded_final_candidate() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    config = _json(ROOT / "training/p3.json")
    result = _json(ROOT / "P2_RESULT.json")
    assert result["status"] == "failed_selection_consumed"
    assert result["aggregate_only_evidence"] is True
    assert result["case_detail_or_pixels_inspected"] is False
    assert result["public_gate_evaluations"] == 0
    assert config["expected_optimizer_steps"] == EXPECTED_OPTIMIZER_STEPS == 768
    assert config["artifact_positive_weight"] == ARTIFACT_POSITIVE_WEIGHT == 1.0
    assert config["fixed_radius_pixels"] == FIXED_RADIUS_PIXELS == 2.5
    assert config["p2_result_sha256"] == sha256_file(ROOT / "P2_RESULT.json")
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["public_gate_archive_opened"] is False
    assert config["public_gate_evaluations"] == 0
    assert protocol["p3_expected_optimizer_steps"] == 768
    assert protocol["p3_fixed_radius_pixels"] == 2.5
    assert protocol["execution_authorized"] is False


def test_p3_photometric_change_preserves_masks_and_fixed_radius_contract() -> None:
    value = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    value[:, 0] = 0.75
    value[:, 1, 2, 3] = 1.0
    value[:, 2, 4, 5] = 1.0
    adjusted = _photometric_batch(value, epoch=2, batch_ordinal=4)
    assert not torch.equal(adjusted[:, 0], value[:, 0])
    assert torch.equal(adjusted[:, 1:], value[:, 1:])
    model = FixedRadiusInferenceModel(create_model().eval()).eval()
    with torch.inference_mode():
        output = model(value)
    assert torch.all(output[:, 1] == FIXED_RADIUS_PIXELS)
    assert output[0, 0, 2, 3] == 0
    assert output[0, 0, 4, 5] == 0
