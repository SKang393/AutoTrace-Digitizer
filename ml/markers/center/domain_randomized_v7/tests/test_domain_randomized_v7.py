# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration tests for marker-center V7."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import ml.markers.center.domain_randomized_v7.dataset as dataset_module
from ml.markers.center.domain_randomized_v7.dataset import (
    DEGRADATION_FAMILIES,
    HEIGHT,
    PROHIBITED_KINDS,
    PUBLIC_SCENE_COUNT,
    RENDERER_FAMILIES,
    REQUIRED_DISJOINT_CLEARANCE,
    TRAIN_SCENE_COUNT,
    VALIDATION_SCENE_COUNT,
    WIDTH,
    validate_scene_feasibility,
)
from ml.markers.center.domain_randomized_v7.model import create_model
from ml.markers.center.domain_randomized_v7.train_p1 import RUNNER_SOURCE_PATHS, THRESHOLDS
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/domain_randomized_v7"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_archives_are_exact_feasible_and_unopened() -> None:
    selection = _json(ROOT / "SELECTION_MANIFEST.json")
    freeze = _json(ROOT / "SPLIT_FREEZE_REPORT.json")
    split_seal = _json(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    dataset = _json(ROOT / "PUBLIC_DATASET_MANIFEST.json")
    assert selection["train"]["count"] == TRAIN_SCENE_COUNT == 384
    assert selection["validation"]["count"] == VALIDATION_SCENE_COUNT == 96
    assert selection["sealed_public"]["count"] == PUBLIC_SCENE_COUNT == 96
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
    assert split_seal["public_gate_archive_opened"] is False
    assert split_seal["public_gate_evaluations"] == 0
    assert freeze["sealed_public_feasibility"]["truth_hard_acceptance_overlap_count"] == 0
    assert len(dataset["fixtures"]) == PUBLIC_SCENE_COUNT
    assert dataset["case_truth_values_emitted"] is False
    assert dataset["fixture_pixels_emitted"] is False


def test_renderer_and_degradation_families_are_split_disjoint() -> None:
    renderer_sets = {name: set(values) for name, values in RENDERER_FAMILIES.items()}
    degradation_sets = {name: set(values) for name, values in DEGRADATION_FAMILIES.items()}
    assert len(renderer_sets["train"]) == 4
    assert len(renderer_sets["validation"]) == 3
    assert len(renderer_sets["sealed_public"]) == 2
    assert len(degradation_sets["train"]) == 5
    assert len(degradation_sets["validation"]) == 3
    assert len(degradation_sets["sealed_public"]) == 2
    assert not (renderer_sets["train"] & renderer_sets["validation"])
    assert not (renderer_sets["train"] & renderer_sets["sealed_public"])
    assert not (renderer_sets["validation"] & renderer_sets["sealed_public"])
    assert not (degradation_sets["train"] & degradation_sets["validation"])
    assert not (degradation_sets["train"] & degradation_sets["sealed_public"])
    assert not (degradation_sets["validation"] & degradation_sets["sealed_public"])


def test_visible_scenes_preserve_dense_contract_and_prohibited_kinds() -> None:
    for split in ("train", "validation"):
        for index in (0, 1, 17):
            scene = dataset_module._draw_scene(split, index)
            validate_scene_feasibility(scene)
            assert scene.tensor.shape == (3, HEIGHT, WIDTH)
            assert scene.center_target.shape == (1, HEIGHT, WIDTH)
            assert scene.radius_target.shape == (1, HEIGHT, WIDTH)
            assert scene.artifact_target.shape == (1, HEIGHT, WIDTH)
            assert np.isfinite(scene.tensor).all()
            assert set(kind for kind, _, _ in scene.hard_negatives) == set(PROHIBITED_KINDS)
            assert scene.scene_id.startswith(f"marker-domain-randomized-v7-{split}-")


def test_infeasible_truth_and_prohibited_region_is_rejected() -> None:
    scene = dataset_module._draw_scene("validation", 0)
    _, hard_x, hard_y = scene.hard_negatives[0]
    conflicted = replace(scene, centers=((hard_x, hard_y, scene.centers[0][2]),) + scene.centers[1:])
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


def test_protocol_config_and_budget_are_checksum_bound_and_p1_only() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    config = _json(ROOT / "training/p1.json")
    ledger = _json(LEDGER_PATH)
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol["revision"])
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    design_paths = tuple(REPO_ROOT / path for path in protocol["design_source_paths"])
    design_hash = source_bundle_sha256(REPO_ROOT, tuple(path.relative_to(REPO_ROOT) for path in design_paths))
    assert design_hash == protocol["design_source_bundle_sha256"]
    assert sha256_file(ROOT / "training/p1.json") == protocol["candidate_config_sha256"]
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert sha256_file(ROOT / "SPLIT_FREEZE_REPORT.json") == protocol["split_freeze_report_sha256"]
    assert sha256_file(REPO_ROOT / protocol["trigger_public_result_path"]) == protocol["trigger_public_result_sha256"]
    assert config["selection_thresholds"] == list(THRESHOLDS)
    assert protocol["execution_authorized"] is True
    assert protocol["public_gate_authorized"] is False
    assert protocol["public_gate_archive_opened"] is False
    assert protocol["public_gate_evaluations"] == 0
    assert protocol["trigger_case_detail_or_pixels_used"] is False
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is False
    assert entry["manifest_created"] is False
    assert entry["model_store_promoted"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
