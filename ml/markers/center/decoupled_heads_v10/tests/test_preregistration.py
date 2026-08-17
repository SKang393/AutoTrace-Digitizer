# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Pre-execution checks for the marker-center V10 defect class."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from ml.markers.center.mask_consensus_v9.dataset import (
    DEGRADATION_FAMILIES as V9_DEGRADATION_FAMILIES,
    RENDERER_FAMILIES as V9_RENDERER_FAMILIES,
)
from ml.markers.center.decoupled_heads_v10.dataset import (
    DEGRADATION_FAMILIES,
    HEIGHT,
    PROHIBITED_KINDS,
    PUBLIC_SCENE_COUNT,
    RENDERER_FAMILIES,
    TRAIN_SCENE_COUNT,
    VALIDATION_SCENE_COUNT,
    WIDTH,
    render_scene,
)
from ml.markers.center.decoupled_heads_v10.model import create_model
from ml.markers.center.decoupled_heads_v10.protocol import (
    DESIGN_SOURCE_PATHS,
    EXPERIMENT_BUDGET,
    ONNX_PARITY_TOLERANCE,
    TRIGGER_RESULT_SHA256,
)
from ml.markers.center.decoupled_heads_v10.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    run as run_public_gate,
)
from ml.markers.center.decoupled_heads_v10.train_p1 import (
    RUNNER_SOURCE_PATHS as P1_RUNNER_SOURCE_PATHS,
)
from ml.markers.center.decoupled_heads_v10.train_p2 import (
    REFLECTION_SCHEDULE,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _reflect_point,
    _reflect_tensor,
    _verify_config_and_inputs as verify_p2_config_and_inputs,
    run as run_p2,
)
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/decoupled_heads_v10"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_is_new_bounded_defect_class_without_weakened_gates() -> None:
    assert EXPERIMENT_BUDGET == 3
    assert ONNX_PARITY_TOLERANCE == 1e-5
    assert TRIGGER_RESULT_SHA256 == "542c6093415c251256ef0cbb3e25ac97251d08b1ca1e259317352117943c6f79"


def test_split_counts_and_all_families_are_disjoint_from_v9() -> None:
    assert (TRAIN_SCENE_COUNT, VALIDATION_SCENE_COUNT, PUBLIC_SCENE_COUNT) == (512, 128, 160)
    for families in (RENDERER_FAMILIES, DEGRADATION_FAMILIES):
        values = [set(families[name]) for name in ("train", "validation", "sealed_public")]
        assert not (values[0] & values[1])
        assert not (values[0] & values[2])
        assert not (values[1] & values[2])
    assert not set().union(*map(set, RENDERER_FAMILIES.values())) & set().union(
        *map(set, V9_RENDERER_FAMILIES.values())
    )
    assert not set().union(*map(set, DEGRADATION_FAMILIES.values())) & set().union(
        *map(set, V9_DEGRADATION_FAMILIES.values())
    )


def test_visible_scene_contract_is_fresh_and_contains_all_prohibited_kinds() -> None:
    scene = render_scene("validation", 0)
    assert scene.tensor.shape == (3, HEIGHT, WIDTH)
    assert scene.center_target.shape == (1, HEIGHT, WIDTH)
    assert scene.radius_target.shape == (1, HEIGHT, WIDTH)
    assert scene.artifact_target.shape == (1, HEIGHT, WIDTH)
    assert np.isfinite(scene.tensor).all()
    assert set(kind for kind, _, _ in scene.hard_negatives) == set(PROHIBITED_KINDS)
    assert scene.scene_id == "marker-decoupled-heads-v10-validation-0000"


def test_model_uses_disjoint_towers_and_preserves_dense_contract() -> None:
    model = create_model()
    center_parameters = {id(value) for value in model.center_tower.parameters()}
    artifact_parameters = {id(value) for value in model.artifact_tower.parameters()}
    assert not center_parameters & artifact_parameters
    value = torch.zeros((1, 3, 24, 32), dtype=torch.float32)
    value[:, 2, 5, 7] = 1.0
    output = model(value)
    assert output.shape == (1, 3, 24, 32)
    assert torch.all(output[:, 1:2] == 2.5)
    assert output[0, 2, 5, 7].item() == 1.0
    assert output[0, 0, 5, 7].item() == 0.0


def test_frozen_splits_and_preregistered_sources_match_exact_bytes() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    freeze = _json(ROOT / "SPLIT_FREEZE_REPORT.json")
    selection = _json(ROOT / "SELECTION_MANIFEST.json")
    public_seal = _json(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    config = _json(ROOT / "training/p2.json")
    gate = _json(ROOT / "gates/sealed-public-v1.json")
    ledger = _json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol["revision"])
    assert protocol["state"] == "candidate_1_failed_selection_candidate_2_preregistered"
    assert protocol["execution_authorized"] is False
    assert protocol["authorized_candidate_id"] is None
    assert protocol["execution_blocker"]
    assert entry["status"] == "candidate_2_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P2"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P3"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["execution_blocker"] == protocol["execution_blocker"]
    assert protocol["preregistration_commit"] == "d4a3987d96a0763730fb9db840ee6c31c4da1abb"
    assert protocol["preregistration_tree"] == "bff1a927922ed9e3e50b315b1f6a1a82cf160c68"
    assert entry["preregistration_commit"] == protocol["preregistration_commit"]
    assert entry["preregistration_tree"] == protocol["preregistration_tree"]
    committed_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", str(protocol["preregistration_commit"])],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert committed_tree == protocol["preregistration_tree"]
    assert entry["refusal_required_before_output"] is True
    assert sha256_file(ROOT / "PROTOCOL.json") == entry["protocol_sha256"]
    assert sha256_file(ROOT / "SPLIT_FREEZE_REPORT.json") == protocol["split_freeze_report_sha256"]
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert sha256_file(ROOT / "PUBLIC_DATASET_MANIFEST.json") == protocol["public_dataset_manifest_sha256"]
    assert sha256_file(ROOT / "P1_RESULT.json") == protocol["p1_result_sha256"]
    assert sha256_file(ROOT / "training/p2.json") == protocol["candidate_config_sha256"]
    assert sha256_file(ROOT / "gates/sealed-public-v1.json") == protocol["public_gate_config_sha256"]
    assert source_bundle_sha256(REPO_ROOT, DESIGN_SOURCE_PATHS) == freeze["generator_source_bundle_sha256"]
    p1_config = _json(ROOT / "training/p1.json")
    assert source_bundle_sha256(REPO_ROOT, P1_RUNNER_SOURCE_PATHS) == p1_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS) == gate["expected_evaluator_source_bundle_sha256"]
    for name in ("train", "validation", "sealed_public"):
        archive_path = REPO_ROOT / selection[name]["archive_path"]
        assert sha256_file(archive_path) == selection[name]["archive_sha256"]
        assert selection[name]["feasibility"]["truth_hard_acceptance_overlap_count"] == 0
        assert selection[name]["feasibility"]["truth_mask_conflict_count"] == 0
        assert selection[name]["feasibility"]["hard_point_missing_from_artifact_truth_count"] == 0
    assert freeze["model_execution_count_at_freeze"] == 0
    assert freeze["optimizer_step_count_at_freeze"] == 0
    assert freeze["public_gate_archive_opened"] is False
    assert freeze["public_gate_evaluations"] == 0
    assert public_seal["public_gate_archive_opened"] is False
    assert public_seal["public_gate_evaluations"] == 0
    p1_result = _json(ROOT / "P1_RESULT.json")
    assert p1_result["status"] == "failed_selection_consumed"
    assert p1_result["optimizer_steps"] == 1792
    assert p1_result["selection_exact_scene_count"] == 120
    assert p1_result["selection_false_positives"] == 11
    assert p1_result["selection_false_negatives"] == 23
    assert p1_result["selection_duplicate_count"] == 0
    assert p1_result["selection_prohibited_structure_hits"] == 0
    assert p1_result["selection_marker_artifact_hits"] == 0
    assert p1_result["artifact_precision"] == 0.7857918313961029
    assert p1_result["artifact_recall"] == 0.9770655093456437
    assert p1_result["onnx_parity_passed"] is True
    assert p1_result["aggregate_only_evidence"] is True
    assert p1_result["case_detail_or_pixels_inspected"] is False
    assert sha256_file(REPO_ROOT / p1_result["candidate_report_path"]) == p1_result["candidate_report_sha256"]
    assert sha256_file(REPO_ROOT / p1_result["checkpoint_path"]) == p1_result["checkpoint_sha256"]
    assert sha256_file(REPO_ROOT / p1_result["onnx_path"]) == p1_result["onnx_sha256"]
    assert not (ROOT / "P2_RESULT.json").exists()
    assert not (REPO_ROOT / "ml/markers/center/artifacts/decoupled-heads-v10/P2-run").exists()


def test_p2_preflight_binds_consumed_p1_aggregate_and_frozen_v10_split() -> None:
    config = _json(ROOT / "training/p2.json")
    selection, train_path, validation_path = verify_p2_config_and_inputs(config)
    assert sha256_file(train_path) == selection["train"]["archive_sha256"]
    assert sha256_file(validation_path) == selection["validation"]["archive_sha256"]
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["p1_checkpoint_reused"] is False
    assert config["frozen_v10_split_reused"] is True
    assert config["augmentation_schedule"] == list(REFLECTION_SCHEDULE)


def test_p2_reflections_transform_tensors_targets_and_coordinates_exactly() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    assert torch.equal(_reflect_tensor(value, 0), value)
    assert torch.equal(_reflect_tensor(value, 1), torch.flip(value, (-1,)))
    assert torch.equal(_reflect_tensor(value, 2), torch.flip(value, (-2,)))
    assert torch.equal(_reflect_tensor(value, 3), torch.flip(value, (-2, -1)))
    assert _reflect_point(1.25, 0.5, width=4, height=3, transform_index=0) == (1.25, 0.5)
    assert _reflect_point(1.25, 0.5, width=4, height=3, transform_index=1) == (1.75, 0.5)
    assert _reflect_point(1.25, 0.5, width=4, height=3, transform_index=2) == (1.25, 1.5)
    assert _reflect_point(1.25, 0.5, width=4, height=3, transform_index=3) == (1.75, 1.5)


def test_p2_refuses_execution_before_separate_authorization(tmp_path: Path) -> None:
    output_path = tmp_path / "P2-run"
    with pytest.raises(RuntimeError, match="must be committed before use|not authorized"):
        run_p2(output_path)
    assert not output_path.exists()


def test_public_gate_refuses_unapproved_candidate_before_model_or_archive_execution(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate-report.json"
    output_path = tmp_path / "public-report.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "P1",
                "selection_gate_passed": True,
                "onnx_path": "does-not-exist.onnx",
                "onnx_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not separately authorized"):
        run_public_gate(candidate_path, output_path)
    assert not output_path.exists()
