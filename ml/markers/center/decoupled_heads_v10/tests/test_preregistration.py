# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Pre-execution checks for the marker-center V10 defect class."""

from __future__ import annotations

import json
from pathlib import Path

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
    RUNNER_SOURCE_PATHS,
    _verify_config_and_inputs,
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
    config = _json(ROOT / "training/p1.json")
    gate = _json(ROOT / "gates/sealed-public-v1.json")
    ledger = _json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol["revision"])
    assert protocol["state"] == "split_frozen_runner_and_public_evaluator_preregistered_execution_blocked"
    assert protocol["execution_authorized"] is False
    assert protocol["authorized_candidate_id"] is None
    assert entry["status"] == "candidate_1_preregistered_execution_blocked"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["execution_authorized"] is False
    assert entry["refusal_required_before_output"] is True
    assert sha256_file(ROOT / "PROTOCOL.json") == entry["protocol_sha256"]
    assert sha256_file(ROOT / "SPLIT_FREEZE_REPORT.json") == protocol["split_freeze_report_sha256"]
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert sha256_file(ROOT / "PUBLIC_DATASET_MANIFEST.json") == protocol["public_dataset_manifest_sha256"]
    assert sha256_file(ROOT / "training/p1.json") == protocol["candidate_config_sha256"]
    assert sha256_file(ROOT / "gates/sealed-public-v1.json") == protocol["public_gate_config_sha256"]
    assert source_bundle_sha256(REPO_ROOT, DESIGN_SOURCE_PATHS) == freeze["generator_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
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
    assert not (ROOT / "P1_RESULT.json").exists()
    assert not (REPO_ROOT / "ml/markers/center/artifacts/decoupled-heads-v10/P1-run").exists()


def test_candidate_preflight_binds_terminal_v9_aggregate_and_fresh_split() -> None:
    config = _json(ROOT / "training/p1.json")
    selection, train_path, validation_path = _verify_config_and_inputs(config)
    assert sha256_file(train_path) == selection["train"]["archive_sha256"]
    assert sha256_file(validation_path) == selection["validation"]["archive_sha256"]
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["prior_checkpoint_reused"] is False
    assert config["prior_fixture_bytes_reused"] is False


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

