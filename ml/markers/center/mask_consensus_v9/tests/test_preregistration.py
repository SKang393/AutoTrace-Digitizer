# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Pre-freeze checks for the marker-center V9 recovery protocol."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from ml.markers.center.mask_consensus_v8.dataset import (
    DEGRADATION_FAMILIES as V8_DEGRADATION_FAMILIES,
    RENDERER_FAMILIES as V8_RENDERER_FAMILIES,
)
from ml.markers.center.mask_consensus_v9.dataset import (
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
from ml.markers.center.mask_consensus_v9.protocol import (
    DESIGN_SOURCE_PATHS,
    EXPERIMENT_BUDGET,
    ONNX_PARITY_TOLERANCE,
    PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
    TRIGGER_RESULT_SHA256,
)
from ml.markers.center.mask_consensus_v9.public_gate import EVALUATOR_SOURCE_PATHS, run as run_public_gate
from ml.markers.center.mask_consensus_v9.train_p1 import (
    _parity_reproduction_within_tolerance,
)
from ml.markers.center.mask_consensus_v9.train_p3 import (
    ARTIFACT_OUTPUT_CONTRACTION,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
    SpecificityParityInferenceModel,
    TVERSKY_FALSE_NEGATIVE_WEIGHT,
    TVERSKY_FALSE_POSITIVE_WEIGHT,
    _verify_config_and_inputs as verify_p3_config_and_inputs,
)
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/mask_consensus_v9"


def test_protocol_is_new_bounded_defect_class_without_weakened_model_gate() -> None:
    assert EXPERIMENT_BUDGET == 3
    assert ONNX_PARITY_TOLERANCE == 1e-5
    assert PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE == 1e-6
    assert TRIGGER_RESULT_SHA256 == "fd2dfa1a196e4ddbb63d5099fbc44e734630370e211e7917509286b5ee3204f8"


def test_split_counts_and_all_renderer_families_are_disjoint_from_v8() -> None:
    assert (TRAIN_SCENE_COUNT, VALIDATION_SCENE_COUNT, PUBLIC_SCENE_COUNT) == (512, 128, 160)
    for families in (RENDERER_FAMILIES, DEGRADATION_FAMILIES):
        values = [set(families[name]) for name in ("train", "validation", "sealed_public")]
        assert not (values[0] & values[1])
        assert not (values[0] & values[2])
        assert not (values[1] & values[2])
    assert not set().union(*map(set, RENDERER_FAMILIES.values())) & set().union(
        *map(set, V8_RENDERER_FAMILIES.values())
    )
    assert not set().union(*map(set, DEGRADATION_FAMILIES.values())) & set().union(
        *map(set, V8_DEGRADATION_FAMILIES.values())
    )


def test_visible_scene_contract_uses_fresh_identity_and_all_prohibited_kinds() -> None:
    scene = render_scene("validation", 0)
    assert scene.tensor.shape == (3, HEIGHT, WIDTH)
    assert scene.center_target.shape == (1, HEIGHT, WIDTH)
    assert scene.radius_target.shape == (1, HEIGHT, WIDTH)
    assert scene.artifact_target.shape == (1, HEIGHT, WIDTH)
    assert np.isfinite(scene.tensor).all()
    assert set(kind for kind, _, _ in scene.hard_negatives) == set(PROHIBITED_KINDS)
    assert scene.scene_id == "marker-mask-consensus-v9-validation-0000"


def test_predecessor_parity_reproduction_tolerance_is_narrow_and_inclusive() -> None:
    expected = (3.3080577850341797e-06, 1.621246337890625e-05, 8.970499038696289e-06)
    observed_v8_p3 = (3.874301910400391e-06, 1.621246337890625e-05, 8.970499038696289e-06)
    assert _parity_reproduction_within_tolerance(observed_v8_p3, expected)
    assert _parity_reproduction_within_tolerance(
        (expected[0] + PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE, expected[1], expected[2]),
        expected,
    )
    assert not _parity_reproduction_within_tolerance((expected[0] + 2e-6, expected[1], expected[2]), expected)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_splits_and_preregistered_sources_match_exact_bytes() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    freeze = _json(ROOT / "SPLIT_FREEZE_REPORT.json")
    selection = _json(ROOT / "SELECTION_MANIFEST.json")
    public_seal = _json(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    config = _json(ROOT / "training/p3.json")
    gate = _json(ROOT / "gates/sealed-public-v1.json")
    ledger = _json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol["revision"])
    assert protocol["state"] == "candidate_3_failed_selection_budget_exhausted"
    assert protocol["execution_authorized"] is False
    assert protocol["authorized_candidate_id"] is None
    assert protocol["execution_blocker"]
    assert entry["status"] == "exhausted"
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["execution_blocker"] == protocol["execution_blocker"]
    assert entry["refusal_required_before_output"] is True
    assert protocol["preregistration_commit"] == "20b803ae8b0f6562c22142029cdcb46eaf4de0cf"
    assert protocol["preregistration_tree"] == "49c15cc1d583589ad1f52fdc81e13d83387958d4"
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
    assert protocol["p2_preregistration_commit"] == "f9416060111a696ca866fc496dab243fdf287c04"
    assert protocol["p2_preregistration_tree"] == "d47a095abb9dc92f55ad80b4cddad9db6df0294a"
    assert entry["p2_preregistration_commit"] == protocol["p2_preregistration_commit"]
    assert entry["p2_preregistration_tree"] == protocol["p2_preregistration_tree"]
    committed_p2_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", str(protocol["p2_preregistration_commit"])],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert committed_p2_tree == protocol["p2_preregistration_tree"]
    assert protocol["p3_preregistration_commit"] == "46f19785196a7fc9e902c9545a4d31540a7663eb"
    assert protocol["p3_preregistration_tree"] == "e4e80eb14699509811bfcac49c2fb995a8133cdc"
    assert entry["p3_preregistration_commit"] == protocol["p3_preregistration_commit"]
    assert entry["p3_preregistration_tree"] == protocol["p3_preregistration_tree"]
    committed_p3_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", str(protocol["p3_preregistration_commit"])],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert committed_p3_tree == protocol["p3_preregistration_tree"]
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert sha256_file(ROOT / "PROTOCOL.json") == entry["protocol_sha256"]
    assert sha256_file(ROOT / "SPLIT_FREEZE_REPORT.json") == protocol["split_freeze_report_sha256"]
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert sha256_file(ROOT / "PUBLIC_DATASET_MANIFEST.json") == protocol["public_dataset_manifest_sha256"]
    assert sha256_file(ROOT / "P1_RESULT.json") == protocol["p1_result_sha256"]
    assert sha256_file(ROOT / "P2_RESULT.json") == protocol["p2_result_sha256"]
    assert sha256_file(ROOT / "P3_RESULT.json") == protocol["p3_result_sha256"]
    assert sha256_file(ROOT / "training/p3.json") == protocol["candidate_config_sha256"]
    assert sha256_file(ROOT / "gates/sealed-public-v1.json") == protocol["public_gate_config_sha256"]
    assert source_bundle_sha256(REPO_ROOT, DESIGN_SOURCE_PATHS) == freeze["generator_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
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
    assert p1_result["status"] == "failed_runner_consumed"
    assert p1_result["failure_message"] == "'p2_parity_by_output_channel'"
    assert p1_result["optimizer_steps"] == 0
    assert p1_result["model_payload_created"] is False
    assert sha256_file(REPO_ROOT / p1_result["candidate_report_path"]) == p1_result["candidate_report_sha256"]
    assert sha256_file(REPO_ROOT / p1_result["training_opened_seal_path"]) == p1_result["training_opened_seal_sha256"]
    assert sha256_file(REPO_ROOT / p1_result["training_result_seal_path"]) == p1_result["training_result_seal_sha256"]
    p2_result = _json(ROOT / "P2_RESULT.json")
    assert p2_result["status"] == "failed_selection_consumed"
    assert p2_result["optimizer_steps"] == 768
    assert p2_result["selection_exact_scene_count"] == 121
    assert p2_result["selection_false_positives"] == 8
    assert p2_result["selection_false_negatives"] == 23
    assert p2_result["onnx_parity_passed"] is False
    assert sha256_file(REPO_ROOT / p2_result["candidate_report_path"]) == p2_result["candidate_report_sha256"]
    assert sha256_file(REPO_ROOT / p2_result["checkpoint_path"]) == p2_result["checkpoint_sha256"]
    assert sha256_file(REPO_ROOT / p2_result["onnx_path"]) == p2_result["onnx_sha256"]
    p3_result = _json(ROOT / "P3_RESULT.json")
    assert p3_result["status"] == "failed_selection_consumed"
    assert p3_result["optimizer_steps"] == 768
    assert p3_result["selection_exact_scene_count"] == 121
    assert p3_result["selection_false_positives"] == 12
    assert p3_result["selection_false_negatives"] == 18
    assert p3_result["selection_duplicate_count"] == 0
    assert p3_result["selection_prohibited_structure_hits"] == 0
    assert p3_result["artifact_precision"] == 0.7897035539741141
    assert p3_result["artifact_recall"] == 0.980064776805665
    assert p3_result["onnx_parity_passed"] is True
    assert sha256_file(REPO_ROOT / p3_result["candidate_report_path"]) == p3_result["candidate_report_sha256"]
    assert sha256_file(REPO_ROOT / p3_result["checkpoint_path"]) == p3_result["checkpoint_sha256"]
    assert sha256_file(REPO_ROOT / p3_result["onnx_path"]) == p3_result["onnx_sha256"]


def test_p3_input_preflight_binds_consumed_p2_and_fixed_specificity_parity_design() -> None:
    config = _json(ROOT / "training/p3.json")
    p2_result, checkpoint_path, onnx_path = verify_p3_config_and_inputs(config)
    assert p2_result["status"] == "failed_selection_consumed"
    assert sha256_file(checkpoint_path) == config["p2_checkpoint_sha256"]
    assert sha256_file(onnx_path) == config["p2_onnx_sha256"]
    assert config["artifact_output_contraction"] == ARTIFACT_OUTPUT_CONTRACTION == 0.5
    assert config["tversky_false_positive_weight"] == TVERSKY_FALSE_POSITIVE_WEIGHT == 0.9
    assert config["tversky_false_negative_weight"] == TVERSKY_FALSE_NEGATIVE_WEIGHT == 0.1
    assert config["p2_checkpoint_reused"] is True
    assert config["case_detail_or_pixels_inspected"] is False


def test_p3_inference_transform_preserves_seed_artifacts_and_contracts_only_learned_output() -> None:
    class ConstantHeads(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            center = torch.full_like(value[:, 0:1], 0.4)
            radius = torch.full_like(value[:, 0:1], 7.0)
            artifact = torch.full_like(value[:, 0:1], 0.8)
            return torch.cat((center, radius, artifact), dim=1)

    value = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    value[:, 2, 0, 0] = 0.9
    output = SpecificityParityInferenceModel(ConstantHeads())(value)
    assert torch.equal(output[:, 0:1], torch.full_like(value[:, 0:1], 0.4))
    assert torch.equal(output[:, 1:2], torch.full_like(value[:, 0:1], 2.5))
    assert output[0, 2, 0, 0].item() == pytest.approx(0.9)
    assert output[0, 2, 1, 1].item() == pytest.approx(0.4)


def test_p3_single_use_seals_bind_the_consumed_failed_selection_report() -> None:
    result = _json(ROOT / "P3_RESULT.json")
    seal_root = REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-mask-consensus-v9/P3"
    opened_path = seal_root / "opened.json"
    result_path = seal_root / "result.json"
    sealed_result = _json(result_path)
    assert sha256_file(opened_path) == result["training_opened_seal_sha256"]
    assert sha256_file(result_path) == result["training_result_seal_sha256"]
    assert sealed_result["status"] == "failed_selection"
    assert sealed_result["opened_sha256"] == result["training_opened_seal_sha256"]
    assert sealed_result["report_sha256"] == result["candidate_report_sha256"]
    assert not (REPO_ROOT / "ml/markers/center/artifacts/mask-consensus-v9/public-gate-report.json").exists()


def test_public_gate_refuses_unapproved_candidate_before_model_or_archive_execution(tmp_path: Path) -> None:
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
