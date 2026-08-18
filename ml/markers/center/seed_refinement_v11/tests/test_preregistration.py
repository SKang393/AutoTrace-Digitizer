# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Pre-execution checks for the marker-center V11 defect class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as functional

from ml.markers.center.decoupled_heads_v10.dataset import (
    DEGRADATION_FAMILIES as V10_DEGRADATION_FAMILIES,
    RENDERER_FAMILIES as V10_RENDERER_FAMILIES,
)
from ml.markers.center.seed_refinement_v11.dataset import (
    DEGRADATION_FAMILIES,
    PUBLIC_SCENE_COUNT,
    RENDERER_FAMILIES,
    TRAIN_SCENE_COUNT,
    VALIDATION_SCENE_COUNT,
    read_archive,
    render_scene,
)
from ml.markers.center.seed_refinement_v11.model import create_model
from ml.markers.center.seed_refinement_v11.protocol import (
    DESIGN_SOURCE_PATHS,
    EXPERIMENT_BUDGET,
    ONNX_PARITY_TOLERANCE,
    TRIGGER_RESULT_SHA256,
)
from ml.markers.center.seed_refinement_v11.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    run as run_public_gate,
)
from ml.markers.center.seed_refinement_v11.train_p1 import (
    RUNNER_SOURCE_PATHS as P1_RUNNER_SOURCE_PATHS,
    _evaluate,
    _export,
    _onnx_output,
    _torch_output,
    _verify_config_and_inputs as verify_p1_config_and_inputs,
)
from ml.markers.center.seed_refinement_v11.train_p2 import (
    P1_RESULT_SHA256,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _unsupported_seed_addition_loss,
    _verify_config_and_inputs as verify_p2_config_and_inputs,
)
from ml.markers.center.seed_refinement_v11.train_p3 import (
    P2_RESULT_SHA256,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
    _false_seed_retention_loss,
    _verify_config_and_inputs as verify_p3_config_and_inputs,
)
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/seed_refinement_v11"
V10_ROOT = REPO_ROOT / "ml/markers/center/decoupled_heads_v10"
P3_RESULT_SHA256 = "4276a6baaf9a0cd15cf4d753cdfac92f5d25fbcf0588075648e7e25158f669ac"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes(manifest: dict[str, object], names: tuple[str, ...]) -> set[str]:
    result: set[str] = set()
    for name in names:
        with np.load(REPO_ROOT / manifest[name]["archive_path"], allow_pickle=False) as archive:
            result.update(str(value) for value in archive["source_sha256"])
    return result


def test_protocol_is_bounded_and_does_not_weaken_any_gate() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    assert protocol["state"] == "exhausted_before_public_gate"
    assert protocol["execution_authorized"] is False
    assert protocol["authorized_candidate_id"] is None
    assert protocol["public_gate_authorized"] is False
    assert protocol["public_gate_archive_opened"] is False
    assert protocol["public_gate_evaluations"] == 0
    assert EXPERIMENT_BUDGET == 3
    assert ONNX_PARITY_TOLERANCE == 1e-5
    assert TRIGGER_RESULT_SHA256 == "c0b580c68346124b878521dc6ef46f1e3ed4fe587c29be35be66d5bb8992b62f"
    assert protocol["selection_gates"]["minimum_artifact_precision"] == 0.9
    assert protocol["selection_gates"]["minimum_artifact_recall"] == 0.95
    assert protocol["selection_gates"]["false_positive_maximum"] == 0
    assert protocol["selection_gates"]["false_negative_maximum"] == 0
    assert protocol["selection_gates"]["duplicate_maximum"] == 0
    assert protocol["selection_gates"]["prohibited_structure_hit_maximum"] == 0
    assert protocol["selection_gates"]["marker_artifact_hit_maximum"] == 0


def test_fresh_families_and_source_hashes_are_disjoint_from_v10() -> None:
    assert (TRAIN_SCENE_COUNT, VALIDATION_SCENE_COUNT, PUBLIC_SCENE_COUNT) == (512, 128, 160)
    for families, prior in (
        (RENDERER_FAMILIES, V10_RENDERER_FAMILIES),
        (DEGRADATION_FAMILIES, V10_DEGRADATION_FAMILIES),
    ):
        current_values = [set(families[name]) for name in ("train", "validation", "sealed_public")]
        assert not (current_values[0] & current_values[1])
        assert not (current_values[0] & current_values[2])
        assert not (current_values[1] & current_values[2])
        assert not set().union(*current_values) & set().union(*map(set, prior.values()))
    current = _json(ROOT / "SELECTION_MANIFEST.json")
    prior = _json(V10_ROOT / "SELECTION_MANIFEST.json")
    current_visible = _source_hashes(current, ("train", "validation"))
    prior_visible = _source_hashes(prior, ("train", "validation"))
    current_public = {
        item["image_sha256"] for item in _json(ROOT / "PUBLIC_DATASET_MANIFEST.json")["fixtures"]
    }
    prior_public = {
        item["image_sha256"] for item in _json(V10_ROOT / "PUBLIC_DATASET_MANIFEST.json")["fixtures"]
    }
    assert len(current_visible | current_public) == 800
    assert not (current_visible | current_public) & (prior_visible | prior_public)


def test_model_can_remove_and_add_seed_pixels_without_changing_dense_shape() -> None:
    model = create_model()
    value = torch.zeros((1, 3, 24, 32), dtype=torch.float32)
    value[:, 2, 5, 7] = 1.0
    with torch.no_grad():
        model.artifact_head.weight.zero_()
        model.artifact_head.bias.fill_(-10.0)
        removed = model(value)
        model.artifact_head.bias.fill_(10.0)
        added = model(torch.zeros_like(value))
    assert removed.shape == (1, 3, 24, 32)
    assert torch.all(removed[:, 1:2] == 2.5)
    assert removed[0, 2, 5, 7].item() < 0.35
    assert added[0, 2, 5, 7].item() > 0.35


def test_untrained_architecture_exports_dynamic_onnx_and_matches_cpu(tmp_path: Path) -> None:
    model = create_model().eval()
    value = np.zeros((1, 3, 28, 36), dtype=np.float32)
    value[:, 2, 4:8, 5:10] = 1.0
    path = tmp_path / "seed-refinement-v11.onnx"
    _export(model, torch.from_numpy(value), path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = _torch_output(model, value)
    actual = _onnx_output(session, value)
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert actual.shape == (1, 3, 28, 36)
    assert float(np.max(np.abs(expected - actual))) <= ONNX_PARITY_TOLERANCE


def test_refined_artifact_output_drives_selection_without_seed_union() -> None:
    selection = _json(ROOT / "SELECTION_MANIFEST.json")
    archive = read_archive(REPO_ROOT / selection["validation"]["archive_path"])
    scene_count = archive["inputs"].shape[0]
    one = {
        key: value[:1] if hasattr(value, "shape") and value.shape and value.shape[0] == scene_count else value
        for key, value in archive.items()
    }
    output = np.zeros((1, 3, *one["inputs"].shape[-2:]), dtype=np.float32)
    output[:, 0:1] = one["center_targets"]
    output[:, 1:2] = 2.5
    output[:, 2:3] = one["artifact_targets"]
    result = _evaluate(one, [output], 0.3)
    assert result["passed"] is True
    assert result["seed_removed_pixels"] > 0
    assert result["seed_added_pixels"] > 0
    assert result["artifact_precision"] == 1.0
    assert result["artifact_recall"] == 1.0


def test_frozen_sources_configs_and_ledger_match_exact_bytes() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    freeze = _json(ROOT / "SPLIT_FREEZE_REPORT.json")
    p1_config = _json(ROOT / "training/p1.json")
    p2_config = _json(ROOT / "training/p2.json")
    p3_config = _json(ROOT / "training/p3.json")
    gate = _json(ROOT / "gates/sealed-public-v1.json")
    ledger = _json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol["revision"])
    assert protocol["state"] == "exhausted_before_public_gate"
    assert protocol["preregistration_commit"] == "729d51e916c8433e4c3ccadccd28d7038007ce12"
    assert protocol["preregistration_tree"] == "66054ffe3f52814c3ccb30b3ada7ec1b4d2d4980"
    assert protocol["p2_preregistration_commit"] == "589e8892718feaa011849132313c1eb6e71f534e"
    assert protocol["p2_preregistration_tree"] == "8b70c750ca256de7ba5d5a04bf7408089afb4e52"
    assert protocol["p2_authorization_commit"] == "7a29f00517d2cb96be13320ca6b805aef3738e1d"
    assert protocol["p2_authorization_tree"] == "64e96328ee65433f6d7cdb55d726fd9e2391af25"
    assert protocol["p3_preregistration_commit"] == "be6a6a33cc42d7ac6fc3d3de8b7f6b1e70ebefbb"
    assert protocol["p3_preregistration_tree"] == "70b603dbc2a7f53846027b1bfc1ebbd58057588f"
    assert protocol["execution_authorized"] is False
    assert protocol["authorized_candidate_id"] is None
    assert protocol["preregistered_candidate_ids"] == []
    assert protocol["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["status"] == "exhausted"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["preregistration_commit"] == protocol["preregistration_commit"]
    assert entry["preregistration_tree"] == protocol["preregistration_tree"]
    assert entry["p2_preregistration_commit"] == protocol["p2_preregistration_commit"]
    assert entry["p2_preregistration_tree"] == protocol["p2_preregistration_tree"]
    assert entry["p3_preregistration_commit"] == protocol["p3_preregistration_commit"]
    assert entry["p3_preregistration_tree"] == protocol["p3_preregistration_tree"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert sha256_file(ROOT / "PROTOCOL.json") == entry["protocol_sha256"]
    assert sha256_file(ROOT / "SPLIT_FREEZE_REPORT.json") == protocol["split_freeze_report_sha256"]
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert sha256_file(ROOT / "PUBLIC_DATASET_MANIFEST.json") == protocol["public_dataset_manifest_sha256"]
    assert sha256_file(ROOT / "training/p1.json") == "05c8a1f41fa435c9158273a22fc5516dd032ab83d068094e4f5d5a67f1fdd083"
    assert sha256_file(ROOT / "training/p2.json") == "424e3070e4d3fb39347293c35b2c2d0d313ea57a15f74d62a1afeaa0c4b6a429"
    assert sha256_file(ROOT / "training/p3.json") == protocol["candidate_config_sha256"]
    assert sha256_file(ROOT / "gates/sealed-public-v1.json") == protocol["public_gate_config_sha256"]
    assert source_bundle_sha256(REPO_ROOT, DESIGN_SOURCE_PATHS) == freeze["generator_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P1_RUNNER_SOURCE_PATHS) == p1_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS) == p2_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS) == p3_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS) == gate["expected_evaluator_source_bundle_sha256"]
    assert freeze["model_execution_count_at_freeze"] == 0
    assert freeze["optimizer_step_count_at_freeze"] == 0
    assert freeze["public_gate_archive_opened"] is False
    assert freeze["public_gate_evaluations"] == 0


def test_p1_preflight_binds_v10_feasibility_and_frozen_v11_archives() -> None:
    config = _json(ROOT / "training/p1.json")
    selection, train_path, validation_path = verify_p1_config_and_inputs(config)
    assert sha256_file(train_path) == selection["train"]["archive_sha256"]
    assert sha256_file(validation_path) == selection["validation"]["archive_sha256"]
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["prior_checkpoint_reused"] is False
    assert config["prior_fixture_bytes_reused"] is False
    assert config["runtime_postprocess_profile"] == "nonmonotonic_seed_refinement_v1"


def test_p1_result_binds_exact_aggregate_report_and_single_use_seals() -> None:
    result = _json(ROOT / "P1_RESULT.json")
    assert sha256_file(ROOT / "P1_RESULT.json") == P1_RESULT_SHA256
    seal_root = REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-seed-refinement-v11/P1"
    assert result["status"] == "failed_selection_consumed"
    assert result["selection_exact_scene_count"] == 122
    assert result["selection_false_positives"] == 0
    assert result["selection_false_negatives"] == 26
    assert result["artifact_precision"] == 0.7815472274567087
    assert result["artifact_recall"] == 0.9687703003413912
    assert result["seed_added_pixels"] == 250166
    assert result["seed_removed_pixels"] == 1175
    assert result["case_detail_or_pixels_inspected"] is False
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    direct_paths = (
        REPO_ROOT / result["candidate_report_path"],
        REPO_ROOT / result["checkpoint_path"],
        REPO_ROOT / result["onnx_path"],
        seal_root / "opened.json",
        seal_root / "result.json",
    )
    if not all(path.is_file() for path in direct_paths):
        pytest.skip("Ignored local P1 payload and seal evidence is not present")
    assert sha256_file(direct_paths[0]) == result["candidate_report_sha256"]
    assert sha256_file(direct_paths[1]) == result["checkpoint_sha256"]
    assert sha256_file(direct_paths[2]) == result["onnx_sha256"]
    assert sha256_file(direct_paths[3]) == result["training_opened_seal_sha256"]
    assert sha256_file(direct_paths[4]) == result["training_result_seal_sha256"]


def test_p2_preflight_reuses_no_checkpoint_and_binds_aggregate_p1_only() -> None:
    config = _json(ROOT / "training/p2.json")
    selection, train_path, validation_path = verify_p2_config_and_inputs(config)
    assert sha256_file(train_path) == selection["train"]["archive_sha256"]
    assert sha256_file(validation_path) == selection["validation"]["archive_sha256"]
    assert config["p1_result_sha256"] == P1_RESULT_SHA256
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["p1_checkpoint_reused"] is False
    assert config["frozen_v11_split_reused"] is True
    assert config["unsupported_seed_addition_loss_weight"] == 3.0


def test_p2_addition_loss_penalizes_only_seed_negative_truth_negative_pixels() -> None:
    prediction = torch.tensor([[[[0.8, 0.8, 0.8, 0.2]]]], dtype=torch.float32)
    seed = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]], dtype=torch.float32)
    truth = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]], dtype=torch.float32)
    loss = _unsupported_seed_addition_loss(prediction, seed, truth)
    expected = functional.binary_cross_entropy(
        torch.tensor([0.8, 0.2]), torch.zeros(2)
    )
    assert torch.allclose(loss, expected)
    assert _unsupported_seed_addition_loss(prediction, torch.ones_like(seed), truth).item() == 0.0


def test_p2_result_binds_exact_aggregate_report_and_single_use_seals() -> None:
    result = _json(ROOT / "P2_RESULT.json")
    assert sha256_file(ROOT / "P2_RESULT.json") == P2_RESULT_SHA256
    seal_root = REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-seed-refinement-v11/P2"
    assert result["status"] == "failed_selection_consumed"
    assert result["selection_exact_scene_count"] == 122
    assert result["selection_false_positives"] == 0
    assert result["selection_false_negatives"] == 29
    assert result["artifact_precision"] == 0.782828150056521
    assert result["artifact_recall"] == 0.9603460905861702
    assert result["seed_added_pixels"] == 244692
    assert result["seed_removed_pixels"] == 1253
    assert result["case_detail_or_pixels_inspected"] is False
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    direct_paths = (
        REPO_ROOT / result["candidate_report_path"],
        REPO_ROOT / result["checkpoint_path"],
        REPO_ROOT / result["onnx_path"],
        seal_root / "opened.json",
        seal_root / "result.json",
    )
    if not all(path.is_file() for path in direct_paths):
        pytest.skip("Ignored local P2 payload and seal evidence is not present")
    assert sha256_file(direct_paths[0]) == result["candidate_report_sha256"]
    assert sha256_file(direct_paths[1]) == result["checkpoint_sha256"]
    assert sha256_file(direct_paths[2]) == result["onnx_sha256"]
    assert sha256_file(direct_paths[3]) == result["training_opened_seal_sha256"]
    assert sha256_file(direct_paths[4]) == result["training_result_seal_sha256"]


def test_p3_preflight_reuses_no_checkpoint_and_binds_aggregate_p2_only() -> None:
    config = _json(ROOT / "training/p3.json")
    selection, train_path, validation_path = verify_p3_config_and_inputs(config)
    assert sha256_file(train_path) == selection["train"]["archive_sha256"]
    assert sha256_file(validation_path) == selection["validation"]["archive_sha256"]
    assert config["p2_result_sha256"] == P2_RESULT_SHA256
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["p2_checkpoint_reused"] is False
    assert config["frozen_v11_split_reused"] is True
    assert config["unsupported_seed_addition_loss_weight"] == 3.0
    assert config["false_seed_retention_loss_weight"] == 6.0


def test_p3_retention_loss_penalizes_only_seed_positive_truth_negative_pixels() -> None:
    prediction = torch.tensor([[[[0.8, 0.8, 0.8, 0.2]]]], dtype=torch.float32)
    seed = torch.tensor([[[[1.0, 0.0, 1.0, 1.0]]]], dtype=torch.float32)
    truth = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]], dtype=torch.float32)
    loss = _false_seed_retention_loss(prediction, seed, truth)
    expected = functional.binary_cross_entropy(
        torch.tensor([0.8, 0.2]), torch.zeros(2)
    )
    assert torch.allclose(loss, expected)
    assert _false_seed_retention_loss(prediction, torch.zeros_like(seed), truth).item() == 0.0


def test_p3_result_binds_exact_aggregate_report_and_exhausts_v11() -> None:
    result = _json(ROOT / "P3_RESULT.json")
    assert sha256_file(ROOT / "P3_RESULT.json") == P3_RESULT_SHA256
    assert result["status"] == "failed_selection_consumed"
    assert result["selection_exact_scene_count"] == 112
    assert result["selection_true_positives"] == 1178
    assert result["selection_false_positives"] == 2
    assert result["selection_false_negatives"] == 38
    assert result["selection_duplicate_count"] == 0
    assert result["selection_prohibited_structure_hits"] == 0
    assert result["selection_marker_artifact_hits"] == 0
    assert result["artifact_precision"] == 0.0
    assert result["artifact_recall"] == 0.0
    assert result["seed_added_pixels"] == 0
    assert result["seed_removed_pixels"] == 289107
    assert result["onnx_parity_passed"] is True
    assert result["case_detail_or_pixels_inspected"] is False
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    seal_root = REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-seed-refinement-v11/P3"
    direct_paths = (
        REPO_ROOT / result["candidate_report_path"],
        REPO_ROOT / result["checkpoint_path"],
        REPO_ROOT / result["onnx_path"],
        seal_root / "opened.json",
        seal_root / "result.json",
    )
    if not all(path.is_file() for path in direct_paths):
        pytest.skip("Ignored local P3 payload and seal evidence is not present")
    assert sha256_file(direct_paths[0]) == result["candidate_report_sha256"]
    assert sha256_file(direct_paths[1]) == result["checkpoint_sha256"]
    assert sha256_file(direct_paths[2]) == result["onnx_sha256"]
    assert sha256_file(direct_paths[3]) == result["training_opened_seal_sha256"]
    assert sha256_file(direct_paths[4]) == result["training_result_seal_sha256"]


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


def test_visible_scene_uses_new_identity_without_private_data() -> None:
    scene = render_scene("validation", 0)
    assert scene.scene_id == "marker-seed-refinement-v11-validation-0000"
    assert scene.renderer_family in RENDERER_FAMILIES["validation"]
    assert scene.degradation_family in DEGRADATION_FAMILIES["validation"]
