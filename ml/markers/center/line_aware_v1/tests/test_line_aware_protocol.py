# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import torch

from ml.markers.center.candidate_level_v1.dataset import (
    DEGRADATIONS as PRIOR_DEGRADATIONS,
    SEALED_PUBLIC_FAMILIES as PRIOR_PUBLIC_FAMILIES,
    SELECTION_FAMILIES as PRIOR_SELECTION_FAMILIES,
)
from ml.markers.center.line_aware_v1.dataset import (
    DEGRADATIONS,
    SEALED_PUBLIC_FAMILIES,
    SELECTION_FAMILIES,
    build_selection_scenes,
    selection_manifest,
)
from ml.markers.center.line_aware_v1.model import LineAwarePatchNet
from ml.markers.center.line_aware_v1.model_p2 import LineAwarePatchNetP2
from ml.markers.center.line_aware_v1.pipeline import extract_proposals, postprocess_predictions
from ml.markers.center.line_aware_v1.train_p1 import RUNNER_SOURCE_PATHS as P1_RUNNER_SOURCE_PATHS
from ml.markers.center.line_aware_v1.train_p2 import RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS, _export
from ml.markers.center.line_aware_v1.train_p3 import RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]


def test_new_families_and_degradations_are_disjoint_from_exposed_revision() -> None:
    prior_families = set(PRIOR_PUBLIC_FAMILIES)
    prior_families.update(item for values in PRIOR_SELECTION_FAMILIES.values() for item in values)
    current_families = set(SEALED_PUBLIC_FAMILIES)
    current_families.update(item for values in SELECTION_FAMILIES.values() for item in values)
    assert prior_families.isdisjoint(current_families)
    prior_degradations = {item for values in PRIOR_DEGRADATIONS.values() for item in values}
    current_degradations = {item for values in DEGRADATIONS.values() for item in values}
    assert prior_degradations.isdisjoint(current_degradations)


def test_selection_renderer_is_deterministic_and_three_channel() -> None:
    first = build_selection_scenes("validation")
    second = build_selection_scenes("validation")
    assert len(first) == 9
    assert torch.equal(first[0].tensor, second[0].tensor)
    assert tuple(first[0].tensor.shape) == (3, 168, 224)
    assert float(torch.max(first[0].tensor[1])) == 1.0
    assert float(torch.max(first[0].tensor[2])) == 1.0
    assert selection_manifest() == selection_manifest()


def test_masked_proposal_origins_are_removed_before_inference() -> None:
    scene = build_selection_scenes("validation")[0]
    proposals = extract_proposals(scene.tensor)
    divider = next(item for item in scene.prohibited if item.kind == "divider")
    distances = torch.sqrt(((proposals.coordinates - torch.tensor((divider.x, divider.y))) ** 2).sum(dim=1))
    assert bool(torch.all(distances > 4.0))


def test_regressed_masked_center_is_rejected_even_with_high_model_score() -> None:
    scene = build_selection_scenes("validation")[0]
    proposals = extract_proposals(scene.tensor)
    divider = next(item for item in scene.prohibited if item.kind == "divider")
    distances = torch.sqrt(((proposals.coordinates - torch.tensor((divider.x, divider.y))) ** 2).sum(dim=1))
    index = int(torch.argmin(distances))
    output = np.zeros((len(proposals.patches), 4), dtype=np.float32)
    output[index, 0] = 1.0
    output[index, 1] = (divider.x - float(proposals.coordinates[index, 0])) / 4.0
    output[index, 2] = (divider.y - float(proposals.coordinates[index, 1])) / 4.0
    output[index, 3] = 4.0
    assert postprocess_predictions(scene, proposals, output, threshold=0.5) == ()


def test_p1_dual_branch_model_retains_frozen_tensor_contract() -> None:
    model = LineAwarePatchNet().eval()
    output = model(torch.zeros((2, 3, 33, 33), dtype=torch.float32))
    assert tuple(output.shape) == (2, 4)
    assert model.export_contract()["architecture"] == "line-aware-dual-branch-patch-cnn-v1"
    assert model.contract.input_channels == ("ink_probability", "text_mask", "artifact_mask")


def test_p2_fixed_pool_exports_with_the_frozen_tensor_contract() -> None:
    model = LineAwarePatchNetP2().eval()
    sample = torch.zeros((2, 3, 33, 33), dtype=torch.float32)
    assert tuple(model(sample).shape) == (2, 4)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "p2-preflight.onnx"
        _export(model, sample, output)
        onnx.checker.check_model(onnx.load(output))
    assert model.export_contract()["architecture"] == "line-aware-dual-branch-patch-cnn-v2-export-safe"


def test_protocol_is_fail_closed_after_single_p3_public_gate_attempt() -> None:
    protocol = json.loads((REPO_ROOT / "ml/markers/center/line_aware_v1/PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["currently_preregistered_candidate"] is None
    assert protocol["selected_candidate"] == "P3"
    assert protocol["sealed_public_gate_authorized"] is False
    assert protocol["status"] == "exhausted_public_gate_configuration_failure"
    assert protocol["consumed_candidates"] == ["P1", "P2", "P3"]
    assert protocol["experiment_budget"] == 3
    assert protocol["prior_candidate_bytes_reused"] is False
    assert protocol["public_gate_budget"] == 1
    assert protocol["public_gate_attempts"] == 1
    assert protocol["public_gate_evaluations"] == 0
    assert protocol["public_gate_budget_consumed"] is True
    failure_path = REPO_ROOT / protocol["public_gate_attempt_failure_path"]
    assert protocol["public_gate_attempt_failure_sha256"] == sha256_file(failure_path)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "fail_closed_before_seal"
    assert failure["attempt_count"] == 1
    assert failure["evaluation_count"] == 0
    assert failure["public_archive_loaded"] is False
    assert failure["gate_output_created"] is False
    assert failure["opened_seal_created"] is False
    assert failure["result_seal_created"] is False
    assert failure["public_gate_budget_consumed"] is True
    assert failure["rerun_authorized"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_frozen_split_and_budget_bind_the_single_authorized_candidate() -> None:
    selection = REPO_ROOT / "ml/markers/center/line_aware_v1/SELECTION_MANIFEST.json"
    seal_path = REPO_ROOT / "ml/markers/center/line_aware_v1/SEALED_PUBLIC_TEST_SEAL.json"
    p1_config_path = REPO_ROOT / "ml/markers/center/line_aware_v1/training/p1.json"
    p2_config_path = REPO_ROOT / "ml/markers/center/line_aware_v1/training/p2.json"
    config_path = REPO_ROOT / "ml/markers/center/line_aware_v1/training/p3.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert seal["scene_count"] == 16
    assert seal["truth_hidden_from_training_runner"] is True
    assert seal["fixture_archive_sha256"] == "69e905d2ae2a07544e2426446f1fc7e0008d7701d7831103c74a1cf7ed9795b6"
    assert config["selection_manifest_sha256"] == sha256_file(selection)
    assert config["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert sha256_file(p1_config_path) == "8b252b458131c5c9b66b47ef8450d16cf5829903f2546e125b4f915a90512e76"
    assert source_bundle_sha256(REPO_ROOT, P1_RUNNER_SOURCE_PATHS) == "eb6d6c5c7b9a960ff86ab9c010c6bc32a1fd856df327699aa21733537e72c26a"
    assert sha256_file(p2_config_path) == "9b329aa26a9261d445d8db48ce2f9c83e18475f723bf71f47ccb68ab4704d5a0"
    assert source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS) == "aeee5089cfe9085cd0ca765ff106caa77748e23b4073f291cb171efcd1e705d7"
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS)
    assert config["selection_threshold"] == 0.07
    assert config["optimizer_steps"] == 0
    assert config["weights_changed"] is False
    ledger = json.loads((REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["revision"] == "marker-center-line-aware-v1")
    assert entry["status"] == "exhausted_public_gate_configuration_failure"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(config_path)
    assert entry["p1_training_report_sha256"] == "9e3bee532c852ba13c3bdde9390ab9dcbf1a1fcf091072f802dddefe20b6d56c"
    assert entry["p1_training_opened_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-line-aware-v1/P1/opened.json"
    )
    assert entry["p1_training_result_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-line-aware-v1/P1/result.json"
    )
    assert entry["p1_optimizer_steps"] == 0
    assert entry["p1_public_gate_evaluations"] == 0
    assert entry["p2_training_report_sha256"] == "3c94148b84f88b30568276ff0650f08bfbc23e123fc1ab310c9c72f20aabcda7"
    assert entry["p2_training_opened_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-line-aware-v1/P2/opened.json"
    )
    assert entry["p2_training_result_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-line-aware-v1/P2/result.json"
    )
    assert entry["p2_checkpoint_sha256"] == config["source_checkpoint_sha256"]
    assert entry["p2_onnx_sha256"] == config["source_onnx_sha256"]
    assert entry["p2_public_gate_evaluations"] == 0
    assert entry["p3_training_report_sha256"] == "7f1ddf0955dbc3de85c8ede6a8bb874a89d68428dd1b9cdbbd391ebc09d9cdfc"
    assert entry["p3_training_opened_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-line-aware-v1/P3/opened.json"
    )
    assert entry["p3_training_result_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-line-aware-v1/P3/result.json"
    )
    assert entry["p3_selection_exact_scene_count"] == entry["p3_selection_scene_count"] == 9
    assert entry["p3_selection_true_positives"] == 63
    assert entry["p3_selection_false_positives"] == 0
    assert entry["p3_selection_false_negatives"] == 0
    assert entry["p3_optimizer_steps"] == 0
    assert entry["p3_weights_changed"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] is None
    assert entry["public_gate_authorized_onnx_sha256"] == entry["p3_onnx_sha256"]
    assert entry["public_gate_authorized_training_report_sha256"] == entry["p3_training_report_sha256"]
    assert entry["public_gate_attempted_candidate_id"] == "P3"
    assert entry["public_gate_attempts"] == 1
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_attempt_failure_sha256"] == sha256_file(
        REPO_ROOT / entry["public_gate_attempt_failure_path"]
    )
    assert entry["public_gate_archive_loaded"] is False
    assert entry["public_gate_output_created"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
