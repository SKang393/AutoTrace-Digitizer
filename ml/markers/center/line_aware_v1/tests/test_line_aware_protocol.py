# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
from ml.markers.center.line_aware_v1.pipeline import extract_proposals, postprocess_predictions
from ml.markers.center.line_aware_v1.train_p1 import RUNNER_SOURCE_PATHS
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


def test_dual_branch_model_has_export_safe_contract() -> None:
    model = LineAwarePatchNet().eval()
    output = model(torch.zeros((2, 3, 33, 33), dtype=torch.float32))
    assert tuple(output.shape) == (2, 4)
    assert model.export_contract()["architecture"] == "line-aware-dual-branch-patch-cnn-v1"
    assert model.contract.input_channels == ("ink_probability", "text_mask", "artifact_mask")


def test_protocol_is_fail_closed_and_p1_only() -> None:
    protocol = json.loads((REPO_ROOT / "ml/markers/center/line_aware_v1/PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["experiment_budget"] == 3
    assert protocol["prior_candidate_bytes_reused"] is False
    assert protocol["public_gate_budget"] == 1
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_frozen_split_and_budget_bind_the_single_authorized_candidate() -> None:
    selection = REPO_ROOT / "ml/markers/center/line_aware_v1/SELECTION_MANIFEST.json"
    seal_path = REPO_ROOT / "ml/markers/center/line_aware_v1/SEALED_PUBLIC_TEST_SEAL.json"
    config_path = REPO_ROOT / "ml/markers/center/line_aware_v1/training/p1.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert seal["scene_count"] == 16
    assert seal["truth_hidden_from_training_runner"] is True
    assert seal["fixture_archive_sha256"] == "69e905d2ae2a07544e2426446f1fc7e0008d7701d7831103c74a1cf7ed9795b6"
    assert config["selection_manifest_sha256"] == sha256_file(selection)
    assert config["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    ledger = json.loads((REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["revision"] == "marker-center-line-aware-v1")
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["public_gate_authorized"] is False
