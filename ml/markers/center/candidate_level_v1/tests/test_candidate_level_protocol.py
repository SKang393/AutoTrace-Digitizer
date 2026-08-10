# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.markers.center.candidate_level_v1.dataset import (
    DEGRADATIONS,
    SEALED_PUBLIC_FAMILIES,
    SELECTION_FAMILIES,
    build_sealed_public_scenes,
    build_selection_scenes,
    load_sealed_public_archive,
    save_sealed_public_archive,
    selection_manifest,
)
from ml.markers.center.candidate_level_v1.model import CandidatePatchNet
from ml.markers.center.candidate_level_v1.prepare_split import SOURCE_PATHS as SPLIT_SOURCE_PATHS
from ml.markers.center.candidate_level_v1.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.markers.center.candidate_level_v1.train_p1 import RUNNER_SOURCE_PATHS
from ml.markers.center.candidate_level_v1.pipeline import (
    PROPOSAL_STRIDE,
    extract_proposals,
    label_proposals,
    postprocess_predictions,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = REPO_ROOT / "ml/markers/center/candidate_level_v1"


def test_renderer_and_degradation_families_are_split_disjoint() -> None:
    train_families = set(SELECTION_FAMILIES["train"])
    validation_families = set(SELECTION_FAMILIES["validation"])
    public_families = set(SEALED_PUBLIC_FAMILIES)
    assert train_families.isdisjoint(validation_families)
    assert train_families.isdisjoint(public_families)
    assert validation_families.isdisjoint(public_families)
    train_degradation = set(DEGRADATIONS["train"])
    validation_degradation = set(DEGRADATIONS["validation"])
    public_degradation = set(DEGRADATIONS["sealed_public"])
    assert train_degradation.isdisjoint(validation_degradation)
    assert train_degradation.isdisjoint(public_degradation)
    assert validation_degradation.isdisjoint(public_degradation)


def test_selection_manifest_is_deterministic_synthetic_and_excludes_chandler() -> None:
    first = selection_manifest()
    second = selection_manifest()
    assert first == second
    assert first["synthetic_only"] is True
    assert first["public_or_private_images"] is False
    assert first["chandler_included"] is False
    assert len(first["cases"]) == 33
    assert all("chandler" not in case["scene_id"].lower() for case in first["cases"])


@pytest.mark.parametrize("split", ("train", "validation"))
def test_every_truth_center_has_a_positive_grid_proposal(split: str) -> None:
    for scene in build_selection_scenes(split):
        proposals = extract_proposals(scene.tensor)
        distances = torch.cdist(proposals.coordinates, torch.tensor(scene.centers))
        assert torch.all(distances.min(dim=0).values <= 3.0), scene.scene_id
        examples = label_proposals(scene, proposals)
        assert int(torch.count_nonzero(examples.labels)) >= len(scene.centers)


def test_candidate_model_contract_and_activated_ranges() -> None:
    model = CandidatePatchNet()
    model.eval()
    with torch.inference_mode():
        output = model(torch.rand((5, 3, 33, 33), generator=torch.Generator().manual_seed(7)))
    assert output.shape == (5, 4)
    assert torch.all((output[:, 0] >= 0) & (output[:, 0] <= 1))
    assert torch.all((output[:, 1:3] >= -0.75) & (output[:, 1:3] <= 0.75))
    assert torch.all((output[:, 3] >= 2.5) & (output[:, 3] <= 8.0))
    assert model.contract.proposal_stride == PROPOSAL_STRIDE
    assert model.contract.runtime_revision == "marker-center-candidate-runtime-v1"


def test_raw_mask_max_gate_rejects_an_axis_candidate() -> None:
    scene = build_selection_scenes("validation")[0]
    proposals = extract_proposals(scene.tensor)
    axis_point = torch.tensor((28.0, 62.0))
    index = int(torch.argmin(torch.linalg.vector_norm(proposals.coordinates - axis_point, dim=1)))
    output = np.zeros((len(proposals.patches), 4), dtype=np.float32)
    output[index] = (0.99, 0.0, 0.0, 4.0)
    assert postprocess_predictions(scene, proposals, output, threshold=0.7) == ()


def test_sealed_archive_round_trip_keeps_exact_fixture_bytes(tmp_path: Path) -> None:
    scenes = build_sealed_public_scenes(987_654)
    path = tmp_path / "fixtures.npz"
    save_sealed_public_archive(scenes, path)
    restored = load_sealed_public_archive(path)
    assert [scene.scene_id for scene in restored] == [scene.scene_id for scene in scenes]
    for expected, actual in zip(scenes, restored, strict=True):
        assert torch.equal(expected.tensor, actual.tensor)
        assert expected.centers == actual.centers
        assert expected.radii == actual.radii
        assert expected.prohibited == actual.prohibited


def test_training_runner_cannot_open_or_import_sealed_fixture_loader() -> None:
    source = (PACKAGE_ROOT / "train_p1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "load_sealed_public_archive" not in imported_names
    assert "np.load" not in source


def test_protocol_declares_new_defect_class_and_keeps_production_closed() -> None:
    protocol = json.loads((PACKAGE_ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["revision"] == "marker-center-candidate-level-v1"
    assert protocol["prior_revision"] == "marker-center-production-repair-v2"
    assert protocol["prior_revision_reuse"] is False
    assert protocol["experiment_budget"] == 3
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["public_contract_schema_changes"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_split_config_and_candidate_are_fully_hash_bound_and_fail_closed() -> None:
    selection_path = PACKAGE_ROOT / "SELECTION_MANIFEST.json"
    seal_path = PACKAGE_ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    candidate_path = PACKAGE_ROOT / "training/p1.json"
    gate_path = PACKAGE_ROOT / "gates/sealed-public-v1.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert candidate["selection_manifest_sha256"] == sha256_file(selection_path)
    assert candidate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert seal["split_generator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, SPLIT_SOURCE_PATHS
    )
    assert candidate["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS
    )
    assert gate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert gate["expected_dataset_manifest_sha256"] == seal["private_manifest_sha256"]
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert gate["expected_gate_config_sha256"] == hashlib.sha256(
        canonical_json_bytes(GATE_CONFIG)
    ).hexdigest()
    assert gate["production_approval"] is False
    assert gate["release_eligible"] is False


def test_canonical_budget_authorizes_only_unused_p1_for_new_revision() -> None:
    ledger = json.loads(
        (REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item
        for item in ledger["revisions"]
        if item["revision"] == "marker-center-candidate-level-v1"
    )
    candidate_path = PACKAGE_ROOT / "training/p1.json"
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["execution_authorized"] is True
    assert entry["public_gate_authorized"] is False
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(candidate_path)
    prior = next(
        item
        for item in ledger["revisions"]
        if item["revision"] == "marker-center-production-repair-v2"
    )
    assert prior["status"] == "exhausted_failed_public_gate"
    assert prior["execution_authorized"] is False
