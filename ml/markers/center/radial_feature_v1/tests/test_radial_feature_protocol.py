# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from ml.markers.center.candidate_level_v1.dataset import (
    DEGRADATIONS as CANDIDATE_DEGRADATIONS,
    SEALED_PUBLIC_FAMILIES as CANDIDATE_PUBLIC_FAMILIES,
    SELECTION_FAMILIES as CANDIDATE_SELECTION_FAMILIES,
)
from ml.markers.center.line_aware_v1.dataset import (
    DEGRADATIONS as LINE_DEGRADATIONS,
    SEALED_PUBLIC_FAMILIES as LINE_PUBLIC_FAMILIES,
    SELECTION_FAMILIES as LINE_SELECTION_FAMILIES,
)
from ml.markers.center.radial_feature_v1.dataset import (
    DEGRADATIONS, SEALED_PUBLIC_FAMILIES, SELECTION_FAMILIES,
    build_selection_scenes, selection_manifest,
)
from ml.markers.center.radial_feature_v1.model import RadialFeatureNet
from ml.markers.center.radial_feature_v1.prepare_split import SOURCE_PATHS as SPLIT_SOURCE_PATHS
from ml.markers.center.radial_feature_v1.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.markers.center.radial_feature_v1.train_p1 import RUNNER_SOURCE_PATHS, _export
from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]


def _all_families(values: dict[str, tuple[str, ...]], public: tuple[str, ...]) -> set[str]:
    result = set(public)
    result.update(item for group in values.values() for item in group)
    return result


def _all_degradations(values: dict[str, tuple[str, ...]]) -> set[str]:
    return {item for group in values.values() for item in group}


def test_families_and_degradations_are_disjoint_from_exposed_revisions() -> None:
    current_families = _all_families(SELECTION_FAMILIES, SEALED_PUBLIC_FAMILIES)
    prior_families = _all_families(CANDIDATE_SELECTION_FAMILIES, CANDIDATE_PUBLIC_FAMILIES)
    prior_families.update(_all_families(LINE_SELECTION_FAMILIES, LINE_PUBLIC_FAMILIES))
    assert current_families.isdisjoint(prior_families)
    current_degradations = _all_degradations(DEGRADATIONS)
    prior_degradations = _all_degradations(CANDIDATE_DEGRADATIONS)
    prior_degradations.update(_all_degradations(LINE_DEGRADATIONS))
    assert current_degradations.isdisjoint(prior_degradations)


def test_renderer_is_deterministic_and_three_channel() -> None:
    first = build_selection_scenes("validation")
    second = build_selection_scenes("validation")
    assert len(first) == 9
    assert torch.equal(first[0].tensor, second[0].tensor)
    assert tuple(first[0].tensor.shape) == (3, 168, 224)
    assert selection_manifest() == selection_manifest()


def test_non_convolutional_model_exports_dynamic_candidate_count() -> None:
    model = RadialFeatureNet().eval()
    assert not any(isinstance(module, torch.nn.Conv2d) for module in model.modules())
    sample = torch.zeros((3, 3, 33, 33), dtype=torch.float32)
    assert tuple(model(sample).shape) == (3, 4)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "radial.onnx"
        _export(model, sample, output)
        assert output.stat().st_size > 0


def test_preregistration_binds_source_split_and_canonical_gate_schema() -> None:
    protocol = json.loads((REPO_ROOT / "ml/markers/center/radial_feature_v1/PROTOCOL.json").read_text(encoding="utf-8"))
    selection_path = REPO_ROOT / "ml/markers/center/radial_feature_v1/SELECTION_MANIFEST.json"
    seal_path = REPO_ROOT / "ml/markers/center/radial_feature_v1/SEALED_PUBLIC_TEST_SEAL.json"
    gate_path = REPO_ROOT / "ml/markers/center/radial_feature_v1/gates/sealed-public-v1.json"
    config_path = REPO_ROOT / "ml/markers/center/radial_feature_v1/training/p1.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["consumed_candidates"] == []
    assert protocol["prior_candidate_bytes_reused"] is False
    assert seal["scene_count"] == 16
    assert seal["truth_hidden_from_training_runner"] is True
    assert seal["chandler_included"] is False
    assert seal["split_generator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    assert sha256_file(selection_path) == config["selection_manifest_sha256"]
    assert sha256_file(seal_path) == config["sealed_public_test_seal_sha256"]
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    assert gate["expected_candidate_hash_keys"] == ["onnx_sha256"]
    assert gate["expected_dataset_manifest_sha256"] == seal["private_manifest_sha256"]
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    assert gate["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))
    assert gate["production_approval"] is False
    assert gate["release_eligible"] is False


def test_canonical_budget_authorizes_only_radial_p1() -> None:
    config_path = REPO_ROOT / "ml/markers/center/radial_feature_v1/training/p1.json"
    ledger = json.loads((REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["revision"] == "marker-center-radial-feature-v1")
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(
        REPO_ROOT / entry["sealed_public_test_seal_path"]
    )
    assert entry["public_gate_config_sha256"] == sha256_file(
        REPO_ROOT / entry["public_gate_config_path"]
    )
    assert entry["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    assert entry["expected_public_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
