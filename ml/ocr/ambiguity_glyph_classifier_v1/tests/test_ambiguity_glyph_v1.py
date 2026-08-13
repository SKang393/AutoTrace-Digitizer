# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.ambiguity_glyph_classifier_v1 import dataset, sealed_gate, train_p1
from ml.ocr.ambiguity_glyph_classifier_v1.model import AmbiguityGlyphNet
from ml.ocr.ambiguity_glyph_classifier_v1.protocol import GATES, GLYPHS, REVISION, protocol_configuration


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/ambiguity_glyph_classifier_v1"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_model_contract_is_export_safe_and_class_order_is_fixed() -> None:
    model = AmbiguityGlyphNet().eval()
    values = torch.zeros((5, 1, 24, 24), dtype=torch.float32)
    assert tuple(model(values).shape) == (5, 4)
    assert GLYPHS == ("O", "o", "l", "I")


def test_fresh_train_and_validation_splits_reproduce_frozen_fingerprints() -> None:
    selection = _load(ROOT / "SELECTION_MANIFEST.json")
    assert dataset.split_fingerprint("train") == selection["train_split_fingerprint"]
    assert dataset.split_fingerprint("validation") == selection["validation_split_fingerprint"]
    _, _, train_values, train_labels = dataset.build_partition("train")
    _, _, validation_values, validation_labels = dataset.build_partition("validation")
    assert train_values.shape == (2560, 1, 24, 24)
    assert validation_values.shape == (640, 1, 24, 24)
    assert np.bincount(train_labels).tolist() == [640, 640, 640, 640]
    assert np.bincount(validation_labels).tolist() == [160, 160, 160, 160]


def test_public_split_reproduces_but_remains_unopened_for_execution() -> None:
    seal = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    manifest, archive, values, labels = dataset.build_partition("sealed_public")
    assert dataset.hash_bytes(manifest) == seal["private_manifest_sha256"]
    assert dataset.hash_bytes(archive) == seal["fixture_archive_sha256"]
    assert values.shape == (960, 1, 24, 24)
    assert np.bincount(labels).tolist() == [240, 240, 240, 240]
    assert seal["truth_hidden_from_model_execution_until_gate"] is True
    assert seal["public_gate_evaluations"] == 0


def test_preregistration_binds_sources_splits_trigger_and_license() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    config = _load(ROOT / "training/p1.json")
    ledger = _load(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert protocol == protocol_configuration(runner_source_bundle_sha256=source_bundle_sha256(REPO_ROOT, train_p1.RUNNER_SOURCE_PATHS))
    assert config["expected_runner_source_bundle_sha256"] == protocol["runner_source_bundle_sha256"]
    assert config["trigger_result_sha256"] == sha256_file(REPO_ROOT / config["trigger_result_path"])
    assert config["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert config["selection_manifest_sha256"] == sha256_file(ROOT / "SELECTION_MANIFEST.json")
    assert config["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert config["public_gate_config_sha256"] == sha256_file(ROOT / "gates/sealed-public-v1.json")
    assert config["model_license"] == protocol["model_license"] == "Apache-2.0"
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["execution_authorized"] is True
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_archive_opened"] is False


def test_public_gate_identity_is_frozen_and_never_approves_release() -> None:
    config = _load(ROOT / "gates/sealed-public-v1.json")
    assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, sealed_gate.EVALUATOR_SOURCE_PATHS)
    assert config["evaluation_limit"] == 1
    assert config["production_approval"] is False
    assert config["release_eligible"] is False
    assert GATES["sealed_per_class_accuracy_minimum"] == 0.95


def test_no_manifest_or_model_store_promotion_exists() -> None:
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*ambiguity*glyph*.json"))
    index = _load(REPO_ROOT / "artifacts/production-model-store/production-model-index.json")
    assert REVISION not in json.dumps(index, sort_keys=True)
