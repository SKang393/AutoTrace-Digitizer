# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.ambiguity_source_group_classifier_v3 import dataset, sealed_gate, train_p1, train_p2
from ml.ocr.ambiguity_source_group_classifier_v3.crop import active_groups, group_tensor
from ml.ocr.ambiguity_source_group_classifier_v3.model import SourceGroupAmbiguityNet
from ml.ocr.ambiguity_source_group_classifier_v3.protocol import (
    GATES,
    GLYPHS,
    IMAGE_SIZE,
    REVISION,
    protocol_configuration,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/ambiguity_source_group_classifier_v3"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_model_contract_and_class_order_are_fixed() -> None:
    values = torch.zeros((5, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    assert tuple(SourceGroupAmbiguityNet().eval()(values).shape) == (5, 4)
    assert GLYPHS == ("O", "o", "l", "I")


def test_fresh_selection_splits_reproduce_and_retain_case_scale() -> None:
    selection = _load(ROOT / "SELECTION_MANIFEST.json")
    assert dataset.split_fingerprint("train") == selection["train_split_fingerprint"]
    assert dataset.split_fingerprint("validation") == selection["validation_split_fingerprint"]
    _, _, values, labels = dataset.build_partition("validation")
    active_heights = np.asarray([
        np.count_nonzero(np.max(value[0], axis=1) > 0.08) for value in values
    ])
    active_widths = np.asarray([
        np.count_nonzero(np.max(value[0], axis=0) > 0.08) for value in values
    ])
    assert float(np.mean(active_heights[labels == 0]) - np.mean(active_heights[labels == 1])) >= 2.0
    assert float(np.mean(active_widths[labels == 0]) - np.mean(active_widths[labels == 1])) >= 2.0
    assert selection["exact_production_crop_adapter"] is True
    assert selection["prior_exposed_fixture_bytes_reused"] is False


def test_frozen_source_bytes_reconstruct_the_exact_adapter_tensor() -> None:
    seal = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    manifest = _load(REPO_ROOT / str(seal["private_manifest_path"]))
    archive_bytes = (REPO_ROOT / str(seal["fixture_archive_path"])).read_bytes()
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        sample = manifest["samples"][0]
        source = archive.read(sample["source_path"])
    assert dataset.hash_bytes(source) == sample["source_sha256"]
    with Image.open(BytesIO(source)) as image:
        groups = active_groups(image.convert("L"))
        tensor = group_tensor(image.convert("L"), groups, int(sample["target_group_index"]))
    assert len(groups) == int(sample["group_count"]) == 3
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert train_p1.RUNNER_SOURCE_PATHS[0].name == "crop.py"
    assert sealed_gate.EVALUATOR_SOURCE_PATHS[0].name == "crop.py"


def test_sealed_public_bytes_reproduce_without_candidate_execution() -> None:
    seal = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    manifest, archive, values, labels = dataset.build_partition("sealed_public")
    assert dataset.hash_bytes(manifest) == seal["private_manifest_sha256"]
    assert dataset.hash_bytes(archive) == seal["fixture_archive_sha256"]
    tensor_label_sha256 = dataset.hash_bytes(
        np.ascontiguousarray(values).tobytes() + np.ascontiguousarray(labels).tobytes()
    )
    assert tensor_label_sha256 == seal["tensor_label_stream_sha256"]
    assert values.shape == (960, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert np.bincount(labels).tolist() == [240, 240, 240, 240]
    assert seal["truth_hidden_from_model_execution_until_gate"] is True
    assert seal["public_gate_evaluations"] == 0


def test_preregistration_binds_sources_splits_trigger_and_license() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    config = _load(ROOT / "training/p2.json")
    ledger = _load(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    runner_sha256 = source_bundle_sha256(REPO_ROOT, train_p2.RUNNER_SOURCE_PATHS)
    assert protocol == protocol_configuration(runner_source_bundle_sha256=runner_sha256)
    assert config["expected_runner_source_bundle_sha256"] == runner_sha256
    assert config["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert config["selection_manifest_sha256"] == sha256_file(ROOT / "SELECTION_MANIFEST.json")
    assert config["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert config["public_gate_config_sha256"] == sha256_file(ROOT / "gates/sealed-public-v1.json")
    assert config["model_license"] == protocol["model_license"] == "Apache-2.0"
    assert config["p1_result_sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert config["p1_checkpoint_sha256"] == sha256_file(REPO_ROOT / config["p1_checkpoint_path"])
    assert entry["status"] == "selection_passed_public_preregistered"
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(ROOT / "training/p2.json")
    assert entry["p1_result_sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert entry["p1_onnx_parity_passed"] is False
    assert entry["p2_selection_report_sha256"] == sha256_file(
        REPO_ROOT / "ml/ocr/ambiguity_source_group_classifier_v3/artifacts/P2-run/candidate-report.json"
    )
    assert entry["p2_onnx_parity_passed"] is True
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_authorized_candidate_id"] == "P2"
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False


def test_public_gate_identity_is_frozen_and_does_not_approve_release() -> None:
    config = _load(ROOT / "gates/sealed-public-v1.json")
    assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, sealed_gate.EVALUATOR_SOURCE_PATHS
    )
    assert config["evaluation_limit"] == 1
    assert config["production_approval"] is False
    assert config["release_eligible"] is False
    assert GATES["sealed_per_class_accuracy_minimum"] == 0.95


def test_no_manifest_or_model_store_promotion_exists() -> None:
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*ambiguity*source*group*.json"))
    index_path = REPO_ROOT / "artifacts/production-model-store/production-model-index.json"
    if index_path.is_file():
        assert REVISION not in json.dumps(_load(index_path), sort_keys=True)
