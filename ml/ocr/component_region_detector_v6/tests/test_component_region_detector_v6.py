# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_region_detector_v6.dataset import (
    build_split,
    encode_proposal,
    load_sealed_public_archive,
    proposal_labels,
    proposal_summary,
    proposals,
    split_fingerprint,
)
from ml.ocr.component_region_detector_v6.prepare_split import SPLIT_SOURCE_PATHS
from ml.ocr.component_region_detector_v6.protocol import ENCODED_WIDTH, REVISION, TASK, protocol_configuration
from ml.ocr.component_region_detector_v6.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_region_detector_v6"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def test_fresh_splits_are_deterministic_disjoint_and_proposal_complete() -> None:
    train = build_split("train")
    validation = build_split("validation")
    assert split_fingerprint(train) == split_fingerprint(build_split("train"))
    assert split_fingerprint(train) != split_fingerprint(validation)
    assert {scene.renderer_family for scene in train}.isdisjoint({scene.renderer_family for scene in validation})
    assert all(scene.raster.shape == (256, 512) and scene.raster.dtype == np.uint8 for scene in validation)
    summary = proposal_summary(validation)
    assert summary["scene_count"] == 48
    assert summary["truth_region_count"] == 144
    assert summary["positive_proposal_count"] == 144
    assert summary["negative_proposal_count"] > 0


def test_proposal_encoding_and_order_are_frozen() -> None:
    scene = build_split("validation")[0]
    candidates = proposals(scene.raster)
    assert list(candidates) == sorted(candidates, key=lambda item: (item.top, item.left, item.bottom, item.right))
    assert proposal_labels(scene, candidates).sum() == len(scene.truths)
    encoded = encode_proposal(scene.raster, candidates[0])
    assert encoded.shape == (1, 32, ENCODED_WIDTH)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()
    assert float(encoded.min()) >= 0.0
    assert float(encoded.max()) <= 1.0


def test_frozen_split_and_gate_hashes_are_directly_bound() -> None:
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
    expected_protocol = json.loads(json.dumps(protocol_configuration()))
    expected_protocol["split_generator_source_paths"] = [p.as_posix() for p in SPLIT_SOURCE_PATHS]
    expected_protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    assert protocol == expected_protocol
    seal_path = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["chandler_included"] is False
    assert seal["generalization_label_included"] is False
    assert seal["predecessor_public_archive_reused"] is False
    assert seal["truth_hidden_from_training_runner"] is True
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    scenes = load_sealed_public_archive(REPO_ROOT / seal["fixture_archive_path"])
    assert split_fingerprint(scenes) == seal["split_fingerprint"]
    assert proposal_summary(scenes)["positive_proposal_count"] == seal["truth_region_count"]
    gate = json.loads((ROOT / "gates/sealed-public-v1.json").read_text(encoding="utf-8"))
    assert gate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    assert gate["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))


def test_budget_records_split_freeze_without_training_authorization() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["task"] == TASK and item["revision"] == REVISION)
    assert entry["status"] == "available"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["selection_manifest_sha256"] == sha256_file(ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert entry["public_gate_config_sha256"] == sha256_file(ROOT / "gates/sealed-public-v1.json")
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
