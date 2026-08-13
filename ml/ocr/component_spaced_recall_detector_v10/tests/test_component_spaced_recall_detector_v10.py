# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed V10 preregistration tests."""

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.component_spaced_recall_detector_v10.dataset import build_split, proposal_summary, split_fingerprint
from ml.ocr.component_spaced_recall_detector_v10.protocol import BASE_ONNX_SHA256, REVISION, SPLITS, protocol_configuration
from ml.ocr.component_spaced_recall_detector_v10.sealed_gate import EVALUATOR_SOURCE_PATHS, SPLIT_CONFIG_PATH


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_spaced_recall_detector_v10"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_splits_are_fresh_disjoint_and_proposal_complete() -> None:
    selection, seal = _load(ROOT / "SELECTION_MANIFEST.json"), _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    fingerprints: set[str] = set()
    for registration in SPLITS:
        scenes = build_split(registration.split)
        summary, fingerprint = proposal_summary(scenes), split_fingerprint(scenes)
        expected = selection[registration.split] if registration.split != "sealed_public" else seal
        assert len(scenes) == registration.scene_count
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        assert summary["positive_proposal_count"] == summary["truth_region_count"]
        assert summary == {key: expected[key] for key in summary}
        assert fingerprint == expected["split_fingerprint"]


def test_p1_is_zero_optimizer_exact_onnx_threshold_only() -> None:
    protocol, config = _load(ROOT / "PROTOCOL.json"), _load(ROOT / "training/p1.json")
    expected = json.loads(json.dumps(protocol_configuration()))
    expected["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected
    assert config["optimizer_steps"] == 0
    assert config["weights_changed"] is False
    assert config["base_onnx_sha256"] == BASE_ONNX_SHA256
    assert config["predecessor_fixture_bytes_reused"] is False


def test_ledger_authorizes_only_unused_p1() -> None:
    ledger = _load(LEDGER)
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is False


def test_public_gate_is_hidden_bound_and_unapproved() -> None:
    seal, gate = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json"), _load(REPO_ROOT / SPLIT_CONFIG_PATH)
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    assert not (ROOT / "P1_RESULT.json").exists()
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*spaced*recall*v10*.json"))

