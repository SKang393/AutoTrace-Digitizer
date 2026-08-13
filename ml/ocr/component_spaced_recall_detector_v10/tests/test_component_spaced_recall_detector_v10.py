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
from ml.ocr.component_spaced_recall_detector_v10.train_p3 import RUNNER_SOURCE_PATHS
from ml.ocr.component_spaced_recall_detector_v10.training_data_p2 import NEGATIVE_CAP_PER_SCENE


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


def test_ledger_records_selected_p3_and_authorized_public_gate() -> None:
    ledger = _load(LEDGER)
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert entry["status"] == "candidate_3_selected_public_gate_pending"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(ROOT / "training/p3.json")
    assert entry["p3_result_sha256"] == sha256_file(ROOT / "P3_RESULT.json")
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_authorized_candidate_id"] == "P3"
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False


def test_p2_preregistration_records_training_only_examples() -> None:
    config = _load(ROOT / "training/p2.json")
    assert config["candidate_id"] == "P2"
    assert config["experiment_ordinal"] == 2
    assert config["source_checkpoint_sha256"] == sha256_file(
        REPO_ROOT / config["source_checkpoint_path"]
    )
    assert config["expected_runner_source_bundle_sha256"] == (
        "430a7da2fac1bd3157351bc4c90ff7a12d7e69e0c4bd6a7bae5ed6fcd37da31f"
    )
    assert config["negative_cap_per_scene"] == NEGATIVE_CAP_PER_SCENE
    assert config["proposal_count"] == 7999
    assert config["positive_proposal_count"] == 1200
    assert config["negative_proposal_count"] == 6799
    assert config["tensor_label_stream_sha256"] == (
        "1f1bc23cd4ed06f95d638bbf53de27089b5572f6abe91d824f4129050abdbf9c"
    )
    assert config["validation_or_public_pixels_used_for_training"] is False
    assert config["public_gate_archive_opened"] is False
    assert config["production_approval"] is False


def test_p3_preregistration_binds_exact_p2_failure_and_runner() -> None:
    config = _load(ROOT / "training/p3.json")
    p2 = _load(ROOT / "P2_RESULT.json")
    assert config["candidate_id"] == "P3"
    assert config["experiment_ordinal"] == 3
    assert config["optimizer_steps"] == 0
    assert config["weights_changed"] is False
    assert config["p2_result_sha256"] == sha256_file(ROOT / "P2_RESULT.json")
    assert p2["status"] == "failed_runner"
    assert p2["failure_phase"] == "selection"
    assert p2["selection_metrics_available_for_approval"] is False
    assert p2["public_gate_archive_opened"] is False
    assert config["p2_checkpoint_sha256"] == p2["checkpoint_sha256"]
    assert config["p2_onnx_sha256"] == p2["onnx_sha256"]
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS
    )
    assert config["public_gate_archive_opened"] is False
    assert config["production_approval"] is False


def test_public_gate_is_hidden_bound_and_unapproved() -> None:
    seal, gate = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json"), _load(REPO_ROOT / SPLIT_CONFIG_PATH)
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    result = _load(ROOT / "P1_RESULT.json")
    assert result["status"] == "failed_selection"
    assert result["selection_true_positives"] == 396
    assert result["selection_false_negatives"] == 4
    assert result["selection_false_positives"] == 0
    assert result["public_gate_archive_opened"] is False
    selected = _load(ROOT / "P3_RESULT.json")
    assert selected["status"] == "selected_public_gate_pending"
    assert selected["public_gate_authorized"] is True
    assert selected["public_gate_evaluations"] == 0
    assert selected["public_gate_archive_opened"] is False
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*spaced*recall*v10*.json"))
