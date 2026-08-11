# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.official_recognition_v1 import evaluate, prepare_split


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/official_recognition_v1"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_fixture_bytes_reproduce_without_private_or_chandler_data() -> None:
    for partition, seal_name in (
        ("selection", "SELECTION_SEAL.json"),
        ("sealed_public", "SEALED_PUBLIC_TEST_SEAL.json"),
    ):
        manifest, archive = prepare_split.build_partition(partition)
        seal = load(ROOT / seal_name)
        assert prepare_split.hash_bytes(manifest) == seal["private_manifest_sha256"]
        assert prepare_split.hash_bytes(archive) == seal["fixture_archive_sha256"]
        assert len(archive) == seal["fixture_archive_bytes"]
        parsed = json.loads(manifest)
        assert parsed["synthetic_only"] is True
        assert parsed["private_or_article_images"] is False
        assert parsed["chandler_included"] is False
        assert all(case["private_or_article_image"] is False for case in parsed["cases"])
        assert all(case["chandler_image"] is False for case in parsed["cases"])


def test_preregistration_binds_exact_model_sources_splits_and_budget() -> None:
    protocol = load(ROOT / "PROTOCOL.json")
    config_path = ROOT / "training/p1.json"
    config = load(config_path)
    gate_path = ROOT / "gates/sealed-public-p1.json"
    ledger = load(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(
        item
        for item in ledger["revisions"]
        if item["task"] == evaluate.TASK and item["revision"] == evaluate.REVISION
    )
    assert protocol["status"] == "p1_preregistered_before_inference"
    assert protocol["paired_detector_rerun"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert config["model_sha256"] == sha256_file(REPO_ROOT / evaluate.MODEL_PATH)
    assert config["inference_yaml_sha256"] == sha256_file(REPO_ROOT / evaluate.INFERENCE_YAML_PATH)
    assert config["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert config["selection_seal_sha256"] == sha256_file(ROOT / "SELECTION_SEAL.json")
    assert config["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert config["public_gate_config_sha256"] == sha256_file(gate_path)
    expected_bundle = source_bundle_sha256(REPO_ROOT, evaluate.RUNNER_SOURCE_PATHS)
    assert config["expected_runner_source_bundle_sha256"] == expected_bundle
    assert protocol["runner_source_bundle_sha256"] == expected_bundle
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_truth_hidden_public_archive_is_ignored_and_unopened() -> None:
    seal = load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    manifest_path = REPO_ROOT / seal["private_manifest_path"]
    assert sha256_file(archive_path) == seal["fixture_archive_sha256"]
    assert sha256_file(manifest_path) == seal["private_manifest_sha256"]
    tracked = subprocess.run(
        ["git", "ls-files", "--", archive_path.relative_to(REPO_ROOT).as_posix(), manifest_path.relative_to(REPO_ROOT).as_posix()],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert tracked.stdout.strip() == ""
    assert not (ROOT / "artifacts/P1-run").exists()


def test_runner_is_recognition_only_and_cannot_approve_by_itself() -> None:
    source = (ROOT / "evaluate.py").read_text(encoding="utf-8")
    assert "detect_regions" not in source
    assert "DETECTION_MODEL_ID" not in source
    assert '"marker_creation_evaluated": False' in source
    assert '"production_approval": False' in source
    assert '"release_eligible": False' in source
    assert evaluate.GATES == {
        "exact_match_minimum": 0.90,
        "character_error_rate_maximum": 0.05,
        "role_accuracy_minimum": 0.90,
        "numeric_exact_match_minimum": 0.90,
        "word_exact_match_minimum": 0.90,
        "ambiguity_exact_match_minimum": 0.90,
        "conversion_parity_maximum_absolute_error": 0.0001,
        "provider": "CPUExecutionProvider",
    }
