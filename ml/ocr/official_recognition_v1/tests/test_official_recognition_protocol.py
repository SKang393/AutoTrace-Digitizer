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


def test_preregistration_and_consumed_result_bind_exact_model_sources_splits_and_budget() -> None:
    protocol = load(ROOT / "PROTOCOL.json")
    config_path = ROOT / "training/p1.json"
    config = load(config_path)
    gate_path = ROOT / "gates/sealed-public-p1.json"
    result_path = ROOT / "P1_RESULT.json"
    result = load(result_path)
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
    gate = load(gate_path)
    assert gate["expected_candidate_hash_keys"] == ["onnx_sha256", "selection_report_sha256"]
    assert gate["expected_dataset_manifest_sha256"] == gate["private_manifest_sha256"]
    assert gate["expected_gate_config_sha256"] == evaluate._hash_bytes(
        evaluate.canonical_json_bytes(dict(evaluate.PUBLIC_GATE_CONFIG))
    )
    expected_bundle = source_bundle_sha256(REPO_ROOT, evaluate.RUNNER_SOURCE_PATHS)
    assert config["expected_runner_source_bundle_sha256"] == expected_bundle
    assert protocol["runner_source_bundle_sha256"] == expected_bundle
    assert entry["status"] == "exhausted_failed_selection"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["p1_result_sha256"] == sha256_file(result_path)
    assert entry["p1_selection_report_sha256"] == result["selection_report_sha256"]
    assert entry["p1_training_opened_seal_sha256"] == result["training_opened_seal_sha256"]
    assert entry["p1_training_result_seal_sha256"] == result["training_result_seal_sha256"]
    assert entry["p1_selection_exact_matches"] == 190
    assert entry["p1_selection_case_count"] == 192
    assert entry["p1_selection_ambiguity_exact_match"] == 0.0
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] is None
    assert entry["public_gate_authorized_on_selection_pass"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_failed_selection_result_is_directly_bound_and_fail_closed() -> None:
    result = load(ROOT / "P1_RESULT.json")
    opened_path = REPO_ROOT / result["training_opened_seal_path"]
    completed_path = REPO_ROOT / result["training_result_seal_path"]
    completed = load(completed_path)
    assert result["status"] == "failed_selection"
    assert result["selection_metrics"] == {
        "ambiguity_exact_match": 0.0,
        "case_count": 192,
        "character_error_rate": 0.008086253369272238,
        "elapsed_ms": 4278.342,
        "exact_match": 0.9895833333333334,
        "exact_matches": 190,
        "inference_calls": 192,
        "input_tensor_stream_sha256": "ce019ce6ed2849059a4774d0edfa651959176a66e16baa12cbcc3c40b43f2644",
        "numeric_exact_match": 1.0,
        "output_tensor_stream_sha256": "cacb5dcb3951ac5956dfef3de2977c34d2898698f049615394a169aec069f75b",
        "passed": False,
        "role_accuracy": 0.9895833333333334,
        "word_exact_match": 0.9583333333333334,
    }
    assert [failure["case_id"] for failure in result["selection_failures"]] == [
        "selection-recognition-0067",
        "selection-recognition-0135",
    ]
    assert all(failure["truth_text"] == "O o l I" for failure in result["selection_failures"])
    assert all(failure["prediction"] == "OolI" for failure in result["selection_failures"])
    assert result["public_gate_evaluations"] == 0
    assert result["public_gate_archive_opened"] is False
    assert result["marker_creation_evaluated"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
    assert result["rerun_allowed"] is False
    assert sha256_file(opened_path) == result["training_opened_seal_sha256"]
    assert sha256_file(completed_path) == result["training_result_seal_sha256"]
    assert completed["opened_sha256"] == result["training_opened_seal_sha256"]
    assert completed["report_sha256"] == result["report_sha256"]


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
    result = load(ROOT / "P1_RESULT.json")
    assert result["public_gate_evaluations"] == 0
    assert result["public_gate_archive_opened"] is False
    assert not (ROOT / "artifacts/P1-run/public-report.json").exists()


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
