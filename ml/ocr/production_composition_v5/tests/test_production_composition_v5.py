# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.production_composition_v5.dataset import build_split, proposal_summary, split_fingerprint
from ml.ocr.production_composition_v5.protocol import (
    AMBIGUITY_PUBLIC_REPORT_SHA256,
    AMBIGUITY_RECOGNIZER_ONNX_SHA256,
    OFFICIAL_RESCUE_SCORE_MINIMUM,
    REVISION,
    SPLITS,
    protocol_configuration,
)
from ml.ocr.production_composition_v5.sealed_gate import EVALUATOR_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/production_composition_v5"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_is_fresh_fail_closed_four_model_composition() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    expected = json.loads(json.dumps(protocol_configuration()))
    expected["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected
    assert protocol["revision"] == REVISION
    assert protocol["models"]["ambiguity_specialist"]["onnx_sha256"] == AMBIGUITY_RECOGNIZER_ONNX_SHA256
    assert protocol["models"]["ambiguity_specialist"]["public_report_sha256"] == AMBIGUITY_PUBLIC_REPORT_SHA256
    assert protocol["predecessor"]["fixture_bytes_reused"] is False
    assert protocol["predecessor"]["public_archive_opened"] is False
    assert OFFICIAL_RESCUE_SCORE_MINIMUM == 0.90
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_fresh_splits_are_disjoint_complete_and_unopened() -> None:
    validation = _load(ROOT / "VALIDATION_SEAL.json")
    public = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    fingerprints = set()
    predecessor_fingerprints = {
        _load(REPO_ROOT / "ml/ocr/production_composition_v4/VALIDATION_SEAL.json")["split_fingerprint"],
        _load(REPO_ROOT / "ml/ocr/production_composition_v4/SEALED_PUBLIC_TEST_SEAL.json")["split_fingerprint"],
    }
    for registration in SPLITS:
        scenes = build_split(registration.split)
        summary = proposal_summary(scenes)
        fingerprint = split_fingerprint(scenes)
        seal = validation if registration.split == "validation" else public
        assert summary["positive_proposal_count"] == summary["truth_region_count"]
        assert summary == {key: seal[key] for key in summary}
        assert fingerprint == seal["split_fingerprint"]
        assert fingerprint not in fingerprints
        assert fingerprint not in predecessor_fingerprints
        fingerprints.add(fingerprint)
        assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert validation["validation_model_execution_count"] == 0
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert public["prior_public_sample_or_pixel_inspection_used"] is False


def test_gate_binds_crop_adapter_and_all_exact_payload_hashes() -> None:
    config = _load(ROOT / "gates/sealed-public-v1.json")
    assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    assert "ambiguity_recognizer_onnx_sha256" in config["expected_candidate_hash_keys"]
    assert Path("ml/ocr/ambiguity_source_group_classifier_v3/crop.py") in EVALUATOR_SOURCE_PATHS
    assert config["production_approval"] is False
    assert config["release_eligible"] is False


def test_consumed_predecessor_and_component_evidence_are_unchanged() -> None:
    assert sha256_file(REPO_ROOT / "ml/ocr/production_composition_v4/VALIDATION_REPORT.json") == (
        "075eb4cfee77591b8c2f16e3752a85364db261425ca477d99b26d940733a978e"
    )
    assert sha256_file(
        REPO_ROOT / "ml/ocr/ambiguity_source_group_classifier_v3/artifacts/P2-run/graph-ambiguity-source-group-v3-p2.onnx"
    ) == AMBIGUITY_RECOGNIZER_ONNX_SHA256
    assert sha256_file(
        REPO_ROOT / "ml/ocr/ambiguity_source_group_classifier_v3/artifacts/public-gate-v1/report.json"
    ) == AMBIGUITY_PUBLIC_REPORT_SHA256
    assert not (ROOT / "PUBLIC_GATE_REPORT.json").exists()


def test_failed_validation_is_consumed_and_public_remains_unopened() -> None:
    report = _load(ROOT / "VALIDATION_REPORT.json")
    metrics = report["metrics"]
    assert sha256_file(ROOT / "VALIDATION_REPORT.json") == "c3894907e9354b841baac5ae9d98997b2f486ff3c87c9d831ead9c827f339d84"
    assert report["status"] == "fail"
    assert report["evaluation_count"] == 1
    assert metrics["true_positives"] == 559
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["duplicate_region_count"] == 0
    assert metrics["official_tick_rescue_count"] == 7
    assert metrics["ambiguity_exact_match"] == 0.8421052631578947
    assert metrics["forbidden_official_rescue_route_count"] == 0
    public = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert not (ROOT / "PUBLIC_GATE_REPORT.json").exists()
