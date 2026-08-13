# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.production_composition_v8.dataset import build_split, proposal_summary, split_fingerprint
from ml.ocr.production_composition_v8.prepare_split import SPLIT_SOURCES
from ml.ocr.production_composition_v8.protocol import (
    AMBIGUITY_INPUT_ALIASES, REVISION, SPLITS, ZERO_CONSENSUS_RESCUE_SCORE_MINIMUM,
    protocol_configuration,
)
from ml.ocr.production_composition_v8.sealed_gate import EVALUATOR_SOURCE_PATHS
from ml.ocr.production_composition_v8.validation_gate import GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/production_composition_v8"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_is_bounded_fail_closed_v7_repair() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    expected = json.loads(json.dumps(protocol_configuration()))
    expected["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected
    assert protocol["revision"] == REVISION
    assert protocol["predecessor"]["validation_report_sha256"] == (
        "7d1b2ace57af890fcb95476cc74e1b73f9caf1c22f4c4c9b5a178ab8b80e5dd8"
    )
    assert protocol["predecessor"]["missed_truth"] == "0"
    assert protocol["predecessor"]["missed_truth_role"] == "x_tick"
    assert protocol["predecessor"]["missed_detector_score"] == 0.8214316368103027
    assert protocol["predecessor"]["official_prediction"] == "0"
    assert protocol["predecessor"]["numeric_prediction"] == "0"
    assert protocol["predecessor"]["numeric_confidence"] == 0.9999194145202637
    assert ZERO_CONSENSUS_RESCUE_SCORE_MINIMUM == 0.82
    assert AMBIGUITY_INPUT_ALIASES == ("!", "i")
    assert GATE_CONFIG["forbidden_zero_consensus_rescue_route_count"] == 0
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_fresh_splits_are_complete_disjoint_and_unopened() -> None:
    validation = _load(ROOT / "VALIDATION_SEAL.json")
    public = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    predecessor_fingerprints = {
        _load(REPO_ROOT / f"ml/ocr/production_composition_v{version}/{name}")["split_fingerprint"]
        for version in (5, 6, 7)
        for name in ("VALIDATION_SEAL.json", "SEALED_PUBLIC_TEST_SEAL.json")
    }
    fingerprints: set[str] = set()
    for registration in SPLITS:
        assert registration.source_index_offset >= 40_000
        scenes = build_split(registration.split)
        summary = proposal_summary(scenes)
        fingerprint = split_fingerprint(scenes)
        seal = validation if registration.split == "validation" else public
        assert summary["positive_proposal_count"] == summary["truth_region_count"]
        assert summary == {key: seal[key] for key in summary}
        assert fingerprint == seal["split_fingerprint"]
        assert fingerprint not in fingerprints | predecessor_fingerprints
        fingerprints.add(fingerprint)
        assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert validation["validation_model_execution_count"] == 0
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert public["prior_public_sample_or_pixel_inspection_used"] is False


def test_freeze_and_gate_bind_all_transitive_sources() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    public_config = _load(ROOT / "gates/sealed-public-v1.json")
    assert protocol["split_generator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, SPLIT_SOURCES)
    assert Path("ml/ocr/production_composition_v6/dataset.py") in SPLIT_SOURCES
    assert Path("ml/ocr/production_composition_v6/pipeline.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/production_composition_v8/pipeline.py") in EVALUATOR_SOURCE_PATHS
    assert public_config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert public_config["production_approval"] is False
    assert public_config["release_eligible"] is False


def test_v7_failure_is_consumed() -> None:
    report_path = REPO_ROOT / "ml/ocr/production_composition_v7/VALIDATION_REPORT.json"
    report = _load(report_path)
    assert sha256_file(report_path) == "7d1b2ace57af890fcb95476cc74e1b73f9caf1c22f4c4c9b5a178ab8b80e5dd8"
    assert report["status"] == "fail"
    assert report["metrics"]["false_negatives"] == 1
    assert report["metrics"]["ambiguity_exact_match"] == 0.9
    assert report["direct_execution"]["numeric_recognizer"]["calls"] == 523


def test_v8_validation_pass_is_consumed() -> None:
    report_path = ROOT / "VALIDATION_REPORT.json"
    report = _load(report_path)
    metrics = report["metrics"]
    assert sha256_file(report_path) == "032c6badcac9fbb5a093fd10b665df5e91bca1a2b8124588b8184efa15b196a9"
    assert report["status"] == "pass"
    assert report["evaluation_count"] == 1
    assert metrics["exact_detection_scene_count"] == metrics["scene_count"] == 128
    assert metrics["true_positives"] == metrics["truth_region_count"] == 640
    assert metrics["false_positives"] == metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
    assert metrics["recognition_exact_match"] == 0.9953125
    assert metrics["character_error_rate"] == 0.000873871249635887
    assert metrics["role_accuracy"] == 1.0
    assert metrics["numeric_exact_match"] == 1.0
    assert metrics["word_exact_match"] == 0.9917355371900827
    assert metrics["ambiguity_exact_match"] == 1.0
    assert metrics["forbidden_zero_consensus_rescue_route_count"] == 0
    assert report["direct_execution"]["numeric_recognizer"]["calls"] == 547
    assert report["production_approval"] is False
    assert report["release_eligible"] is False


def test_v8_public_pass_is_consumed_but_not_production_approval() -> None:
    report_path = ROOT / "PUBLIC_GATE_REPORT.json"
    report = _load(report_path)
    metrics = report["metrics"]
    assert sha256_file(report_path) == "43384271fefedab374613141a858367b00d86a14d41b1d0994a66d602f6329b4"
    assert report["status"] == "pass"
    assert report["evaluation_count"] == 1
    assert report["validation_report_sha256"] == (
        "032c6badcac9fbb5a093fd10b665df5e91bca1a2b8124588b8184efa15b196a9"
    )
    assert metrics["exact_detection_scene_count"] == metrics["scene_count"] == 160
    assert metrics["true_positives"] == metrics["truth_region_count"] == 800
    assert metrics["false_positives"] == metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
    assert metrics["recognition_exact_match"] == 0.99375
    assert metrics["character_error_rate"] == 0.0013979496738117428
    assert metrics["role_accuracy"] == 1.0
    assert metrics["numeric_exact_match"] == 0.996875
    assert metrics["word_exact_match"] == 0.9911894273127754
    assert metrics["ambiguity_exact_match"] == 1.0
    assert metrics["forbidden_zero_consensus_rescue_route_count"] == 0
    assert report["direct_execution"]["numeric_recognizer"]["calls"] == 682
    assert report["production_approval"] is False
    assert report["release_eligible"] is False
    assert "direct C# production composition" in report["remaining_mandatory_evidence"]
