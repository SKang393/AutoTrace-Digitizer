# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.production_composition_v6 import pipeline as v6_pipeline
from ml.ocr.production_composition_v7 import pipeline as v7_pipeline
from ml.ocr.production_composition_v7.dataset import build_split, proposal_summary, split_fingerprint
from ml.ocr.production_composition_v7.prepare_split import SPLIT_SOURCES
from ml.ocr.production_composition_v7.protocol import REVISION, SPLITS, protocol_configuration
from ml.ocr.production_composition_v7.sealed_gate import EVALUATOR_SOURCE_PATHS
from ml.ocr.production_composition_v7.validation_gate import GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/production_composition_v7"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_is_an_isolated_fail_closed_invariant_repair() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    expected = json.loads(json.dumps(protocol_configuration()))
    expected["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected
    assert protocol["revision"] == REVISION
    assert protocol["predecessor"]["validation_report_sha256"] == (
        "b50b8fc1f20da8e589a7436e4d8b41143f85f12a45445fc68ee38483175aa12f"
    )
    assert protocol["predecessor"]["all_scientific_metrics_passed"] is True
    assert protocol["predecessor"]["numeric_onnx_calls"] == 512
    assert protocol["predecessor"]["accepted_truth_regions"] == 600
    assert protocol["composition_source"]["scientific_behavior_changed"] is False
    assert protocol["gates"]["numeric_onnx_direct_execution_minimum"] == 1
    assert GATE_CONFIG["numeric_onnx_direct_execution_minimum"] == 1
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_v7_executes_the_exact_v6_scientific_pipeline() -> None:
    assert v7_pipeline.evaluate_scenes is v6_pipeline.evaluate_scenes
    assert v7_pipeline.DirectRunner is v6_pipeline.DirectRunner
    assert Path("ml/ocr/production_composition_v6/pipeline.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/production_composition_v6/dataset.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/production_composition_v6/protocol.py") in EVALUATOR_SOURCE_PATHS


def test_fresh_splits_are_complete_disjoint_and_unopened() -> None:
    validation = _load(ROOT / "VALIDATION_SEAL.json")
    public = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    predecessor_fingerprints = {
        _load(REPO_ROOT / f"ml/ocr/production_composition_v{version}/{name}")["split_fingerprint"]
        for version in (5, 6)
        for name in ("VALIDATION_SEAL.json", "SEALED_PUBLIC_TEST_SEAL.json")
    }
    fingerprints: set[str] = set()
    for registration in SPLITS:
        assert registration.source_index_offset >= 20_000
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


def test_freeze_and_gate_bind_transitive_sources() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    public_config = _load(ROOT / "gates/sealed-public-v1.json")
    assert protocol["split_generator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, SPLIT_SOURCES)
    assert Path("ml/ocr/production_composition_v6/dataset.py") in SPLIT_SOURCES
    assert Path("ml/ocr/production_composition_v6/protocol.py") in SPLIT_SOURCES
    assert public_config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert public_config["production_approval"] is False
    assert public_config["release_eligible"] is False


def test_consumed_v6_failure_remains_exact() -> None:
    report_path = REPO_ROOT / "ml/ocr/production_composition_v6/VALIDATION_REPORT.json"
    report = _load(report_path)
    metrics = report["metrics"]
    assert sha256_file(report_path) == "b50b8fc1f20da8e589a7436e4d8b41143f85f12a45445fc68ee38483175aa12f"
    assert report["status"] == "fail"
    assert metrics["exact_detection_scene_count"] == metrics["scene_count"] == 120
    assert metrics["true_positives"] == metrics["truth_region_count"] == 600
    assert metrics["false_positives"] == metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
    assert metrics["recognition_exact_match"] >= 0.90
    assert metrics["character_error_rate"] <= 0.05
    assert metrics["role_accuracy"] >= 0.90
    assert metrics["numeric_exact_match"] >= 0.90
    assert metrics["word_exact_match"] >= 0.90
    assert metrics["ambiguity_exact_match"] >= 0.90
    assert report["direct_execution"]["numeric_recognizer"]["calls"] == 512


def test_v7_validation_failure_is_consumed_and_public_remains_unopened() -> None:
    report_path = ROOT / "VALIDATION_REPORT.json"
    report = _load(report_path)
    metrics = report["metrics"]
    assert sha256_file(report_path) == "7d1b2ace57af890fcb95476cc74e1b73f9caf1c22f4c4c9b5a178ab8b80e5dd8"
    assert report["status"] == "fail"
    assert report["evaluation_count"] == 1
    assert metrics["scene_count"] == 124
    assert metrics["exact_detection_scene_count"] == 123
    assert metrics["true_positives"] == 619
    assert metrics["truth_region_count"] == 620
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 1
    assert metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
    assert metrics["recognition_exact_match"] == 0.9903225806451613
    assert metrics["character_error_rate"] == 0.0015028554253080854
    assert metrics["role_accuracy"] == 0.9983870967741936
    assert metrics["numeric_exact_match"] == 1.0
    assert metrics["word_exact_match"] == 0.9914772727272727
    assert metrics["ambiguity_exact_match"] == 0.9
    assert report["direct_execution"]["numeric_recognizer"]["calls"] == 523
    failure = next(case for case in metrics["cases"] if case["false_negatives"])
    assert failure["scene_id"] == "ocr-production-composition-v7-validation-00079"
    assert failure["accepted_region_count"] == 4
    assert not (ROOT / "PUBLIC_GATE_REPORT.json").exists()
