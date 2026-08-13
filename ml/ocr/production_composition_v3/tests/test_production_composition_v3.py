# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed tests for preregistered OCR production-composition V3."""

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.production_composition_v3.dataset import build_split, proposal_summary, split_fingerprint
from ml.ocr.production_composition_v3.protocol import (
    DETECTOR_ONNX_SHA256, DETECTOR_PUBLIC_REPORT_SHA256, REVISION, SPLITS, protocol_configuration,
)
from ml.ocr.production_composition_v3.sealed_gate import EVALUATOR_SOURCE_PATHS, SPLIT_CONFIG_PATH


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/production_composition_v3"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_freezes_fresh_v10_composition_without_approval() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    expected = json.loads(json.dumps(protocol_configuration()))
    expected["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected
    assert protocol["revision"] == REVISION
    assert protocol["models"]["detector"]["onnx_sha256"] == DETECTOR_ONNX_SHA256
    assert protocol["models"]["detector"]["public_component_report_sha256"] == DETECTOR_PUBLIC_REPORT_SHA256
    assert protocol["models"]["detector"]["threshold"] == 0.95
    assert protocol["predecessor"]["fixture_bytes_reused"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["manifest_created"] is False
    assert protocol["model_store_promoted"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert [item.scene_count for item in SPLITS] == [80, 112]
    assert not (ROOT / "VALIDATION_REPORT.json").exists()
    assert not (ROOT / "PUBLIC_GATE_REPORT.json").exists()


def test_future_public_gate_binds_transitive_sources_and_validation() -> None:
    assert SPLIT_CONFIG_PATH == Path("ml/ocr/production_composition_v3/gates/sealed-public-v1.json")
    assert Path("ml/ocr/production_composition_v2/pipeline.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/production_composition_v2/protocol.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/component_spaced_recall_detector_v10/dataset.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/component_spaced_recall_detector_v10/protocol.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/component_recall_detector_v9/dataset.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/component_ensemble_v5/protocol.py") in EVALUATOR_SOURCE_PATHS
    assert Path("ml/ocr/official_recognition_spacing_v2/spacing.py") in EVALUATOR_SOURCE_PATHS


def test_fresh_archives_are_disjoint_proposal_complete_and_unopened() -> None:
    validation = _load(ROOT / "VALIDATION_SEAL.json")
    public = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    fingerprints: set[str] = set()
    for registration in SPLITS:
        scenes = build_split(registration.split)
        summary = proposal_summary(scenes)
        fingerprint = split_fingerprint(scenes)
        seal = validation if registration.split == "validation" else public
        assert len(scenes) == registration.scene_count
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        assert summary["positive_proposal_count"] == summary["truth_region_count"]
        assert summary == {key: seal[key] for key in summary}
        assert fingerprint == seal["split_fingerprint"]
        assert seal["predecessor_fixture_bytes_reused"] is False
        assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
        assert sha256_file(REPO_ROOT / seal["private_manifest_path"]) == seal["private_manifest_sha256"]
    assert validation["validation_model_execution_count"] == 0
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert public["prior_public_sample_or_pixel_inspection_used"] is False


def test_consumed_predecessors_are_unchanged() -> None:
    assert sha256_file(REPO_ROOT / "ml/ocr/production_composition_v2/VALIDATION_REPORT.json") == (
        "7a20ae70e9c970f2d10dd80f03a41ab363424cf1a33d98327e835727b587bed1"
    )
    assert sha256_file(REPO_ROOT / "ml/ocr/component_spaced_recall_detector_v10/P3_RESULT.json") == (
        "46c75ec72e1b01c6b296b4618fe05b6528e4cf9cf559e12775b240239bef6957"
    )


def test_frozen_gate_source_bundle_will_include_all_declared_sources() -> None:
    existing = tuple(path for path in EVALUATOR_SOURCE_PATHS if (REPO_ROOT / path).exists())
    assert existing == EVALUATOR_SOURCE_PATHS
    assert len(source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)) == 64
