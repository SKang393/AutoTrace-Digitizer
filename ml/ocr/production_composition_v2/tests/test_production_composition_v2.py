# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed tests for OCR production-composition V2."""

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.production_composition_v2.dataset import build_split, proposal_summary, split_fingerprint
from ml.ocr.production_composition_v2.protocol import REVISION, SPLITS, protocol_configuration
from ml.ocr.production_composition_v2.sealed_gate import EVALUATOR_SOURCE_PATHS, SPLIT_CONFIG_PATH


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/production_composition_v2"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fresh_splits_are_disjoint_and_proposal_complete() -> None:
    validation, public = _load(ROOT / "VALIDATION_SEAL.json"), _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    fingerprints: set[str] = set()
    for registration in SPLITS:
        scenes = build_split(registration.split)
        summary, fingerprint = proposal_summary(scenes), split_fingerprint(scenes)
        expected = validation if registration.split == "validation" else public
        assert len(scenes) == registration.scene_count
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        assert summary["positive_proposal_count"] == summary["truth_region_count"]
        assert summary == {key: expected[key] for key in summary}
        assert fingerprint == expected["split_fingerprint"]


def test_protocol_binds_exact_v9_spacing_p2_and_numeric_v5() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    expected = json.loads(json.dumps(protocol_configuration()))
    expected["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected
    assert protocol["predecessor"]["fixture_bytes_reused"] is False
    assert protocol["marker_creation_evaluated"] is False
    gate = _load(REPO_ROOT / SPLIT_CONFIG_PATH)
    assert gate["expected_candidate_hash_keys"] == [
        "detector_onnx_sha256", "official_recognizer_onnx_sha256",
        "numeric_recognizer_onnx_sha256", "spacing_source_sha256", "validation_report_sha256",
    ]
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)


def test_archives_are_byte_bound_and_public_truth_is_unopened() -> None:
    validation, public = _load(ROOT / "VALIDATION_SEAL.json"), _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert validation["validation_model_execution_count"] == 0
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert public["prior_public_sample_or_pixel_inspection_used"] is False
    for seal in (validation, public):
        assert seal["synthetic_only"] is True
        assert seal["private_or_article_images"] is False
        assert seal["chandler_included"] is False
        assert seal["generalization_label_included"] is False
        assert seal["predecessor_fixture_bytes_reused"] is False
        assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
        assert sha256_file(REPO_ROOT / seal["private_manifest_path"]) == seal["private_manifest_sha256"]


def test_preregistration_cannot_promote_or_open_public_gate() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    assert protocol["revision"] == REVISION
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert not (ROOT / "VALIDATION_REPORT.json").exists()
    assert not (ROOT / "PUBLIC_GATE_REPORT.json").exists()
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*composition-v2*.json"))

