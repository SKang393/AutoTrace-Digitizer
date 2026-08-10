# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

from ml.ocr.official_bakeoff import production_evaluate as gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = Path(gate.__file__).with_name("PRODUCTION_GATE_PROTOCOL.json")
EVALUATOR = REPOSITORY_ROOT / "ml" / "ocr" / "production_gate.py"
FONT_ROOT = Path(r"C:\Windows\Fonts")


def test_protocol_freezes_public_scope_and_one_evaluation_budget() -> None:
    protocol = gate.load_strict_json(PROTOCOL)

    assert protocol["status"] == "frozen_before_inference"
    assert protocol["scope"] == "public_synthetic"
    assert protocol["private_data"] is False
    assert protocol["chandler_used"] is False
    assert protocol["experiment_budget"] == {
        "official_model_evaluations": 1,
        "split_regeneration_after_inference": 0,
        "threshold_changes_after_inference": 0,
    }


@pytest.mark.skipif(not (FONT_ROOT / "arial.ttf").is_file(), reason="Windows fonts unavailable")
def test_freeze_is_deterministic_and_separates_display_from_normalized_truth(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = gate.freeze_split(first, PROTOCOL, EVALUATOR, FONT_ROOT)
    second_result = gate.freeze_split(second, PROTOCOL, EVALUATOR, FONT_ROOT)

    assert first_result["sealed_split_sha256"] == second_result["sealed_split_sha256"]
    assert first_result["case_count"] == 220
    assert first_result["text_counts"] == {"validation": 100, "sealed_test": 100}
    assert first_result["exclusion_count"] == 20
    assert len(first_result["csharp_contract_blockers"]) == 1
    ambiguity = [
        case
        for case in first_result["split"]["cases"]
        if case["kind"] == "text" and case["family"] == "ambiguity"
    ]
    assert len(ambiguity) == 40
    assert all(case["display_text"] != case["truth_text"] for case in ambiguity)
    assert all(case["source_sha256"] == gate.hash_file(first / case["source_path"]) for case in ambiguity)
    assert first_result["fixture_archive_sha256"] == gate.hash_file(first / "fixtures.zip")
    gate.verify_fixture_archive((first / "fixtures.zip").read_bytes(), first_result["split"]["cases"])


@pytest.mark.skipif(not (FONT_ROOT / "arial.ttf").is_file(), reason="Windows fonts unavailable")
def test_freeze_refuses_overwrite_and_verifier_rejects_changed_source(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    result = gate.freeze_split(frozen, PROTOCOL, EVALUATOR, FONT_ROOT)
    with pytest.raises(gate.ProductionGateError, match="already exists"):
        gate.freeze_split(frozen, PROTOCOL, EVALUATOR, FONT_ROOT)

    source = frozen / result["split"]["cases"][0]["source_path"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(gate.ProductionGateError, match="source changed"):
        gate.verify_frozen_split(frozen, PROTOCOL, EVALUATOR)


def test_strict_json_rejects_duplicate_properties(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")

    with pytest.raises(gate.DuplicateJsonKeyError, match="schema"):
        gate.load_strict_json(path)


def test_character_dictionary_matches_converted_model_output() -> None:
    source = (
        REPOSITORY_ROOT
        / "ml"
        / "ocr"
        / "official_bakeoff"
        / "runs"
        / "extracted"
        / f"{gate.RECOGNITION_MODEL_ID}_infer"
        / "inference.yml"
    )
    if not source.is_file():
        pytest.skip("Ignored official recognition source unavailable")
    alphabet = gate.read_character_alphabet(source)

    assert len(alphabet) == 437
    assert all(character in alphabet for character in "0123456789OolI-.%")


def test_ctc_decode_collapses_repeats_and_blank() -> None:
    alphabet = "01O"
    values = np.zeros((1, 7, len(alphabet) + 1), dtype=np.float32)
    for time, class_index in enumerate((0, 1, 1, 0, 3, 3, 2)):
        values[0, time, class_index] = 1.0

    assert gate.decode_ctc(values, alphabet) == "0O1"


def test_role_classifier_uses_distinct_graph_locations() -> None:
    assert gate.classify_role(gate.Region(150, 10, 80, 30, 1), 384, 192) == "phase_header"
    assert gate.classify_role(gate.Region(10, 75, 40, 30, 1), 384, 192) == "y_tick"
    assert gate.classify_role(gate.Region(150, 145, 50, 30, 1), 384, 192) == "x_tick"
    assert gate.classify_role(gate.Region(300, 145, 60, 30, 1), 384, 192) == "participant"
    assert gate.classify_role(gate.Region(240, 75, 80, 30, 1), 384, 192) == "annotation"


def test_preprocessing_is_bgr_normalized_and_fixed_recognition_width() -> None:
    image = Image.new("RGB", (64, 32), (255, 0, 0))
    detector, size = gate.detector_tensor(image)
    recognition = gate.recognition_tensor(image)

    assert size == (64, 32)
    assert detector.shape == (1, 3, 32, 64)
    assert detector[0, 0, 0, 0] == pytest.approx((0 - 0.485) / 0.229)
    assert detector[0, 2, 0, 0] == pytest.approx((1 - 0.406) / 0.225)
    assert recognition.shape == (1, 3, 48, 320)
    assert recognition[0, 0, 0, 0] == pytest.approx(-1.0)
    assert recognition[0, 2, 0, 0] == pytest.approx(1.0)
    assert np.all(recognition[:, :, :, 96:] == 0)


def test_marker_evidence_is_mandatory_and_checksum_bound(tmp_path: Path) -> None:
    evaluated, counts, content, blockers = gate._load_marker_evidence(
        None, {"case-a": "d" * 64}, "a" * 64, "b" * 64, "c" * 64, "e" * 64
    )
    assert evaluated is False
    assert counts == {"case-a": 0}
    assert content is None
    assert blockers

    evidence = {
        "schema": gate.MARKER_SCHEMA,
        "profile": gate.PROFILE,
        "provider": "cpu",
        "run_id": "12345678-1234-4234-8234-123456789abc",
        "stage": "markers",
        "composition_id": "production-ocr-to-marker-composed-v1",
        "marker_model_id": "fixture-marker",
        "marker_model_sha256": "f" * 64,
        "sealed_split_sha256": "a" * 64,
        "detection_model_sha256": "b" * 64,
        "recognition_model_sha256": "c" * 64,
        "ocr_core_predictions_sha256": "e" * 64,
        "records": [
            {
                "case_id": "case-a",
                "source_sha256": "d" * 64,
                "marker_creation_count": 0,
            }
        ],
    }
    path = tmp_path / "marker.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    evaluated, counts, content, blockers = gate._load_marker_evidence(
        path,
        {"case-a": "d" * 64},
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "e" * 64,
    )
    assert evaluated is True
    assert counts == {"case-a": 0}
    assert content == path.read_bytes()
    assert blockers == []


def test_thresholds_fail_closed_on_any_mandatory_metric() -> None:
    passing = {
        "validation_exact_match": 0.90,
        "validation_cer": 0.05,
        "validation_role_accuracy": 0.90,
        "sealed_test_exact_match": 0.90,
        "sealed_test_cer": 0.05,
        "sealed_test_role_accuracy": 0.90,
        "onnx_max_abs_error": 1e-4,
        "detection_exact_rate": 1.0,
        "marker_creation_count": 0,
    }
    assert gate._threshold_blockers(passing) == []

    for key, value in {
        "validation_exact_match": 0.899,
        "validation_cer": 0.051,
        "validation_role_accuracy": 0.899,
        "sealed_test_exact_match": 0.899,
        "sealed_test_cer": 0.051,
        "sealed_test_role_accuracy": 0.899,
        "onnx_max_abs_error": 0.000101,
        "detection_exact_rate": 0.999,
        "marker_creation_count": 1,
    }.items():
        failed = dict(passing)
        failed[key] = value
        assert gate._threshold_blockers(failed), key
