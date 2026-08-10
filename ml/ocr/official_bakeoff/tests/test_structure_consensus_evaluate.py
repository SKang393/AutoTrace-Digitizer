# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.ocr.official_bakeoff import structure_consensus_evaluate as gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = Path(gate.__file__).with_name("STRUCTURE_CONSENSUS_GATE_PROTOCOL.json")
METRICS = REPOSITORY_ROOT / "ml" / "ocr" / "production_gate.py"
FONT = REPOSITORY_ROOT / "src" / "GraphReader.App" / "Assets" / "Fonts" / "NotoSans-Regular.ttf"


def test_protocol_binds_gate_workflow_and_one_run_budget() -> None:
    protocol = gate.validate_protocol(PROTOCOL, METRICS)

    assert protocol["status"] == "frozen_before_fixture_generation_and_inference"
    assert protocol["profile"] == gate.PROFILE
    assert protocol["private_data"] is False
    assert protocol["chandler_used"] is False
    assert protocol["execution_workflow_sha256"] == gate.hash_file(Path(gate.__file__))
    assert (
        "src/GraphReader.App/Integration/Workflow/ProductionOcrApprovalGate.cs"
        in protocol["reviewed_source_sha256"]
    )
    assert protocol["experiment_budget"]["official_composition_evaluations"] == 1
    assert protocol["experiment_budget"]["workflow_changes_after_inference"] == 0


def test_freeze_is_deterministic_new_and_checksum_binds_masked_pixels(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = gate.freeze_split(first, PROTOCOL, METRICS, FONT)
    second_result = gate.freeze_split(second, PROTOCOL, METRICS, FONT)

    assert first_result["case_count"] == 500
    assert first_result["sealed_split_sha256"] == second_result["sealed_split_sha256"]
    assert first_result["fixture_archive_sha256"] == second_result["fixture_archive_sha256"]
    protocol = gate.load_strict_json(PROTOCOL)
    assert first_result["sealed_split_sha256"] != protocol["prior_exposed_split_forbidden"]["split_sha256"]
    assert first_result["fixture_archive_sha256"] != protocol["prior_exposed_split_forbidden"]["fixture_archive_sha256"]
    case = first_result["split"]["cases"][0]
    with Image.open(first / case["source_path"]) as loaded:
        image = loaded.convert("RGB")
    assert gate.hash_bytes(gate._source_bgr(image)) == case["source_bgr_sha256"]
    assert gate.hash_bytes(gate._masked_bgr(image, case["mask_rectangles"])) == case["detector_image_bgr_sha256"]

    verified = gate.verify_frozen_split(first, PROTOCOL, METRICS)
    assert len(verified["split"]["cases"]) == 500
    assert len(verified["fixture_archive_bytes"]) <= gate.MAXIMUM_RESOURCE_BYTES


def test_freeze_refuses_overwrite_and_verifier_rejects_changed_fixture(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    result = gate.freeze_split(frozen, PROTOCOL, METRICS, FONT)
    with pytest.raises(gate.ProductionGateError, match="already exists"):
        gate.freeze_split(frozen, PROTOCOL, METRICS, FONT)

    first = frozen / result["split"]["cases"][0]["source_path"]
    first.write_bytes(first.read_bytes() + b"changed")
    with pytest.raises(gate.ProductionGateError, match="source changed"):
        gate.verify_frozen_split(frozen, PROTOCOL, METRICS)


def test_detector_tensor_is_bgr_float32_and_uses_production_resize_contract() -> None:
    rgb = Image.new("RGB", (320, 160), (255, 0, 0))
    bgr = gate._source_bgr(rgb)

    tensor = gate.detector_tensor(bgr, rgb.width, rgb.height)

    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, 512, 1024)
    assert tensor[0, 0, 0, 0] == pytest.approx((0.0 - 0.485) / 0.229, rel=1e-6)
    assert tensor[0, 2, 0, 0] == pytest.approx((1.0 - 0.406) / 0.225, rel=1e-6)


def test_consensus_is_one_to_one_filters_structure_and_never_substitutes_candidate() -> None:
    bounds = gate.Box(10, 10, 30, 24)
    models = (
        gate.ModelRegion("high", bounds, 0.95),
        gate.ModelRegion("low", bounds, 0.70),
    )
    candidates = (
        gate.StructureCandidate("text", bounds, 0.90, 0.10, False, 3),
        gate.StructureCandidate("graph", bounds, 0.95, 0.95, True, 1),
        gate.StructureCandidate("unmatched", gate.Box(70, 40, 80, 50), 0.90, 0.10, False, 2),
    )

    matches, final = gate.compose_consensus(models, candidates)

    assert matches == (("high", "text", 1.0),)
    assert final == (models[0],)
    assert gate.compose_consensus((), candidates) == ((), ())


def test_thresholds_fail_closed_on_duplicate_exclusion_and_marker_counts() -> None:
    passing = {
        "validation_exact_match": 0.90,
        "validation_cer": 0.05,
        "validation_role_accuracy": 0.90,
        "sealed_test_exact_match": 0.90,
        "sealed_test_cer": 0.05,
        "sealed_test_role_accuracy": 0.90,
        "onnx_max_abs_error": 1e-4,
        "detection_exact_rate": 1.0,
        "duplicate_region_count": 0,
        "exclusion_false_region_count": 0,
        "marker_creation_count": 0,
    }
    assert gate._threshold_blockers(passing) == []
    for field in (
        "duplicate_region_count",
        "exclusion_false_region_count",
        "marker_creation_count",
    ):
        failed = dict(passing)
        failed[field] = 1
        assert gate._threshold_blockers(failed), field


def test_marker_evidence_binds_exact_masked_detector_pixels(tmp_path: Path) -> None:
    case = {
        "case_id": "case-a",
        "source_sha256": "a" * 64,
        "detector_image_bgr_sha256": "b" * 64,
    }
    evidence = {
        "schema": gate.MARKER_SCHEMA,
        "profile": gate.PROFILE,
        "provider": "cpu",
        "run_id": "12345678-1234-4234-8234-123456789abc",
        "stage": "markers",
        "composition_id": gate.MARKER_COMPOSITION_ID,
        "marker_model_id": "marker-fixture",
        "marker_model_sha256": "c" * 64,
        "sealed_split_sha256": "d" * 64,
        "detection_model_sha256": "e" * 64,
        "recognition_model_sha256": "f" * 64,
        "ocr_core_predictions_sha256": "1" * 64,
        "records": [
            {
                "case_id": "case-a",
                "source_sha256": "a" * 64,
                "detector_image_bgr_sha256": "0" * 64,
                "marker_creation_count": 0,
            }
        ],
    }
    path = tmp_path / "marker.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(gate.ProductionGateError, match="changed or is incomplete"):
        gate._marker_evidence(
            path,
            [case],
            "d" * 64,
            "e" * 64,
            "f" * 64,
            "1" * 64,
        )
