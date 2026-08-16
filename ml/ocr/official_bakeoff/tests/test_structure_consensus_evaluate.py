# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.ocr.official_bakeoff import structure_consensus_evaluate as gate


PROTOCOL = Path(gate.__file__).with_name("STRUCTURE_CONSENSUS_GATE_PROTOCOL.json")


def test_protocol_binds_gate_workflow_and_one_run_budget() -> None:
    protocol = gate.load_strict_json(PROTOCOL)

    assert protocol["status"] == "frozen_before_fixture_generation_and_inference"
    assert protocol["profile"] == gate.PROFILE
    assert protocol["private_data"] is False
    assert protocol["chandler_used"] is False
    assert protocol["execution_workflow_sha256"] == (
        "65de1c76288c2cd9646386afb941bf641a1a87c5abfec5d76b6cb0f7a818c992"
    )
    assert protocol["reviewed_source_sha256"]["src/GraphReader.Ocr/LocalOnnxTextRegionDetector.cs"] == (
        "c17cbd77bc646f7646f2f3f60b2120be735201b79c0b32a48318a30464b0aa38"
    )
    assert (
        "src/GraphReader.App/Integration/Workflow/ProductionOcrApprovalGate.cs"
        in protocol["reviewed_source_sha256"]
    )
    assert protocol["experiment_budget"]["official_composition_evaluations"] == 1
    assert protocol["experiment_budget"]["workflow_changes_after_inference"] == 0


def test_consumed_freeze_contract_remains_checksum_bound_without_regeneration() -> None:
    protocol = gate.load_strict_json(PROTOCOL)
    assert protocol["prior_exposed_split_forbidden"] == {
        "split_sha256": "1fc3b2e72f89cbfb0d8854ec8701368e7ae764cbd5c6fef17b7e497d06ec9f09",
        "fixture_archive_sha256": "69eeeff73f4cfd2dd6580ad9538f1a89527f8e5b320ce6a9cd7155d2bd22ea99",
    }
    assert protocol["new_split"]["fixture_bytes_checksum_bound"] is True
    assert protocol["new_split"]["masked_detector_inputs_checksum_bound"] is True
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    rectangles = [{"kind": "axis", "left": 1, "top": 1, "right": 3, "bottom": 3}]

    source = gate._source_bgr(image)
    first_masked = gate._masked_bgr(image, rectangles)
    second_masked = gate._masked_bgr(image, rectangles)

    assert gate.hash_bytes(first_masked) == gate.hash_bytes(second_masked)
    assert gate.hash_bytes(first_masked) != gate.hash_bytes(source)


def test_consumed_freeze_writer_refuses_overwrite(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    gate._write_new(evidence, b"first")

    with pytest.raises(gate.ProductionGateError, match="Refusing to overwrite frozen evidence"):
        gate._write_new(evidence, b"second")

    assert evidence.read_bytes() == b"first"


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


def test_connected_component_candidate_matches_production_vertical_glyph_grouping() -> None:
    pixels = np.full((40, 40, 3), 255, dtype=np.uint8)
    pixels[5:7, 10:16, :] = 0
    pixels[10:12, 10:16, :] = 0

    candidates = gate.connected_component_candidates(pixels.tobytes(order="C"), 40, 40)

    assert len(candidates) == 1
    assert candidates[0].component_count == 2
    assert candidates[0].bounds == gate.Box(10.0, 5.0, 16.0, 12.0)


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


def test_official_evaluation_prohibits_external_marker_result_injection() -> None:
    evaluated, counts, content, blockers = gate._missing_marker_evidence(
        [{"case_id": "case-a"}, {"case_id": "case-b"}]
    )

    assert evaluated is False
    assert counts == {"case-a": 0, "case-b": 0}
    assert content is None
    assert "external precomputed marker-result injection is prohibited" in blockers[0]

    args = gate.parse_args(
        [
            "evaluate",
            "--frozen-root", "frozen",
            "--conversion-report", "conversion.json",
            "--source-root", "source",
            "--output-root", "output",
        ]
    )
    assert not hasattr(args, "marker_evidence")


def test_every_embedded_resource_is_bounded_before_report_creation() -> None:
    with pytest.raises(gate.ProductionGateError, match="exceeds the gate resource limit"):
        gate._embedded_resource(
            "application/json",
            b"x" * (gate.MAXIMUM_RESOURCE_BYTES + 1),
            "Oversized evidence",
        )
