# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.graph_text_detector_v1 import dataset, dataset_p2, dataset_p3, protocol
from ml.ocr.graph_text_detector_v1.model import GraphTextRegionNet
from ml.ocr.graph_text_detector_v1.model_p2 import GraphTextRegionNetP2
from ml.ocr.graph_text_detector_v1.train_p3 import RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS


ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_fail_closed_and_fixes_db_thresholds() -> None:
    value = protocol.protocol_configuration()
    assert value["production_approval"] is False
    assert value["release_eligible"] is False
    assert value["experiment_budget"] == 3
    assert value["postprocessing"] == {
        "algorithm": "db_postprocess_v1",
        "score_mode": "fast",
        "probability_threshold": 0.30,
        "box_confidence_threshold": 0.60,
        "unclip_ratio": 1.5,
        "minimum_side_length": 3,
        "maximum_regions": 1000,
    }


def test_training_renderer_is_deterministic_and_excludes_private_labels() -> None:
    first = dataset.render_training_patch(17)
    repeated = dataset.render_training_patch(17)
    different = dataset.render_training_patch(18)
    assert np.array_equal(first.bgr, repeated.bgr)
    assert np.array_equal(first.target, repeated.target)
    assert not np.array_equal(first.bgr, different.bgr)
    assert first.bgr.shape == (protocol.PATCH_HEIGHT, protocol.PATCH_WIDTH, 3)
    assert first.target.shape == (protocol.PATCH_HEIGHT, protocol.PATCH_WIDTH)
    assert "Chandler" not in dataset.GENERIC_TEXT
    assert "Generalization" not in dataset.GENERIC_TEXT


def test_validation_split_is_fixed_and_distinct_from_diagnostic_dimensions() -> None:
    frames = dataset.build_validation_split()
    assert len(frames) == protocol.VALIDATION_TEXT_COUNT + protocol.VALIDATION_EXCLUSION_COUNT
    assert sum(frame.kind == "text" for frame in frames) == protocol.VALIDATION_TEXT_COUNT
    assert sum(frame.kind == "exclusion" for frame in frames) == protocol.VALIDATION_EXCLUSION_COUNT
    assert all(len(frame.detector_bgr) == dataset.FRAME_WIDTH * dataset.FRAME_HEIGHT * 3 for frame in frames)
    assert (dataset.FRAME_WIDTH, dataset.FRAME_HEIGHT) != (384, 192) or all(
        frame.renderer_family != "offset-lattice-graph-diagnostic-v1" for frame in frames
    )


def test_model_returns_same_size_finite_probabilities() -> None:
    model = GraphTextRegionNet().eval()
    with torch.inference_mode():
        output = model(torch.zeros((2, 3, 128, 256), dtype=torch.float32))
    assert output.shape == (2, 1, 128, 256)
    assert torch.isfinite(output).all()
    assert torch.all((output >= 0) & (output <= 1))


def test_model_rejects_non_multiple_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible by eight"):
        GraphTextRegionNet()(torch.zeros((1, 3, 127, 256), dtype=torch.float32))


def test_frozen_metadata_remains_unapproved_when_present() -> None:
    for relative in (
        "ml/ocr/graph_text_detector_v1/PROTOCOL.json",
        "ml/ocr/graph_text_detector_v1/SELECTION_MANIFEST.json",
        "ml/ocr/graph_text_detector_v1/SEALED_PUBLIC_TEST_SEAL.json",
        "ml/ocr/graph_text_detector_v1/training/p1.json",
    ):
        path = ROOT / relative
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            assert value.get("production_approval", value.get("public_release_eligible")) is False


def test_p1_result_is_consumed_and_remains_unapproved() -> None:
    value = json.loads((ROOT / "ml/ocr/graph_text_detector_v1/P1_RESULT.json").read_text(encoding="utf-8"))
    assert value["status"] == "failed_runner_and_selection_diagnostic"
    assert value["optimizer_steps"] == 1536
    assert value["probability_violation_count"] == 1
    assert value["clipped_output_diagnostic"]["exact_fixture_count"] == 0
    assert value["clipped_output_diagnostic"]["false_region_count"] == 409
    assert value["clipped_output_diagnostic"]["exclusion_false_region_count"] == 78
    assert value["public_gate_evaluations"] == 0
    assert value["sealed_public_archive_opened"] is False
    assert value["production_approval"] is False


def test_p2_training_renderer_is_deterministic_and_production_scaled() -> None:
    first = dataset_p2.render_production_scale_patch(17)
    repeated = dataset_p2.render_production_scale_patch(17)
    exclusion = dataset_p2.render_production_scale_patch(protocol.TRAIN_SAMPLE_COUNT * 3 // 4)
    assert np.array_equal(first.bgr, repeated.bgr)
    assert np.array_equal(first.target, repeated.target)
    assert first.bgr.shape == (dataset_p2.P2_PATCH_HEIGHT, dataset_p2.P2_PATCH_WIDTH, 3)
    assert first.target.shape == (dataset_p2.P2_PATCH_HEIGHT, dataset_p2.P2_PATCH_WIDTH)
    assert np.count_nonzero(first.target) > 0
    assert np.count_nonzero(exclusion.target) == 0
    assert first.renderer_family == "production-scale-context-crops-v2"


def test_p2_model_enforces_strict_probability_range() -> None:
    model = GraphTextRegionNetP2().eval()
    with torch.inference_mode():
        output = model(torch.randn((1, 3, 192, 512), dtype=torch.float32) * 100.0)
    assert output.shape == (1, 1, 192, 512)
    assert torch.isfinite(output).all()
    assert torch.all((output >= 0) & (output <= 1))


def test_p2_result_is_consumed_and_remains_unapproved() -> None:
    value = json.loads((ROOT / "ml/ocr/graph_text_detector_v1/P2_RESULT.json").read_text(encoding="utf-8"))
    assert value["status"] == "failed_selection"
    assert value["optimizer_steps"] == 1536
    assert value["selection_metrics"]["text_exact_fixture_count"] == 0
    assert value["selection_metrics"]["exclusion_exact_fixture_count"] == 22
    assert value["predicted_box_diagnostic"]["median_height_ratio"] == 3.142156862745098
    assert value["probability_contract_passed"] is True
    assert value["onnx_parity_passed"] is True
    assert value["public_gate_evaluations"] == 0
    assert value["sealed_public_archive_opened"] is False
    assert value["production_approval"] is False


def test_p3_uses_db_shrink_targets_and_targeted_hard_negatives() -> None:
    p2_text = dataset_p2.render_production_scale_patch(17)
    p3_text = dataset_p3.render_db_shrink_patch(17)
    p3_legend = dataset_p3.render_db_shrink_patch(386)
    assert np.array_equal(p2_text.bgr, p3_text.bgr)
    assert 0 < np.count_nonzero(p3_text.target) < np.count_nonzero(p2_text.target)
    assert np.count_nonzero(p3_legend.target) == 0
    assert "compact_legend_and_arrow:validation_gamma" in p3_legend.degradation_family
    assert dataset_p3.P3_SHRINK_RATIO == 0.40


def test_p3_result_exhausts_canonical_detector_budget() -> None:
    result_path = ROOT / "ml/ocr/graph_text_detector_v1/P3_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_selection"
    assert result["optimizer_steps"] == 1536
    assert result["selection_metrics"]["exact_fixture_count"] == 63
    assert result["selection_metrics"]["text_exact_fixture_count"] == 39
    assert result["selection_metrics"]["text_missed_fixture_count"] == 29
    assert result["selection_metrics"]["false_region_count"] == 10
    assert result["selection_metrics"]["exclusion_false_region_count"] == 0
    assert result["probability_contract_passed"] is True
    assert result["onnx_parity_passed"] is True
    assert result["public_gate_evaluations"] == 0
    assert result["sealed_public_archive_opened"] is False
    assert result["production_approval"] is False
    seal_directory = ROOT / "ml/markers/training-seals/ocr-detection/graph-text-region-detector-v1/P3"
    opened_path = seal_directory / "opened.json"
    training_result_path = seal_directory / "result.json"
    training_result = json.loads(training_result_path.read_text(encoding="utf-8"))
    assert sha256_file(opened_path) == result["training_opened_seal_sha256"]
    assert sha256_file(training_result_path) == result["training_result_seal_sha256"]
    assert training_result["opened_sha256"] == result["training_opened_seal_sha256"]
    assert training_result["report_sha256"] == result["training_report_sha256"]
    assert training_result["status"] == "failed_selection"

    ledger = json.loads(
        (ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entries = [
        item
        for item in ledger["revisions"]
        if item["task"] == protocol.TASK and item["revision"] == protocol.REVISION
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "exhausted"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["authorized_candidate_id"] is None
    assert entry["execution_authorized"] is False
    assert entry["public_gate_authorized"] is False
    assert source_bundle_sha256(ROOT, P3_RUNNER_SOURCE_PATHS) == entry["expected_runner_source_bundle_sha256"]
    assert sha256_file(ROOT / entry["trigger_evidence_path"]) == entry["trigger_evidence_sha256"]
    assert sha256_file(ROOT / entry["p1_result_path"]) == entry["p1_result_sha256"]
    assert sha256_file(ROOT / entry["p2_preregistration_path"]) == entry["p2_preregistration_sha256"]
    assert sha256_file(ROOT / entry["p2_result_path"]) == entry["p2_result_sha256"]
    assert sha256_file(ROOT / entry["p3_preregistration_path"]) == entry["p3_preregistration_sha256"]
    assert sha256_file(result_path) == entry["p3_result_sha256"]
