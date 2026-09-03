# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Focused, pre-training checks for the V36 geometry and evidence contract."""

import json
from pathlib import Path

import numpy as np

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.shrink_region_detector_v36.dataset import (
    TileSample,
    expand_core_box,
    expand_geometry,
    shrink_box,
    tile_starts,
)
from ml.ocr.shrink_region_detector_v36.model import SourceScaleProposalNet
from ml.ocr.shrink_region_detector_v36.protocol import protocol_configuration
from ml.ocr.shrink_region_detector_v36.pipeline import postprocess_probability_map
from ml.ocr.shrink_region_detector_v36.train_p1 import SOURCE_PATHS
from ml.markers.gate_seal import source_bundle_sha256


def test_protocol_binds_v35_and_is_single_candidate_synthetic_only() -> None:
    protocol = protocol_configuration()
    trigger = protocol["trigger_evidence"]
    assert trigger["v35_result_sha256"] == "519b003c2155153e0ed152e26f19471511e0923cd20045f0e233ccecb63a8ed8"
    assert trigger["v35_diagnostic_sha256"] == "c6b035f8bac27f27d2b157a2af015f12f600d1287b1135ce4f0cddfbd7d21526"
    assert protocol["experiment_budget"] == 1
    assert protocol["selection_gates"]["public_or_sealed_reads"] == 0
    assert protocol["private_or_article_images"] is False


def test_model_contract_retains_v35_dynamic_batch_shape() -> None:
    import torch

    model = SourceScaleProposalNet().eval()
    for count in (1, 7, 64):
        output = model(torch.zeros((count, 1, 256, 256), dtype=torch.float32))
        assert tuple(output.shape) == (count, 1, 256, 256)


def test_db_shrink_expand_pair_is_exact_and_clipped() -> None:
    source = Box(0, 0, 31, 17)
    geometry = shrink_box(source, canvas_width=31, canvas_height=17)
    assert geometry.core.left >= source.left
    assert geometry.core.top >= source.top
    assert geometry.core.right <= source.right
    assert geometry.core.bottom <= source.bottom
    assert expand_geometry(geometry) == source
    predicted = expand_core_box(geometry.core, canvas_width=31, canvas_height=17)
    intersection = predicted.width * predicted.height
    union = source.width * source.height + predicted.width * predicted.height - intersection
    assert intersection / union >= 0.5
    clipped = expand_core_box(Box(0, 0, 3, 3), canvas_width=5, canvas_height=4)
    assert clipped == Box(0, 0, 5, 4)


def test_small_core_recovery_stays_above_the_iou_gate() -> None:
    source = Box(50, 50, 54, 54)
    core = shrink_box(source).core
    predicted = expand_core_box(core, canvas_width=100, canvas_height=100)
    x0, y0 = max(source.left, predicted.left), max(source.top, predicted.top)
    x1, y1 = min(source.right, predicted.right), min(source.bottom, predicted.bottom)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = source.width * source.height + predicted.width * predicted.height - intersection
    assert intersection / union >= 0.5


def test_adjacent_truth_boxes_have_separate_cores_and_expansion_is_deterministic() -> None:
    left = shrink_box(Box(10, 10, 30, 24)).core
    right = shrink_box(Box(30, 10, 50, 24)).core
    assert left.right <= right.left
    first = expand_core_box(left, canvas_width=100, canvas_height=100)
    second = expand_core_box(right, canvas_width=100, canvas_height=100)
    assert first == expand_core_box(left, canvas_width=100, canvas_height=100)
    assert second == expand_core_box(right, canvas_width=100, canvas_height=100)
    assert first.width <= 22 and first.height <= 16
    assert second.width <= 22 and second.height <= 16


def test_postprocess_expands_core_components_without_morphology_merge() -> None:
    probability = np.zeros((32, 64), dtype=np.float32)
    probability[12:16, 18:22] = 1.0
    probability[12:16, 42:46] = 1.0
    tiles = (TileSample("scene", 0, 0, 64, 32, np.zeros((256, 256), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8)),)
    boxes = postprocess_probability_map(probability, tiles)
    assert len(boxes) == 2
    assert boxes[0].right <= boxes[1].left or boxes[1].right <= boxes[0].left


def test_tile_edges_are_covered_without_private_data() -> None:
    assert tile_starts(256) == (0,)
    assert tile_starts(257)[-1] == 1
    config = json.loads((Path(__file__).parents[1] / "training" / "p1.json").read_text(encoding="utf-8"))
    assert config["public_or_sealed_reads"] == 0
    assert config["private_or_article_images"] is False


def test_runner_source_bundle_hash_matches_current_sources() -> None:
    config = json.loads((Path(__file__).parents[1] / "training" / "p1.json").read_text(encoding="utf-8"))
    actual = source_bundle_sha256(Path(__file__).parents[4], SOURCE_PATHS)
    assert actual == "d789fd6e82ebb1466f66756bb77bd47b2cd9a60b487940cd922e6ca060cf8cdd"
    assert config["expected_runner_source_bundle_sha256"] == actual
    protocol = json.loads((Path(__file__).parents[1] / "PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["expected_runner_source_bundle_sha256"] == actual


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    root = Path(__file__).parents[4]
    result = json.loads((Path(__file__).parents[1] / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["dev_metrics"]["precision"] == 0.15254237288135594
    assert result["dev_metrics"]["recall"] == 0.313953488372093
    assert result["onnx_parity_passed"] is True
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    ledger = json.loads((root / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
