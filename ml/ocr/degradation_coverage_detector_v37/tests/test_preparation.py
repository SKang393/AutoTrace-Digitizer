# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Focused preparation checks for V37, with no training or authorization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.real_range_classifier_finetune_v32.dataset import (
    build_split as build_v32_split,
    split_fingerprint as v32_split_fingerprint,
)
from ml.ocr.real_range_detector_v35.dataset import build_tiles as build_v35_tiles
from ml.ocr.real_range_detector_v35.model import SourceScaleProposalNet

from ml.ocr.degradation_coverage_detector_v37.dataset import (
    _degrade,
    build_base_train_split,
    build_split,
    build_tiles,
)
from ml.ocr.degradation_coverage_detector_v37.protocol import (
    DEGRADATION_GRIDS,
    DEGRADATION_KINDS,
    EXPECTED_V32_DEV_FINGERPRINT,
    EXPECTED_V32_TRAIN_FINGERPRINT,
    TRAIN_VARIANT_SEEDS,
    V35_DIAGNOSTIC_SHA256,
    V35_RESULT_SHA256,
    V36_DIAGNOSTIC_SHA256,
    V36_RESULT_SHA256,
    protocol_configuration,
)
from ml.ocr.degradation_coverage_detector_v37.runner import SOURCE_PATHS


def test_protocol_binds_both_failures_and_no_evidence_reads() -> None:
    protocol = protocol_configuration()
    trigger = protocol["trigger_evidence"]
    assert trigger["v35_result_sha256"] == V35_RESULT_SHA256
    assert trigger["v35_diagnostic_sha256"] == V35_DIAGNOSTIC_SHA256
    assert trigger["v36_result_sha256"] == V36_RESULT_SHA256
    assert trigger["v36_diagnostic_sha256"] == V36_DIAGNOSTIC_SHA256
    assert trigger["v36_expansion_oracle_passed"] is True
    assert protocol["experiment_budget"] == 1
    assert protocol["selection_gates"]["public_or_sealed_reads"] == 0
    assert protocol["real_reads"] == 0
    assert protocol["private_or_article_images"] is False


def test_v35_contract_is_reused_and_v36_cores_are_rejected() -> None:
    protocol = protocol_configuration()
    retained = protocol["retained_v35_contract"]
    assert retained["architecture"] == "detail-skip-source-scale-segmentation-v1"
    assert retained["tile_size"] == 256
    assert retained["tile_overlap"] == 64
    assert retained["minimum_component_area"] == 8
    assert protocol["trigger_evidence"]["v35_full_box_failure"] is True
    assert protocol["trigger_evidence"]["v36_shrink_core_failure"] is True

    model = SourceScaleProposalNet().eval()
    import torch

    for count in (1, 7, 64):
        assert tuple(model(torch.zeros((count, 1, 256, 256))).shape) == (count, 1, 256, 256)


def test_v32_base_train_is_retained_byte_for_byte_and_dev_is_passthrough() -> None:
    base = build_base_train_split()
    original = build_v32_split("train")
    assert len(base) == 5 == len(original)
    for left, right in zip(base, original, strict=True):
        assert left.scene_id == right.scene_id
        assert left.truths == right.truths
        assert np.array_equal(left.raster, right.raster)
    assert v32_split_fingerprint("train") == EXPECTED_V32_TRAIN_FINGERPRINT
    assert v32_split_fingerprint("dev") == EXPECTED_V32_DEV_FINGERPRINT
    for left, right in zip(build_split("dev"), build_v32_split("dev"), strict=True):
        assert left.scene_id == right.scene_id
        assert left.truths == right.truths
        assert np.array_equal(left.raster, right.raster)
    assert EXPECTED_V32_DEV_FINGERPRINT == "67952b4575972542087281b2c14958e86518ae0e12e88d43f5c47c16252a3687"


def test_train_variants_are_fresh_deterministic_and_cover_each_grid() -> None:
    base = build_base_train_split()
    train = build_split("train")
    variants = train[len(base):]
    assert len(variants) == 5
    assert len(set(TRAIN_VARIANT_SEEDS)) == len(TRAIN_VARIANT_SEEDS)
    for base_index, source in enumerate(base):
        variant = variants[base_index]
        seed = TRAIN_VARIANT_SEEDS[base_index]
        assert variant.scene_id.endswith(f"-v37-composite-{seed}")
        assert variant.degradation_family == "v37_train_composite"
        assert variant.truths == source.truths
        assert variant.raster.shape == source.raster.shape
        first = source.raster
        second = source.raster
        for kind_index, kind in enumerate(DEGRADATION_KINDS):
            value = DEGRADATION_GRIDS[kind][base_index]
            first = _degrade(first, kind, value, seed + kind_index)
            second = _degrade(second, kind, value, seed + kind_index)
        assert np.array_equal(first, second)
        assert np.array_equal(first, variant.raster)
    assert all(not item.scene_id.endswith("-dev") for item in variants)


def test_full_box_targets_and_coordinates_match_v35() -> None:
    v35 = build_v35_tiles("train")
    v37 = build_tiles("train")
    assert len(v37) > len(v35)
    for expected, actual in zip(v35, v37[:len(v35)], strict=True):
        assert (expected.scene_id, expected.left, expected.top, expected.valid_width, expected.valid_height) == (
            actual.scene_id, actual.left, actual.top, actual.valid_width, actual.valid_height,
        )
        assert np.array_equal(expected.target, actual.target)


def test_trigger_hashes_match_and_current_source_bundle_is_bound() -> None:
    root = Path(__file__).parents[4]
    paths = {
        "ml/ocr/real_range_detector_v35/P1_RESULT.json": V35_RESULT_SHA256,
        "ml/ocr/real_range_detector_v35/diagnostics/DIAGNOSTIC.json": V35_DIAGNOSTIC_SHA256,
        "ml/ocr/shrink_region_detector_v36/P1_RESULT.json": V36_RESULT_SHA256,
        "ml/ocr/shrink_region_detector_v36/diagnostics/DIAGNOSTIC.json": V36_DIAGNOSTIC_SHA256,
    }
    for path, expected in paths.items():
        assert sha256_file(root / path) == expected
    config = json.loads((root / "ml/ocr/degradation_coverage_detector_v37/training/p1.json").read_text(encoding="utf-8"))
    actual = source_bundle_sha256(root, SOURCE_PATHS)
    assert config["expected_runner_source_bundle_sha256"] == actual


def test_preparation_has_no_optimizer_or_authorization() -> None:
    config = json.loads((Path(__file__).parents[1] / "training" / "p1.json").read_text(encoding="utf-8"))
    assert config["state"] == "prepared_not_authorized"
    assert config["experiment_budget"] == 1
    assert config["optimizer_steps"] == 0
    assert config["authorization_acquired"] is False
    assert config["public_or_sealed_reads"] == 0
    assert config["real_reads"] == 0
    assert config["seed"] == 20260935
    runner = (Path(__file__).parents[1] / "runner.py").read_text(encoding="utf-8")
    assert "acquire_training_candidate" in runner
    assert "evaluate_scenes(build_v32_split(\"dev\")" in runner


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    root = Path(__file__).parents[4]
    result = json.loads((Path(__file__).parents[1] / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["dev_metrics"]["precision"] == 0.23696682464454977
    assert result["dev_metrics"]["recall"] == 0.5813953488372093
    assert result["onnx_parity_passed"] is False
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    ledger = json.loads((root / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
