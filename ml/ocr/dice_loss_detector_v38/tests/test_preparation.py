# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Focused V38 preparation checks. No candidate is authorized or trained."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.degradation_coverage_detector_v37.dataset import (
    build_split as build_v37_split,
    build_tiles as build_v37_tiles,
    split_fingerprint as v37_split_fingerprint,
    to_arrays as v37_to_arrays,
)
from ml.ocr.dice_loss_detector_v38.dataset import build_split, build_tiles, to_arrays
from ml.ocr.dice_loss_detector_v38.loss import (
    batch_soft_dice_loss,
    composite_pixel_loss,
    weighted_bce_loss,
)
from ml.ocr.dice_loss_detector_v38.protocol import (
    BCE_LOSS_WEIGHT,
    DICE_EPSILON,
    DICE_LOSS_WEIGHT,
    EXPECTED_V32_DEV_FINGERPRINT,
    EXPECTED_V37_TRAIN_FINGERPRINT,
    POSITIVE_WEIGHT,
    V37_DIAGNOSTIC_SHA256,
    V37_RESULT_SHA256,
    protocol_configuration,
)
from ml.ocr.dice_loss_detector_v38.runner import SOURCE_PATHS, prepare_training


ROOT = Path(__file__).parents[4]
REVISION_ROOT = ROOT / "ml/ocr/dice_loss_detector_v38"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_binds_v37_and_keeps_all_gates_closed() -> None:
    protocol = protocol_configuration()
    trigger = protocol["trigger_evidence"]
    assert trigger["v37_result_sha256"] == V37_RESULT_SHA256
    assert trigger["v37_diagnostic_sha256"] == V37_DIAGNOSTIC_SHA256
    assert protocol["experiment_budget"] == 1
    assert protocol["selection_gates"]["raw_proposal_precision_minimum"] == 0.95
    assert protocol["selection_gates"]["raw_proposal_recall_minimum"] == 0.95
    assert protocol["public_or_sealed_reads"] == 0
    assert protocol["real_reads"] == 0
    assert protocol["private_or_article_images"] is False
    objective = protocol["pixel_objective"]
    assert objective["bce_weight"] == BCE_LOSS_WEIGHT == 1.0
    assert objective["dice_weight"] == DICE_LOSS_WEIGHT == 1.0
    assert objective["dice_epsilon"] == DICE_EPSILON == 1e-6
    assert objective["dice_reduction"] == "batch"


def test_v37_train_scenes_tiles_and_tensors_are_unchanged() -> None:
    for expected, actual in zip(build_v37_split("train"), build_split("train"), strict=True):
        assert expected.scene_id == actual.scene_id
        assert expected.truths == actual.truths
        assert np.array_equal(expected.raster, actual.raster)
    expected_tiles = build_v37_tiles("train")
    actual_tiles = build_tiles("train")
    assert len(expected_tiles) == len(actual_tiles)
    for expected, actual in zip(expected_tiles, actual_tiles, strict=True):
        assert (expected.scene_id, expected.left, expected.top, expected.valid_width, expected.valid_height) == (
            actual.scene_id, actual.left, actual.top, actual.valid_width, actual.valid_height,
        )
        assert np.array_equal(expected.image, actual.image)
        assert np.array_equal(expected.target, actual.target)
    expected_images, expected_targets = v37_to_arrays(expected_tiles)
    actual_images, actual_targets = to_arrays(actual_tiles)
    assert np.array_equal(expected_images, actual_images)
    assert np.array_equal(expected_targets, actual_targets)
    assert v37_split_fingerprint("train") == EXPECTED_V37_TRAIN_FINGERPRINT


def test_v32_dev_fingerprint_is_fixed_passthrough() -> None:
    assert v37_split_fingerprint("dev") == EXPECTED_V32_DEV_FINGERPRINT
    for expected, actual in zip(build_v37_split("dev"), build_split("dev"), strict=True):
        assert expected.scene_id == actual.scene_id
        assert expected.truths == actual.truths
        assert np.array_equal(expected.raster, actual.raster)


def test_bce_term_matches_v37_exactly_and_composite_weights_are_equal() -> None:
    logits = torch.tensor([[[[-4.0, -0.25], [0.75, 2.5]]], [[[1.0, -1.5], [0.0, 3.0]]]])
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]], [[[1.0, 0.0], [0.0, 1.0]]]])
    expected = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POSITIVE_WEIGHT]))(logits, target)
    total, bce, dice = composite_pixel_loss(logits, target)
    assert torch.equal(weighted_bce_loss(logits, target), expected)
    assert torch.equal(bce, expected)
    assert torch.equal(total, bce + dice)


def test_batch_soft_dice_scalar_is_deterministic() -> None:
    logits = torch.tensor([[[[-2.0, 0.0], [2.0, 4.0]]], [[[1.0, -1.0], [0.5, -0.5]]]])
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]], [[[1.0, 0.0], [0.0, 1.0]]]])
    first = batch_soft_dice_loss(logits, target)
    second = batch_soft_dice_loss(logits, target)
    assert torch.equal(first, second)
    probabilities = torch.sigmoid(logits)
    expected = 1.0 - ((2.0 * (probabilities * target).sum() + DICE_EPSILON) / (probabilities.sum() + target.sum() + DICE_EPSILON))
    assert torch.equal(first, expected)
    assert torch.isfinite(first)


def test_preparation_cannot_step_an_optimizer(monkeypatch) -> None:
    def unexpected_step(*_args, **_kwargs):
        raise AssertionError("optimizer step occurred during preparation")

    monkeypatch.setattr(torch.optim.AdamW, "step", unexpected_step)
    values, targets = prepare_training()
    assert values.shape == targets.shape
    config = _json(REVISION_ROOT / "training/p1.json")
    assert config["optimizer_steps"] == 0
    assert config["authorization_acquired"] is False
    assert config["public_or_sealed_reads"] == 0
    assert config["real_reads"] == 0


def test_v37_hashes_and_current_source_bundle_are_bound() -> None:
    assert sha256_file(ROOT / "ml/ocr/degradation_coverage_detector_v37/P1_RESULT.json") == V37_RESULT_SHA256
    assert sha256_file(ROOT / "ml/ocr/degradation_coverage_detector_v37/diagnostics/DIAGNOSTIC.json") == V37_DIAGNOSTIC_SHA256
    config = _json(REVISION_ROOT / "training/p1.json")
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, SOURCE_PATHS)


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    result = _json(REVISION_ROOT / "P1_RESULT.json")
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["dev_metrics"]["precision"] == 0.22169811320754718
    assert result["dev_metrics"]["recall"] == 0.5465116279069767
    assert result["onnx_parity_passed"] is False
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    ledger = _json(ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
