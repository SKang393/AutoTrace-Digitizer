# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed contract tests for the balanced-recall detector budget."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.markers.gate_seal import source_bundle_sha256
from ml.ocr.graph_text_balanced_v2.dataset import (
    GENERIC_TEXT,
    build_validation_split,
    render_training_patch,
    split_fingerprint,
)
from ml.ocr.graph_text_balanced_v2.model import BalancedRecallTextRegionNet
from ml.ocr.graph_text_balanced_v2.protocol import (
    BOX_CONFIDENCE_THRESHOLD,
    DB_SHRINK_RATIO,
    DICE_LOSS_WEIGHT,
    POSITIVE_BCE_WEIGHT,
    PROBABILITY_THRESHOLD,
    REVISION,
    VALIDATION_EXCLUSION_COUNT,
    VALIDATION_TEXT_COUNT,
    protocol_configuration,
)
from ml.ocr.graph_text_balanced_v2.train_p1 import _loss as p1_loss
from ml.ocr.graph_text_balanced_v2.train_p2 import (
    HARD_NEGATIVE_LOSS_WEIGHT,
    HARD_NEGATIVE_TOPK_FRACTION,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _loss as p2_loss,
)
from ml.ocr.graph_text_balanced_v2.train_p3 import (
    HARD_NEGATIVE_SCOPE,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
    _loss as p3_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
REVISION_ROOT = REPO_ROOT / "ml/ocr/graph_text_balanced_v2"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_protocol_freezes_balanced_recall_defect_and_fail_closed_gates() -> None:
    protocol = protocol_configuration()

    assert protocol["revision"] == REVISION == "graph-text-balanced-recall-v2"
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["experiment_budget"] == 3
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert protocol["trigger"]["prior_text_missed_fixture_count"] == 29
    assert protocol["trigger"]["prior_exclusion_false_region_count"] == 0
    assert protocol["training"]["positive_bce_weight"] == POSITIVE_BCE_WEIGHT == 8.0
    assert protocol["training"]["dice_loss_weight"] == DICE_LOSS_WEIGHT == 2.0
    assert protocol["training"]["db_shrink_ratio"] == DB_SHRINK_RATIO == 0.40
    assert protocol["postprocessing"]["probability_threshold"] == PROBABILITY_THRESHOLD == 0.30
    assert protocol["postprocessing"]["box_confidence_threshold"] == BOX_CONFIDENCE_THRESHOLD == 0.60
    assert protocol["selection_gates"]["exact_region_count_every_fixture"] is True
    assert protocol["selection_gates"]["false_region_count"] == 0
    assert protocol["selection_gates"]["exclusion_false_region_count"] == 0
    assert protocol["fixed_experiment_policy"]["threshold_sweeps"] == 0
    assert protocol["fixed_experiment_policy"]["sealed_public_evaluations_before_selection"] == 0


def test_training_renderer_is_deterministic_and_contains_no_private_case_labels() -> None:
    first = render_training_patch(0)
    repeated = render_training_patch(0)
    exclusion = render_training_patch(639)

    assert first.kind == "text"
    assert np.array_equal(first.bgr, repeated.bgr)
    assert np.array_equal(first.target, repeated.target)
    assert np.count_nonzero(first.target) > 0
    assert exclusion.kind == "exclusion"
    assert np.count_nonzero(exclusion.target) == 0
    assert all("chandler" not in label.lower() for label in GENERIC_TEXT)
    assert all("generalization" not in label.lower() for label in GENERIC_TEXT)


def test_validation_split_is_frozen_and_uses_distinct_selection_families() -> None:
    frames = build_validation_split()

    assert len(frames) == VALIDATION_TEXT_COUNT + VALIDATION_EXCLUSION_COUNT == 96
    assert sum(frame.kind == "text" for frame in frames) == VALIDATION_TEXT_COUNT
    assert sum(frame.kind == "exclusion" for frame in frames) == VALIDATION_EXCLUSION_COUNT
    assert {frame.renderer_family for frame in frames} == {"mirrored-margin-crossbar-frame-v2"}
    assert all(frame.degradation_family.startswith("validation-quantized-haze-raster-v2:") for frame in frames)
    assert split_fingerprint(frames) == "f71d59016d380e8fc83037951ff1c2e4bb1fe238c7183dd82d98bd9a4fe442a6"


def test_model_preserves_shape_probability_contract_and_rejects_bad_dimensions() -> None:
    model = BalancedRecallTextRegionNet().eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 192, 512), dtype=torch.float32))

    assert tuple(output.shape) == (1, 1, 192, 512)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    with pytest.raises(ValueError, match=r"\[batch,3,H,W\]"):
        model(torch.zeros((1, 1, 192, 512), dtype=torch.float32))
    with pytest.raises(ValueError, match="divisible by eight"):
        model(torch.zeros((1, 3, 191, 512), dtype=torch.float32))


def test_weighted_loss_is_finite_for_positive_and_empty_targets() -> None:
    probabilities = torch.full((2, 1, 16, 16), 0.25, dtype=torch.float32)
    targets = torch.zeros_like(probabilities)
    targets[0, :, 4:12, 5:11] = 1.0

    total, weighted_binary, dice = p1_loss(probabilities, targets)

    assert torch.isfinite(total)
    assert torch.isfinite(weighted_binary)
    assert torch.isfinite(dice)
    assert float(total) > float(weighted_binary) > 0.0
    assert float(dice) > 0.0


def test_p2_adds_only_frozen_hard_negative_loss_and_binds_runner_source() -> None:
    probabilities = torch.full((2, 1, 16, 16), 0.25, dtype=torch.float32)
    targets = torch.zeros_like(probabilities)
    targets[0, :, 4:12, 5:11] = 1.0

    p1_total, p1_binary, p1_dice = p1_loss(probabilities, targets)
    p2_total, p2_binary, p2_dice, hard_negative = p2_loss(probabilities, targets)

    assert HARD_NEGATIVE_TOPK_FRACTION == 0.02
    assert HARD_NEGATIVE_LOSS_WEIGHT == 2.0
    assert torch.equal(p1_binary, p2_binary)
    assert torch.equal(p1_dice, p2_dice)
    assert torch.isfinite(hard_negative)
    assert float(hard_negative) > 0.0
    assert torch.allclose(
        p2_total,
        p1_total + (HARD_NEGATIVE_LOSS_WEIGHT * hard_negative),
    )
    config = _json(REVISION_ROOT / "training/p2.json")
    assert source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS) == config[
        "expected_runner_source_bundle_sha256"
    ]


def test_p3_limits_hard_negative_loss_to_empty_target_exclusions() -> None:
    probabilities = torch.full((2, 1, 16, 16), 0.25, dtype=torch.float32)
    targets = torch.zeros_like(probabilities)
    targets[0, :, 4:12, 5:11] = 1.0

    _, _, _, p3_hard_negative = p3_loss(probabilities, targets)
    _, _, _, exclusion_only_hard_negative = p2_loss(probabilities[1:], targets[1:])
    p1_text_total, _, _ = p1_loss(probabilities[:1], targets[:1])
    p3_text_total, _, _, text_only_hard_negative = p3_loss(probabilities[:1], targets[:1])

    assert HARD_NEGATIVE_SCOPE == "empty_target_exclusion_patches_only"
    assert torch.equal(p3_hard_negative, exclusion_only_hard_negative)
    assert float(text_only_hard_negative) == 0.0
    assert torch.equal(p3_text_total, p1_text_total)
    config = _json(REVISION_ROOT / "training/p3.json")
    p2_config = _json(REVISION_ROOT / "training/p2.json")
    for field in (
        "architecture",
        "batch_size",
        "db_shrink_ratio",
        "dice_loss_weight",
        "epochs",
        "hard_negative_loss_weight",
        "hard_negative_topk_fraction",
        "learning_rate",
        "onnx_parity_tolerance",
        "positive_bce_weight",
        "seed",
        "selection_manifest_sha256",
        "training_split_fingerprint",
        "weight_decay",
    ):
        assert config[field] == p2_config[field]
    assert source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS) == config[
        "expected_runner_source_bundle_sha256"
    ]


def test_result_hashes_and_ledger_exhaust_failed_p3() -> None:
    expected_hashes = {
        "PROTOCOL.json": "2ca2e0cc41cb77ef2f551e27b7006ae15c4494296a0571c1f28edb908c39c02c",
        "SELECTION_MANIFEST.json": "58ec2800591431eaaa98ccbd07d7359afe2c4b01251f197cf443fcb079c5aec9",
        "SEALED_PUBLIC_TEST_SEAL.json": "4cfb3e06a6d859661ea9df6244212aefb0b7b238f4a2434cedb525cf7bc5f0c0",
        "P1_PREREGISTRATION.json": "62880c6a75b9d11fce44c52a9af93f263affd1d744ca2b9c50db895ca4b5cf7b",
        "P1_RESULT.json": "8bca5784b358d958f4936db7886d3a68c7ad53c964924efc9dcb18c30703884f",
        "P2_PREREGISTRATION.json": "846f0840ee17d463f97aeb76f4e1cf81f8991957d677d8f280bfcfdafe9794fd",
        "P2_RESULT.json": "d45cdb04914014081da272964b7e74fce7514ab35baf06982df63ed1fe97d58f",
        "P3_PREREGISTRATION.json": "e8ac1d307b7e9e807150d4d00bfea31bbc58ca569c93e7e7915e011df51a5db4",
        "P3_RESULT.json": "2d21585f703384db014ae9d2c45f15b3ff3fb1ef2cdc2aaf02918e560001e288",
        "training/p1.json": "84aebf97f9879a1556077b6b97d4592b4d0abcc49fcaaed1fa2ba4c8fccfdfa3",
        "training/p2.json": "17c5669b4276c1986209331b77c6c4bb1f6893dea8456b88794fc7c4f1606bbe",
        "training/p3.json": "f663bd2fb9cbf3e0ea1f4a4119213f4885b7d6637020022cbe3eb0ccbe0746c7",
    }
    for relative_path, expected_hash in expected_hashes.items():
        assert _sha256(REVISION_ROOT / relative_path) == expected_hash

    preregistration = _json(REVISION_ROOT / "P1_PREREGISTRATION.json")
    assert preregistration["sealed_public_archive_opened"] is False
    assert preregistration["public_gate_authorized"] is False
    assert preregistration["public_gate_evaluations"] == 0
    assert preregistration["production_approval"] is False
    p1_result = _json(REVISION_ROOT / "P1_RESULT.json")
    assert p1_result["status"] == "failed_selection"
    assert p1_result["selection_metrics"]["exact_fixture_count"] == 52
    assert p1_result["selection_metrics"]["false_region_count"] == 38
    assert p1_result["selection_metrics"]["exclusion_false_region_count"] == 9
    assert p1_result["sealed_public_archive_opened"] is False
    p2_preregistration = _json(REVISION_ROOT / "P2_PREREGISTRATION.json")
    assert p2_preregistration["p1_consumed"] is True
    assert p2_preregistration["sealed_public_archive_opened"] is False
    assert p2_preregistration["public_gate_authorized"] is False
    p2_result = _json(REVISION_ROOT / "P2_RESULT.json")
    assert p2_result["status"] == "failed_selection"
    assert p2_result["selection_metrics"]["exact_fixture_count"] == 56
    assert p2_result["selection_metrics"]["false_region_count"] == 41
    assert p2_result["selection_metrics"]["text_missed_fixture_count"] == 20
    assert p2_result["sealed_public_archive_opened"] is False
    p3_preregistration = _json(REVISION_ROOT / "P3_PREREGISTRATION.json")
    assert p3_preregistration["p1_consumed"] is True
    assert p3_preregistration["p2_consumed"] is True
    assert p3_preregistration["sealed_public_archive_opened"] is False
    assert p3_preregistration["public_gate_authorized"] is False
    p3_result = _json(REVISION_ROOT / "P3_RESULT.json")
    assert p3_result["status"] == "failed_selection"
    assert p3_result["selection_metrics"]["exact_fixture_count"] == 30
    assert p3_result["selection_metrics"]["false_region_count"] == 82
    assert p3_result["selection_metrics"]["exclusion_false_region_count"] == 20
    assert p3_result["sealed_public_archive_opened"] is False

    ledger = _json(LEDGER_PATH)
    entries = [entry for entry in ledger["revisions"] if entry["revision"] == REVISION]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "exhausted_failed_selection"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    assert entry["trigger_evidence_sha256"] == "b9628f2fea238977f982e43d30210c434b0628bce1f52b555f7744746a905f7b"
    assert entry["p1_expected_runner_source_bundle_sha256"] == "ae223519a32642f5dba246f00c8cd6a559c72717e8056367fc51afbcc9c2d85c"
    assert entry["p2_expected_runner_source_bundle_sha256"] == "3c91e9ac38f682c6713c7927ab08d6d8410decc3a85e2f10d17dcf675e716c79"
    assert entry["p3_expected_runner_source_bundle_sha256"] == "6e4a4c34bbbda4890770c7e8008a56f4320f044aaea8e50ee9f75bfb6a9e536d"
    assert entry["candidate_config_sha256"]["P1"] == expected_hashes["training/p1.json"]
    assert entry["candidate_config_sha256"]["P2"] == expected_hashes["training/p2.json"]
    assert entry["candidate_config_sha256"]["P3"] == expected_hashes["training/p3.json"]
    assert entry["p1_result_sha256"] == expected_hashes["P1_RESULT.json"]
    assert entry["p2_preregistration_sha256"] == expected_hashes["P2_PREREGISTRATION.json"]
    assert entry["p2_result_sha256"] == expected_hashes["P2_RESULT.json"]
    assert entry["p3_preregistration_sha256"] == expected_hashes["P3_PREREGISTRATION.json"]
    assert entry["p3_result_sha256"] == expected_hashes["P3_RESULT.json"]
