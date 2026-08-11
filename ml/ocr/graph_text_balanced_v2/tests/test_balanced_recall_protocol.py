# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed contract tests for balanced-recall detector P1."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

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
from ml.ocr.graph_text_balanced_v2.train_p1 import _loss


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

    total, weighted_binary, dice = _loss(probabilities, targets)

    assert torch.isfinite(total)
    assert torch.isfinite(weighted_binary)
    assert torch.isfinite(dice)
    assert float(total) > float(weighted_binary) > 0.0
    assert float(dice) > 0.0


def test_preregistration_hashes_and_ledger_authorize_only_p1() -> None:
    expected_hashes = {
        "PROTOCOL.json": "2ca2e0cc41cb77ef2f551e27b7006ae15c4494296a0571c1f28edb908c39c02c",
        "SELECTION_MANIFEST.json": "58ec2800591431eaaa98ccbd07d7359afe2c4b01251f197cf443fcb079c5aec9",
        "SEALED_PUBLIC_TEST_SEAL.json": "4cfb3e06a6d859661ea9df6244212aefb0b7b238f4a2434cedb525cf7bc5f0c0",
        "P1_PREREGISTRATION.json": "62880c6a75b9d11fce44c52a9af93f263affd1d744ca2b9c50db895ca4b5cf7b",
        "training/p1.json": "84aebf97f9879a1556077b6b97d4592b4d0abcc49fcaaed1fa2ba4c8fccfdfa3",
    }
    for relative_path, expected_hash in expected_hashes.items():
        assert _sha256(REVISION_ROOT / relative_path) == expected_hash

    preregistration = _json(REVISION_ROOT / "P1_PREREGISTRATION.json")
    assert preregistration["sealed_public_archive_opened"] is False
    assert preregistration["public_gate_authorized"] is False
    assert preregistration["public_gate_evaluations"] == 0
    assert preregistration["production_approval"] is False

    ledger = _json(LEDGER_PATH)
    entries = [entry for entry in ledger["revisions"] if entry["revision"] == REVISION]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    assert entry["trigger_evidence_sha256"] == "b9628f2fea238977f982e43d30210c434b0628bce1f52b555f7744746a905f7b"
    assert entry["expected_runner_source_bundle_sha256"] == "ae223519a32642f5dba246f00c8cd6a559c72717e8056367fc51afbcc9c2d85c"
    assert entry["candidate_config_sha256"]["P1"] == expected_hashes["training/p1.json"]
