# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed V32 preregistration tests."""

import json
from pathlib import Path

from ml.ocr.real_range_classifier_finetune_v32.protocol import protocol_configuration
from ml.ocr.real_range_classifier_finetune_v32.dataset import build_split


def test_protocol_is_synthetic_dev_only_and_reuses_v10_checkpoint() -> None:
    protocol = protocol_configuration()
    assert protocol["state"] == "preregistered_before_training"
    assert protocol["source_model"]["weights_reused"] is True
    assert protocol["source_model"]["train_from_scratch"] is False
    assert protocol["selection_gates"]["sealed_public_evaluations"] == 0
    assert protocol["data_scope"].startswith("fresh project-owned")


def test_train_and_dev_families_are_disjoint() -> None:
    train = build_split("train")
    dev = build_split("dev")
    assert len(train) == len(dev) == 5
    for attribute in (
        "renderer_family",
        "font_family",
        "degradation_family",
        "template_family",
        "marker_family",
    ):
        assert {getattr(item, attribute) for item in train}.isdisjoint(
            {getattr(item, attribute) for item in dev}
        )
    assert [item.scene_id for item in train] != [item.scene_id for item in dev]


def test_source_bundle_hash_is_filled_consistently() -> None:
    root = Path(__file__).parents[1]
    config = json.loads((root / "training" / "p1.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    expected = "25911ae4f7b03b362eec982cdd80e730a44fb9d45ceb700b7f6e143c008156eb"
    assert config["expected_runner_source_bundle_sha256"] == expected
    assert protocol["expected_runner_source_bundle_sha256"] == expected


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    root = Path(__file__).parents[4]
    result = json.loads((Path(__file__).parents[1] / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    assert result["dev_gate_passed"] is False
    ledger = json.loads((root / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
