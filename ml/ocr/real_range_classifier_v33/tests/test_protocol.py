# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed V33 contract tests."""

import json
from pathlib import Path

import torch

from ml.ocr.real_range_classifier_v33.model import RealRangeClassifierV33
from ml.ocr.real_range_classifier_v33.protocol import protocol_configuration


def test_protocol_records_sourcing_order_and_fixed_gate() -> None:
    protocol = protocol_configuration()
    decision = protocol["model_sourcing_decision"]
    assert decision["approved_pretrained_attempted"] is True
    assert decision["approved_fine_tune_attempted"] is True
    assert decision["project_owned_architecture_permitted"] is True
    assert protocol["selection_gates"]["sealed_public_evaluations"] == 0
    assert protocol["proposal_contract"]["proposal_score_threshold"] == 0.82
    assert protocol["proposal_contract"]["truth_match_iou_minimum"] == 0.5


def test_model_contract_supports_dynamic_proposal_counts() -> None:
    model = RealRangeClassifierV33().eval()
    for count in (1, 7, 64, 257):
        output = model(torch.zeros((count, 2, 32, 140), dtype=torch.float32))
        assert tuple(output.shape) == (count, 2)


def test_runner_source_hash_is_filled_consistently() -> None:
    root = Path(__file__).parents[1]
    config = json.loads((root / "training" / "p1.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    expected = "9f07552a844f7c5e33b1d15ec2f0a165888deb657a841b195072f434448d2f31"
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
