# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import torch

from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.scale_classifier_v16 import protocol


def test_runtime_contract_and_source_hash_placeholder() -> None:
    config = json.loads((Path(__file__).parents[1] / "training" / "p1.json").read_text(encoding="utf-8"))
    assert config["expected_runner_source_bundle_sha256"] == (
        "c703d5281fd77f0dd6066edfaf4a21da7e7c31c52bfaccb85ef376b10cce6707"
    )
    assert config["public_gate_evaluations"] == 0
    assert config["sealed_runs"] == 0
    assert config["private_data"] is False
    assert config["input_proposal_manifest_sha256"] == protocol.V13_MANIFEST_SHA256


def test_multiscale_model_supports_dynamic_candidate_batches() -> None:
    model = ScaleClassifierNet(ModelConfig(seed=20260902)).eval()
    for count in (1, 8, 37):
        output = model(torch.zeros((count, 3, 33, 33)))
        assert tuple(output.shape) == (count, 4)
        assert float(output[:, 0].detach().min()) >= 0.0
        assert float(output[:, 0].detach().max()) <= 1.0
        assert float(output[:, 3].detach().min()) >= 2.5
        assert float(output[:, 3].detach().max()) <= 8.0


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    root = Path(__file__).parents[5]
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
