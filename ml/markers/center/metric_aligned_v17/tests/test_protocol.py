# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import torch

from ml.markers.center.metric_aligned_v17 import protocol
from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet


def test_only_label_radius_changes_from_v16() -> None:
    config = json.loads((Path(__file__).parents[1] / "training" / "p1.json").read_text(encoding="utf-8"))
    assert config["label_positive_distance_px"] == 3.0
    assert config["positive_oversampling_policy"] == protocol.POSITIVE_OVERSAMPLING_POLICY
    assert config["v16_result_sha256"] == protocol.V16_RESULT_SHA256
    assert config["model_license"] == "Apache-2.0"
    assert config["expected_runner_source_bundle_sha256"] == (
        "7a558b96bfd660236ba238ecbfd9665af32c6c6dfaad156b3e8014440f566a41"
    )
    assert config["public_gate_evaluations"] == 0
    assert config["sealed_runs"] == 0


def test_reused_v16_architecture_retains_runtime_contract() -> None:
    model = ScaleClassifierNet(ModelConfig(seed=20260902)).eval()
    for count in (1, 8, 37):
        with torch.inference_mode():
            output = model(torch.zeros((count, 3, 33, 33)))
        assert tuple(output.shape) == (count, 4)
        assert float(output[:, 0].min()) >= 0.0
        assert float(output[:, 0].max()) <= 1.0
        assert float(output[:, 3].min()) >= 2.5
        assert float(output[:, 3].max()) <= 8.0


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
