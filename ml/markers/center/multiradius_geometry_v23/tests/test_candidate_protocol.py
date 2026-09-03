# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

from ml.markers.center.multiradius_geometry_v23 import protocol
from ml.markers.center.multiradius_geometry_v23.candidate_runner import SOURCE_PATHS, evaluate_candidate
from ml.markers.center.multiradius_geometry_v23.diagnose_v23 import DEFAULT_ONNX
from ml.markers.gate_seal import source_bundle_sha256


ROOT = Path(__file__).parents[5]
def test_candidate_config_binds_v21_and_fixed_geometry_contract() -> None:
    config = json.loads((Path(__file__).parents[1] / "candidate_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "startable_after_authorization"
    assert config["v21_result_sha256"] == protocol.V21_RESULT_SHA256
    assert config["v21_diagnostic_sha256"] == protocol.V21_DIAGNOSTIC_SHA256
    assert config["v21_onnx_sha256"] == protocol.V21_ONNX_SHA256
    assert config["fixed_dev_manifest_sha256"] == protocol.V13_MANIFEST_SHA256
    assert config["ring_radii_px"] == list(protocol.RING_RADII_PX)
    assert config["confidence_threshold"] == protocol.CONFIDENCE_THRESHOLD
    assert config["optimizer_steps"] == 0
    assert config["training_performed"] is False
    assert config["sealed_candidate_budget"] == 1
    assert config["sealed_runs"] == 0
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, SOURCE_PATHS)


def test_zero_optimizer_runner_reproduces_feasible_aggregate() -> None:
    result = evaluate_candidate(DEFAULT_ONNX)
    assert result["status"] == "feasible_startable"
    assert result["inference"]["accepted_candidate_count"] == 93
    assert result["metrics"]["accepted_candidate_count"] == 93
    assert result["metrics"]["true_positives"] == 93
    assert result["metrics"]["false_positives"] == 0
    assert result["metrics"]["false_negatives"] == 3
    assert result["metrics"]["miss_count"] == 3
    assert result["metrics"]["prohibited_structure_hits"] == 0
    assert result["scope"]["optimizer_steps"] == 0
    assert result["scope"]["training_performed"] is False


def test_tracked_dev_pass_remains_unconsumed_and_unapproved() -> None:
    result = json.loads((Path(__file__).parents[1] / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "dev_passed_pending_runtime_and_real_sealed"
    assert result["dev_gate_passed"] is True
    assert result["dev_metrics"]["precision"] == 1.0
    assert result["dev_metrics"]["recall"] == 0.96875
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    assert result["production_approval"] is False
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "dev_passed_pending_runtime_and_real_sealed"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
