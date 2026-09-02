# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

from ml.markers.center.geometry_finetune_v15 import protocol
from ml.markers.center.radial_feature_v1.model import RadialFeatureNet


def test_v15_is_train_dev_only_and_uses_runtime_contract() -> None:
    config = json.loads((Path(__file__).parents[1] / "training" / "p1.json").read_text(encoding="utf-8"))
    assert config["expected_runner_source_bundle_sha256"] == (
        "df1fa185859403d452db7aba8d9a79c9de7e5ee99a65ab737b1c1b9a0634f398"
    )
    assert config["public_gate_evaluations"] == 0
    assert config["sealed_runs"] == 0
    assert config["private_data"] is False
    assert tuple(config["selection_thresholds"]) == protocol.THRESHOLDS
    assert tuple(RadialFeatureNet.contract.output_shape) == ("candidate_count", 4)


def test_source_is_technically_compatible_with_dynamic_patch_contract() -> None:
    contract = RadialFeatureNet.contract
    assert contract.patch_size == 33
    assert contract.proposal_stride == 4
    assert contract.output_columns[-1] == "radius_pixels"
    assert protocol.RUNTIME_RADIUS_CONTRACT == {"minimum_pixels": 2.5, "maximum_pixels": 8.0}


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    root = Path(__file__).parents[5]
    result = json.loads((Path(__file__).parents[1] / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["optimizer_steps"] == 342
    assert result["dev_gate_passed"] is False
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    ledger = json.loads((root / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == protocol.REVISION)
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
