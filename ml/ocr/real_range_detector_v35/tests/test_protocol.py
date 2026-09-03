# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed V35 contract tests."""

import json
from pathlib import Path

import torch

from ml.ocr.real_range_detector_v35.model import SourceScaleProposalNet
from ml.ocr.real_range_detector_v35.dataset import TileSample
from ml.ocr.real_range_detector_v35.pipeline import tile_to_source_box
from ml.ocr.real_range_detector_v35.protocol import protocol_configuration
import numpy as np


def test_protocol_binds_v34_ceiling_and_all_prior_routes() -> None:
    protocol = protocol_configuration()
    trigger = protocol["trigger_evidence"]
    assert trigger["v34_raw_proposal_recall"] == 0.7093023256
    assert trigger["v34_deterministic_expansion_failed"] is True
    assert trigger["classifier_only_insufficient"] is True
    decision = protocol["model_sourcing_decision"]
    assert decision["learned_proposal_detector_permitted"] is True
    assert protocol["selection_gates"]["public_or_sealed_reads"] == 0


def test_model_contract_is_source_scale_and_dynamic_batch_safe() -> None:
    model = SourceScaleProposalNet().eval()
    for count in (1, 7, 64):
        output = model(torch.zeros((count, 1, 256, 256), dtype=torch.float32))
        assert tuple(output.shape) == (count, 1, 256, 256)


def test_tile_coordinate_mapping_is_reversible_at_edges() -> None:
    tile = TileSample("scene", 100, 200, 40, 50, np.zeros((256, 256), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8))
    mapped = tile_to_source_box(tile, 35, 45, 12, 12)
    assert (mapped.left, mapped.top, mapped.right, mapped.bottom) == (135, 245, 140, 250)


def test_runner_source_hash_is_filled_consistently() -> None:
    root = Path(__file__).parents[1]
    config = json.loads((root / "training" / "p1.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    expected = "c17ea14e07c9f4a198ce2cb60dedfd33e08b10a6657efae093dcc9f6bbb800f0"
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
