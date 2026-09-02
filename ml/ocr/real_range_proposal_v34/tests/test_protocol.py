# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed V34 protocol tests."""

import json
from pathlib import Path

from ml.ocr.real_range_proposal_v34.protocol import protocol_configuration


def test_protocol_binds_proposal_ceiling_and_classifier_failures() -> None:
    protocol = protocol_configuration()
    trigger = protocol["trigger_evidence"]
    assert trigger["v32_raw_proposal_maximum_match_true_positives"] == 61
    assert trigger["v32_raw_proposal_truth_regions"] == 86
    assert trigger["v32_raw_proposal_recall"] == 0.7093023256
    assert trigger["classifier_only_cannot_recover_missing_raw_proposals"] is True
    assert protocol["selection_gates"]["public_or_sealed_reads"] == 0


def test_protocol_is_deterministic_repair_before_learned_detector() -> None:
    protocol = protocol_configuration()
    assert protocol["model_sourcing"]["learned_detector_needed"] is False
    assert protocol["proposal_contract"]["repair_algorithm"] == "percentile-contrast-union-expand-group-v1"
    assert protocol["proposal_contract"]["expansion_margin_pixels"] == 1
    assert protocol["selection_gates"]["raw_proposal_precision_minimum"] == 0.95
    assert protocol["selection_gates"]["raw_proposal_recall_minimum"] == 0.95


def test_diagnostic_source_hash_is_filled_consistently() -> None:
    root = Path(__file__).parents[1]
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    config = json.loads((root / "evaluation" / "p1.json").read_text(encoding="utf-8"))
    expected = "2b6b268a0a5bf58d78b3b1c7a7daf60e8ed659c668c2865f06f766781a01cec5"
    assert protocol["expected_runner_source_bundle_sha256"] == expected
    assert config["expected_runner_source_bundle_sha256"] == expected


def test_tracked_diagnostic_fails_without_candidate_consumption() -> None:
    report = json.loads((Path(__file__).parents[1] / "DEV_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed_dev_diagnostic"
    assert report["candidate_consumed"] is False
    assert report["real_sealed_reads"] == 0
    assert report["strategies"]["v34_deterministic_expansion"]["recall"] == (
        report["strategies"]["v32_base"]["recall"]
    )
