# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.proposal_confirmation_calibrator_v19.model import ProposalConfirmationCalibrator
from ml.ocr.proposal_confirmation_calibrator_v19.protocol import (
    DETECTOR_PATH, DETECTOR_SHA256, FEATURE_COUNT, RECOGNIZER_PATH, RECOGNIZER_SHA256,
    REVISION, TRIGGER_RESULT_PATH, TRIGGER_RESULT_SHA256, protocol_configuration,
)
from ml.ocr.proposal_confirmation_calibrator_v19.sealed_gate import EVALUATOR_SOURCE_PATHS
from ml.ocr.proposal_confirmation_calibrator_v19.train_p1 import RUNNER_SOURCE_PATHS


ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "ml/ocr/proposal_confirmation_calibrator_v19"


def _read(name: str) -> dict[str, object]:
    return json.loads((MODULE / name).read_text(encoding="utf-8"))


def test_protocol_and_fixed_inputs_are_checksum_bound() -> None:
    assert (MODULE / "PROTOCOL.json").read_bytes() == canonical_json_bytes(protocol_configuration())
    assert sha256_file(ROOT / DETECTOR_PATH) == DETECTOR_SHA256
    assert sha256_file(ROOT / RECOGNIZER_PATH) == RECOGNIZER_SHA256
    assert sha256_file(ROOT / TRIGGER_RESULT_PATH) == TRIGGER_RESULT_SHA256
    protocol = protocol_configuration()
    assert protocol["revision"] == REVISION
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["execution_authorized"] is False
    assert protocol["trigger_evidence"]["case_level_details_used"] is False
    assert protocol["trigger_evidence"]["fixture_bytes_scene_truth_or_case_identity_used"] is False


def test_stored_splits_and_source_bundles_match_frozen_manifests() -> None:
    selection = _read("SELECTION_MANIFEST.json")
    for split in ("train", "validation", "sealed_public"):
        item = selection[split]
        assert sha256_file(ROOT / item["fixture_archive_path"]) == item["fixture_archive_sha256"]
        assert sha256_file(ROOT / item["private_manifest_path"]) == item["private_manifest_sha256"]
    config = _read("training/p1.json")
    gate = _read("gates/sealed-public-v1.json")
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS) == gate["expected_evaluator_source_bundle_sha256"]
    assert config["expected_optimizer_steps"] == 180
    assert config["proposal_count"] == 2304


def test_calibrator_contract_is_small_and_deterministic() -> None:
    first = ProposalConfirmationCalibrator(seed=20262219)
    second = ProposalConfirmationCalibrator(seed=20262219)
    values = torch.from_numpy(np.arange(FEATURE_COUNT * 3, dtype=np.float32).reshape(3, FEATURE_COUNT) / 100.0)
    with torch.inference_mode():
        first_output = first(values)
        second_output = second(values)
    assert first_output.shape == (3, 2)
    assert torch.equal(first_output, second_output)


def test_public_gate_and_production_remain_fail_closed() -> None:
    seal = _read("SEALED_PUBLIC_TEST_SEAL.json")
    gate = _read("gates/sealed-public-v1.json")
    result = _read("P1_RESULT.json")
    ledger = json.loads(
        (ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in ledger["revisions"] if item.get("revision") == REVISION)
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert gate["production_approval"] is False
    assert gate["release_eligible"] is False
    assert result["status"] == "failed_selection"
    assert result["selection_gate_passed"] is False
    assert result["case_level_details_emitted"] is False
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["selection_metrics"]["false_positives"] == 1
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["prohibited_structure_hits"] == 1
    assert result["passing_threshold_window"] == []
    assert entry["status"] == "candidate_1_failed_selection"
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["execution_authorized"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["p1_result_sha256"] == sha256_file(MODULE / "P1_RESULT.json")
    assert not (MODULE / "PUBLIC_GATE_REPORT.json").exists()
    local_report = MODULE / "artifacts/P1-run/candidate-report.json"
    if local_report.exists():
        assert sha256_file(local_report) == sha256_file(MODULE / "P1_RESULT.json")
