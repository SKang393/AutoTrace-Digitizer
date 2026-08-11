# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_geometric_v4.dataset import build_split
from ml.ocr.component_structural_filter_v1.pipeline import component_height_ratios, decode_raster
from ml.ocr.component_structural_filter_v1.protocol import (
    REVISION,
    SOURCE_ONNX_SHA256,
    STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
    TASK,
)
from ml.ocr.component_structural_filter_v1.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.component_structural_filter_v1.selection import CONFIG_PATH, RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_structural_filter_v1"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def test_protocol_preregisters_one_zero_training_candidate() -> None:
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["task"] == TASK
    assert protocol["revision"] == REVISION
    assert protocol["candidate_ids"] == ["P1"]
    assert protocol["experiment_budget"] == 1
    assert protocol["optimizer_steps"] == 0
    assert protocol["weights_changed"] is False
    assert protocol["rule"] == {
        "field": "component_height_ratio",
        "operator": ">=",
        "position": "before_classifier",
        "reject_minimum": STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
    }
    assert protocol["source_onnx_sha256"] == SOURCE_ONNX_SHA256
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_fixed_rule_rejects_all_validation_dividers_without_rejecting_numeric_labels() -> None:
    validation = build_split("validation")
    dividers = [sample for sample in validation if sample.exclusion_kind == "divider"]
    positives = [sample for sample in validation if sample.exclusion_kind is None]
    assert len(dividers) == 14
    assert max(max(component_height_ratios(sample.raster)) for sample in positives) < STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO
    assert min(max(component_height_ratios(sample.raster)) for sample in dividers) >= STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO

    calls = 0

    def should_not_run(_: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("The classifier must not run for a structural rejection")

    prediction, rejected, _ = decode_raster(dividers[0].raster, should_not_run, 0.55)
    assert prediction == ""
    assert rejected is True
    assert calls == 0


def test_candidate_and_public_source_bundles_are_frozen() -> None:
    candidate = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert candidate["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS
    )
    public = json.loads((ROOT / "gates/sealed-public-v1.json").read_text(encoding="utf-8"))
    assert public["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert public["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))
    assert public["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")


def test_inherited_public_archive_remains_bound_and_unopened() -> None:
    seal = json.loads((ROOT / "SEALED_PUBLIC_TEST_SEAL.json").read_text(encoding="utf-8"))
    assert seal["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert seal["fixture_archive_sha256"] == "7845bc0628740fb24b95e0367bceab8b5b6d186adacc5f58c5c6c3bdcc97dc76"
    assert seal["truth_hidden_from_selection_runner"] is True
    assert seal["chandler_included"] is False
    assert not (ROOT / "PUBLIC_GATE_REPORT.json").exists()


def test_canonical_budget_records_selection_and_exact_public_authorization() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["task"] == TASK and item["revision"] == REVISION)
    assert entry["status"] == "selection_passed_public_preregistered"
    assert entry["experiment_budget"] == 1
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(REPO_ROOT / CONFIG_PATH)
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["public_gate_config_sha256"] == sha256_file(ROOT / "gates/sealed-public-v1.json")
    result_path = ROOT / "P1_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert entry["p1_result_sha256"] == sha256_file(result_path)
    assert result["status"] == "selected"
    assert result["selection_gate_passed"] is True
    assert result["optimizer_steps"] == 0
    assert result["weights_changed"] is False
    assert result["sealed_public_archive_opened"] is False
    assert result["validation_marker_exclusion_accuracy"] == 1.0
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_authorized_onnx_sha256"] == result["source_onnx_sha256"]
    assert entry["public_gate_authorized_selection_report_sha256"] == result["report_sha256"]
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_tracked_training_seals_bind_the_selected_report() -> None:
    result = json.loads((ROOT / "P1_RESULT.json").read_text(encoding="utf-8"))
    seal_root = REPO_ROOT / "ml/markers/training-seals/ocr-recognition" / REVISION / "P1"
    opened_path = seal_root / "opened.json"
    completed_path = seal_root / "result.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    assert result["training_opened_seal_sha256"] == sha256_file(opened_path)
    assert result["training_result_seal_sha256"] == sha256_file(completed_path)
    assert completed["opened_sha256"] == result["training_opened_seal_sha256"]
    assert completed["report_sha256"] == result["report_sha256"]
    assert completed["status"] == "selected"


def test_no_tracked_approval_manifest_exists_for_structural_filter() -> None:
    manifests = list((REPO_ROOT / "models/manifest/ocr").glob("*.json"))
    assert all(REVISION not in path.read_text(encoding="utf-8") for path in manifests)
