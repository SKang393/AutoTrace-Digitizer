# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ml.ocr.selected_confidence_public_gate_v9_p2.dataset import build_scene
from ml.ocr.selected_confidence_public_gate_v9_p2.protocol import (
    P2_SELECTION_RESULT_SHA256,
    REQUIRED_ROLES,
    configuration,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]


def test_tracked_protocol_matches_python_contract() -> None:
    assert json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8")) == configuration()


def test_public_gate_is_single_run_and_fail_closed() -> None:
    value = configuration()
    assert value["candidate_id"] == "P2"
    assert value["predecessor_selection"]["selection_gates_passed"] is True
    assert value["predecessor_selection"]["case_level_evidence_used"] is False
    assert value["split"]["required_roles"] == list(REQUIRED_ROLES)
    assert value["gates"]["single_authorized_execution"] is True
    assert value["public_execution_authorized"] is False
    assert value["marker_creation_evaluated"] is False
    assert value["artifact_mask_production_approval"] is False
    assert value["private_validation_authorized"] is False
    assert value["production_approval"] is False
    assert value["release_eligible"] is False


def test_predecessor_selection_result_is_checksum_bound_and_consumed() -> None:
    path = REPO_ROOT / "ml/ocr/selected_confidence_acceptance_v9_p2/P2_SELECTION_RESULT.json"
    assert sha256(path.read_bytes()).hexdigest() == P2_SELECTION_RESULT_SHA256
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["selection_gates_passed"] is True
    assert result["execution_consumed"] is True
    assert result["rerun_or_repair_authorized"] is False
    assert result["public_gate_authorized"] is False


def test_fresh_scene_is_deterministic_eight_role_and_private_free() -> None:
    first = build_scene(0xA71C3904D82561E7, 73)
    second = build_scene(0xA71C3904D82561E7, 73)
    assert first.scene_id == "ocr-selected-confidence-v9-p2-public-00073"
    assert np.array_equal(first.raster, second.raster)
    assert first.raster.shape == (320, 640)
    assert first.raster.dtype.name == "uint8"
    assert {truth.role for truth in first.text_truths} == set(REQUIRED_ROLES)
    labels = {truth.display_text for truth in first.text_truths}
    assert "Generalization" not in labels
    assert "Chandler" not in labels


def test_no_public_identity_authorization_or_result_exists_at_preregistration() -> None:
    seal = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    authorization = ROOT / "PUBLIC_GATE_AUTHORIZATION.json"
    result = ROOT / "PUBLIC_GATE_RESULT.json"
    if not seal.exists():
        assert not authorization.exists()
        assert not result.exists()


def test_authorization_binds_frozen_public_identity_and_remains_fail_closed() -> None:
    seal_path = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    authorization_path = ROOT / "PUBLIC_GATE_AUTHORIZATION.json"
    assert seal_path.exists()
    assert authorization_path.exists()

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert authorization["schema"] == "graphreader.ocr-selected-confidence-public-authorization.v1"
    assert authorization["candidate_id"] == "P2"
    assert authorization["sealed_identity_commit"] == "209ca6ffb917eada1657e90a00cb59fc84e685e0"
    assert authorization["fixture_archive_sha256"] == seal["fixture_archive_sha256"]
    assert authorization["fixture_manifest_sha256"] == seal["fixture_manifest_sha256"]
    assert authorization["public_seal_sha256"] == sha256(seal_path.read_bytes()).hexdigest()
    assert authorization["execution_authorized"] is True
    assert authorization["public_gate_authorized"] is True
    assert authorization["execution_count_authorized"] == 1
    assert authorization["provider"] == "CPUExecutionProvider"
    assert authorization["rerun_or_repair_authorized"] is False
    assert authorization["marker_stage_authorized"] is False
    assert authorization["artifact_mask_production_approval"] is False
    assert authorization["manifest_creation_authorized"] is False
    assert authorization["model_store_promotion_authorized"] is False
    assert authorization["private_validation_authorized"] is False
    assert authorization["production_approval"] is False
    assert authorization["release_eligible"] is False


def test_consumed_result_records_aggregate_failure_and_cannot_promote() -> None:
    result_path = ROOT / "PUBLIC_GATE_RESULT.json"
    authorization_path = ROOT / "PUBLIC_GATE_AUTHORIZATION.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["schema"] == "graphreader.ocr-selected-confidence-public-result.v1"
    assert result["candidate_id"] == "P2"
    assert result["execution_consumed"] is True
    assert result["execution_count"] == 1
    assert result["authorization_sha256"] == sha256(authorization_path.read_bytes()).hexdigest()
    assert result["report_contains_aggregate_only"] is True
    assert result["case_level_evidence_used"] is False
    assert result["direct_runtime_evidence_passed"] is True
    assert result["public_gate_passed"] is False
    assert result["full_eight_role_coverage_proven"] is False
    assert result["metrics"]["scene_count"] == 160
    assert result["metrics"]["truth_region_count"] == 1280
    assert result["metrics"]["exact_detection_scene_count"] == 87
    assert result["metrics"]["false_positives"] == 1
    assert result["metrics"]["false_negatives"] == 74
    assert result["metrics"]["duplicate_region_count"] == 0
    assert result["metrics"]["prohibited_structure_hits"] == 1
    assert result["rerun_or_repair_authorized"] is False
    assert result["marker_stage_authorized"] is False
    assert result["artifact_mask_production_approval"] is False
    assert result["manifest_creation_authorized"] is False
    assert result["model_store_promotion_authorized"] is False
    assert result["private_validation_authorized"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False

    local_report = REPO_ROOT / result["report_path"]
    if local_report.exists():
        assert sha256(local_report.read_bytes()).hexdigest() == result["report_sha256"]
