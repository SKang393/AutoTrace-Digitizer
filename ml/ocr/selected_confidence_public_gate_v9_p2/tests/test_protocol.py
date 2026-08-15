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
