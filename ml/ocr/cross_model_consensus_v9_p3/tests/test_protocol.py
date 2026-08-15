# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ml.ocr.cross_model_consensus_v9_p3.dataset import build_scene
from ml.ocr.cross_model_consensus_v9_p3.protocol import (
    P2_PUBLIC_RESULT_SHA256,
    REQUIRED_ROLES,
    V11_PUBLIC_RESULT_SHA256,
    configuration,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]


def test_tracked_protocol_matches_python_contract() -> None:
    assert json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8")) == configuration()


def test_final_candidate_is_zero_optimizer_single_run_and_fail_closed() -> None:
    value = configuration()
    assert value["candidate_id"] == "P3"
    assert value["candidate_budget"]["candidate_number"] == 3
    assert value["candidate_budget"]["candidate_limit"] == 3
    assert value["candidate_budget"]["final_candidate"] is True
    assert value["candidate_budget"]["optimizer_steps"] == 0
    assert value["splits"]["predecessor_public_bytes_used_for_selection_or_tuning"] is False
    assert value["selection_execution_authorized"] is False
    assert value["public_execution_authorized"] is False
    assert value["private_validation_authorized"] is False
    assert value["production_approval"] is False
    assert value["release_eligible"] is False


def test_predecessor_results_are_checksum_bound_without_case_detail() -> None:
    p2 = REPO_ROOT / "ml/ocr/selected_confidence_public_gate_v9_p2/PUBLIC_GATE_RESULT.json"
    v11 = REPO_ROOT / "ml/ocr/composite_proposal_role_v11/PUBLIC_GATE_RESULT.json"
    assert sha256(p2.read_bytes()).hexdigest() == P2_PUBLIC_RESULT_SHA256
    assert sha256(v11.read_bytes()).hexdigest() == V11_PUBLIC_RESULT_SHA256
    p2_result = json.loads(p2.read_text(encoding="utf-8"))
    assert p2_result["report_contains_aggregate_only"] is True
    assert p2_result["case_level_evidence_used"] is False


def test_fresh_selection_and_public_scenes_are_disjoint_and_private_free() -> None:
    selection = build_scene("selection", 0xA183D94270BC5519, 37)
    public = build_scene("sealed_public", 0xC715E8394A0261BD, 37)
    assert selection.scene_id != public.scene_id
    assert not np.array_equal(selection.raster, public.raster)
    assert selection.raster.shape == public.raster.shape == (320, 640)
    assert {item.role for item in selection.text_truths} == set(REQUIRED_ROLES)
    assert {item.role for item in public.text_truths} == set(REQUIRED_ROLES)
    labels = {item.display_text for item in selection.text_truths + public.text_truths}
    assert "Chandler" not in labels
    assert "Generalization" not in labels


def test_no_split_authorization_or_result_exists_at_preregistration() -> None:
    if not (ROOT / "SPLIT_SEAL.json").exists():
        assert not (ROOT / "SELECTION_AUTHORIZATION.json").exists()
        assert not (ROOT / "P3_SELECTION_RESULT.json").exists()
        assert not (ROOT / "PUBLIC_GATE_AUTHORIZATION.json").exists()
        assert not (ROOT / "PUBLIC_GATE_RESULT.json").exists()
