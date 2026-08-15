# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ml.ocr.recognizer_confirmed_acceptance_v9.protocol import (
    DEGRADATION_FAMILIES as P1_DEGRADATIONS,
    RENDERER_FAMILIES as P1_RENDERERS,
)
from ml.ocr.selected_confidence_acceptance_v9_p2.dataset import build_scene
from ml.ocr.selected_confidence_acceptance_v9_p2.protocol import (
    MODEL_SHA256,
    configuration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_protocol_matches_python_contract() -> None:
    assert json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8")) == configuration()


def test_p2_is_fail_closed_and_uses_only_the_p1_aggregate() -> None:
    value = configuration()
    assert value["candidate_id"] == "P2"
    assert value["candidate_number"] == 2
    assert value["experiment_budget"] == 3
    assert value["predecessor_aggregate"]["case_level_evidence_used"] is False
    assert value["predecessor_aggregate"]["false_positives"] == 4
    assert value["candidate"]["selected_text_minimum_confidence"] == 0.75
    assert value["selection_execution_authorized"] is False
    assert value["public_gate_authorized"] is False
    assert value["production_approval"] is False
    assert value["release_eligible"] is False


def test_fresh_scene_is_deterministic_distinct_and_private_free() -> None:
    first = build_scene(0x1C70E4A9823B5D11, 11)
    second = build_scene(0x1C70E4A9823B5D11, 11)
    assert first.scene_id == "ocr-selected-confidence-v9-p2-selection-00011"
    assert np.array_equal(first.raster, second.raster)
    assert first.raster.shape == (320, 640)
    assert first.raster.dtype.name == "uint8"
    assert first.renderer_family not in P1_RENDERERS
    assert first.degradation_family not in P1_DEGRADATIONS
    assert {truth.role for truth in first.text_truths} == {
        "y_tick", "x_tick", "phase_heading", "annotation", "legend_text",
    }
    labels = {truth.display_text for truth in first.text_truths}
    assert "Generalization" not in labels
    assert "Chandler" not in labels


def test_no_selection_identity_or_execution_result_exists_at_preregistration() -> None:
    seal = ROOT / "SELECTION_SEAL.json"
    authorization = ROOT / "SELECTION_AUTHORIZATION.json"
    if not seal.exists():
        assert not authorization.exists()
    assert not (ROOT / "P2_SELECTION_RESULT.json").exists()


def test_frozen_selection_authorization_is_exactly_one_run_and_fail_closed() -> None:
    seal_path = ROOT / "SELECTION_SEAL.json"
    authorization_path = ROOT / "SELECTION_AUTHORIZATION.json"
    if not authorization_path.exists():
        return

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert authorization["schema"] == (
        "graphreader.ocr-selected-confidence-selection-authorization.v1"
    )
    assert authorization["candidate_id"] == "P2"
    assert authorization["execution_authorized"] is True
    assert authorization["execution_count_authorized"] == 1
    assert authorization["provider"] == "CPUExecutionProvider"
    assert len(authorization["sealed_identity_commit"]) == 40
    int(authorization["sealed_identity_commit"], 16)
    assert authorization["fixture_archive_sha256"] == seal["fixture_archive_sha256"]
    assert authorization["fixture_manifest_sha256"] == seal["fixture_manifest_sha256"]
    assert authorization["split_seal_sha256"] == hashlib.sha256(
        seal_path.read_bytes()
    ).hexdigest()
    assert sorted(authorization["candidate_sha256"]) == sorted(MODEL_SHA256.values())
    assert authorization["exact_test"] == (
        "OcrV9P2CandidateSelectionTests."
        "FreshP2SelectionExecutesOnceThroughCSharpCpuCandidate"
    )
    assert authorization["result_path"] == (
        "artifacts/production-validation/ocr-v9-p2-selection-report.json"
    )
    for field in (
        "rerun_or_repair_authorized",
        "public_gate_authorized",
        "manifest_creation_authorized",
        "model_store_promotion_authorized",
        "private_validation_authorized",
        "production_approval",
        "release_eligible",
    ):
        assert authorization[field] is False
