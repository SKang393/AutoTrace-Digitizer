# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from ml.ocr.recognizer_confirmed_acceptance_v9.dataset import build_scene
from ml.ocr.recognizer_confirmed_acceptance_v9.protocol import configuration
from ml.ocr.production_csharp_marker_gate_v4.protocol import (
    DEGRADATION_FAMILIES as V4_DEGRADATION_FAMILIES,
    RENDERER_FAMILIES as V4_RENDERER_FAMILIES,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_protocol_matches_python_contract() -> None:
    assert json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8")) == configuration()


def test_p1_is_fail_closed_before_selection_materialization() -> None:
    value = configuration()
    assert value["candidate_id"] == "P1"
    assert value["experiment_budget"] == 3
    assert value["candidate"]["official_minimum_confidence"] == 0.65
    assert value["selection_execution_authorized"] is False
    assert value["public_gate_authorized"] is False
    assert value["production_approval"] is False
    assert value["release_eligible"] is False


def test_fresh_selection_scene_is_deterministic_and_excludes_private_labels() -> None:
    first = build_scene(0x123456789ABCDEF, 7)
    second = build_scene(0x123456789ABCDEF, 7)
    assert first.scene_id == "ocr-recognizer-confirmed-v9-selection-00007"
    assert np.array_equal(first.raster, second.raster)
    assert first.renderer_family.endswith("v9-selection-h")
    assert first.degradation_family.endswith("v9-selection")
    assert len(first.text_truths) == 5
    labels = {truth.display_text for truth in first.text_truths}
    assert "Generalization" not in labels
    assert "Chandler" not in labels


def test_selection_families_are_distinct_and_all_roles_are_present() -> None:
    value = configuration()
    split = value["split"]
    assert set(split["renderer_families"]).isdisjoint(V4_RENDERER_FAMILIES)
    assert set(split["degradation_families"]).isdisjoint(V4_DEGRADATION_FAMILIES)
    scene = build_scene(0x8C1D4A20F77B1193, 0)
    assert scene.raster.shape == (320, 640)
    assert scene.raster.dtype.name == "uint8"
    assert {truth.role for truth in scene.text_truths} == {
        "y_tick", "x_tick", "phase_heading", "annotation", "legend_text",
    }


def test_future_selection_seal_and_authorization_remain_fail_closed() -> None:
    seal_path = ROOT / "SELECTION_SEAL.json"
    authorization_path = ROOT / "SELECTION_AUTHORIZATION.json"
    if not seal_path.exists():
        assert not authorization_path.exists()
        return

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["schema"] == "graphreader.ocr-recognizer-confirmed-selection-seal.v1"
    assert seal["candidate_id"] == "P1"
    assert seal["scene_count"] == 96
    assert seal["truth_region_count"] == 480
    assert seal["model_execution_count_at_freeze"] == 0
    assert seal["secret_seed_serialized"] is False
    assert "secret_seed" not in seal
    assert seal["selection_execution_authorized"] is False
    assert seal["public_gate_authorized"] is False
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
    for relative_path, expected in seal["source_sha256"].items():
        assert sha256((ROOT.parents[2] / relative_path).read_bytes()).hexdigest() == expected

    if not authorization_path.exists():
        return
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert authorization["schema"] == (
        "graphreader.ocr-recognizer-confirmed-selection-authorization.v1"
    )
    assert authorization["execution_authorized"] is True
    assert authorization["execution_count_authorized"] == 1
    assert authorization["fixture_archive_sha256"] == seal["fixture_archive_sha256"]
    assert authorization["fixture_manifest_sha256"] == seal["fixture_manifest_sha256"]
    assert authorization["split_seal_sha256"] == sha256(seal_path.read_bytes()).hexdigest()
    assert authorization["candidate_sha256"] == sorted(
        item["sha256"] for item in configuration()["models"].values()
    )
    assert authorization["exact_test"] == (
        "OcrV9CandidateSelectionTests."
        "FreshVisibleSelectionExecutesOnceThroughCSharpCpuCandidate"
    )
    assert authorization["rerun_or_repair_authorized"] is False
    assert authorization["public_gate_authorized"] is False
    assert authorization["manifest_creation_authorized"] is False
    assert authorization["model_store_promotion_authorized"] is False
    assert authorization["private_validation_authorized"] is False
    assert authorization["production_approval"] is False
    assert authorization["release_eligible"] is False


def test_consumed_p1_result_records_only_aggregate_failure() -> None:
    result_path = ROOT / "P1_SELECTION_RESULT.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema"] == "graphreader.ocr-recognizer-confirmed-selection-result.v1"
    assert result["candidate_id"] == "P1"
    assert result["selection_report_sha256"] == (
        "49c4c84ca4667e2263a0b66a9ae054ec00a7e3ecb542a8d66d871ce36ff643b0"
    )
    assert result["metrics"] == {
        "scene_count": 96,
        "truth_region_count": 480,
        "exact_detection_scene_count": 92,
        "true_positives": 480,
        "false_positives": 4,
        "false_negatives": 0,
        "duplicate_region_count": 0,
        "recognition_exact_match": 1.0,
        "character_error_rate": 0.0,
        "role_accuracy": 1.0,
        "numeric_exact_match": 1.0,
        "word_exact_match": 1.0,
        "ambiguity_exact_match": 1.0,
        "prohibited_structure_hits": 4,
    }
    assert "cases" not in result
    assert "predictions" not in result
    assert result["selection_gates_passed"] is False
    assert result["execution_consumed"] is True
    assert result["rerun_or_repair_authorized"] is False
    assert result["case_level_tuning_authorized"] is False
    assert result["public_gate_authorized"] is False
    assert result["private_validation_authorized"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
