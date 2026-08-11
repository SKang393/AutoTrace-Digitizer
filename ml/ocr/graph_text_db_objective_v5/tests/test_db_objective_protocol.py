# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed checks for the preregistered DB-objective detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.graph_text_db_objective_v5.dataset import (
    GENERIC_TEXT,
    _db_supervision,
    build_validation_split,
    render_training_tiles,
    split_fingerprint,
    training_split_fingerprint,
)
from ml.ocr.graph_text_db_objective_v5.losses import db_objective_loss
from ml.ocr.graph_text_db_objective_v5.losses_p2 import db_objective_loss_with_boundary_margin
from ml.ocr.graph_text_db_objective_v5.model import DbObjectiveTextRegionNet
from ml.ocr.graph_text_db_objective_v5.diagnose_p2 import (
    EXPECTED_ONNX_SHA256 as P2_DIAGNOSIS_ONNX_SHA256,
    EXPECTED_REPORT_SHA256 as P2_DIAGNOSIS_REPORT_SHA256,
    OUTPUT_PATH as P2_DIAGNOSIS_OUTPUT_PATH,
)
from ml.ocr.graph_text_db_objective_v5.prepare_split import SPLIT_SOURCE_PATHS
from ml.ocr.graph_text_db_objective_v5.protocol import (
    BATCH_SIZE,
    EPOCHS,
    EXPERIMENT_BUDGET,
    REVISION,
    SPLITS,
    TRAIN_SAMPLE_COUNT,
    protocol_configuration,
)
from ml.ocr.graph_text_db_objective_v5.train_p1 import RUNNER_SOURCE_PATHS
from ml.ocr.graph_text_db_objective_v5.train_p2 import RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[4]
REVISION_ROOT = REPO_ROOT / "ml/ocr/graph_text_db_objective_v5"


def test_protocol_is_distinct_fixed_and_fail_closed() -> None:
    protocol = protocol_configuration()
    assert protocol["revision"] == REVISION == "graph-text-db-objective-v5"
    assert protocol["architecture"] == "dual-head-db-stride4-v1"
    assert protocol["experiment_budget"] == EXPERIMENT_BUDGET == 3
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["training"]["db_binarization_k"] == 50.0
    assert protocol["training"]["shrink_loss_weight"] == 5.0
    assert protocol["training"]["threshold_loss_weight"] == 10.0
    assert protocol["training"]["binary_loss_weight"] == 1.0
    assert protocol["postprocessing"]["probability_threshold"] == 0.30
    assert protocol["postprocessing"]["box_confidence_threshold"] == 0.60
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_split_families_are_unique_and_forbidden_sources_are_absent() -> None:
    current = {item.renderer_family for item in SPLITS} | {item.degradation_family for item in SPLITS}
    prior: set[str] = set()
    for directory in ("graph_text_detector_v1", "graph_text_balanced_v2", "graph_text_ignore_band_v3", "graph_text_stride4_v4"):
        protocol = json.loads((REPO_ROOT / "ml/ocr" / directory / "PROTOCOL.json").read_text(encoding="utf-8"))
        for split in protocol["splits"]:
            prior.add(split["renderer_family"])
            prior.add(split["degradation_family"])
    assert current.isdisjoint(prior)
    joined = " ".join(GENERIC_TEXT).casefold()
    assert "generalization" not in joined
    assert "chandler" not in joined


def test_training_tiles_are_deterministic_and_bind_all_db_maps() -> None:
    first = render_training_tiles(0)
    repeated = render_training_tiles(0)
    exclusion = render_training_tiles(639)
    assert len(first) == len(repeated) == len(exclusion) == 3
    for left, right in zip(first, repeated, strict=True):
        assert left.tile_id == right.tile_id
        assert np.array_equal(left.bgr, right.bgr)
        assert np.array_equal(left.shrink_target, right.shrink_target)
        assert np.array_equal(left.threshold_target, right.threshold_target)
        assert left.bgr.shape == (192, 512, 3)
        assert left.shrink_target.shape == left.shrink_mask.shape == (192, 512)
        assert left.threshold_target.shape == left.threshold_mask.shape == (192, 512)
    assert any(np.any(tile.shrink_target) for tile in first)
    assert any(np.any(tile.threshold_mask) for tile in first)
    assert all(not np.any(tile.shrink_target) for tile in exclusion)
    assert all(not np.any(tile.threshold_mask) for tile in exclusion)


def test_db_supervision_has_fixed_range_and_boundary_peak() -> None:
    target = np.zeros((64, 128), dtype=np.uint8)
    target[20:44, 30:98] = 255
    shrink, shrink_mask, threshold, threshold_mask = _db_supervision(target)
    assert np.any(shrink)
    assert np.any(shrink_mask == 0)
    assert np.any(threshold_mask)
    values = threshold[threshold_mask > 0]
    assert float(values.min()) >= 0.30
    assert float(values.max()) == pytest.approx(0.70, abs=1e-6)


def test_model_training_maps_and_export_map_are_probabilities() -> None:
    model = DbObjectiveTextRegionNet()
    value = torch.zeros((1, 3, 32, 64), dtype=torch.float32)
    inference = model(value)
    shrink, threshold, binary = model.forward_training(value)
    assert inference.shape == shrink.shape == threshold.shape == binary.shape == (1, 1, 32, 64)
    assert torch.equal(inference, shrink)
    for output in (inference, threshold, binary):
        assert torch.isfinite(output).all()
        assert float(output.detach().min()) >= 0.0
        assert float(output.detach().max()) <= 1.0


def test_db_loss_is_finite_and_reaches_both_heads() -> None:
    model = DbObjectiveTextRegionNet()
    value = torch.zeros((2, 3, 32, 64), dtype=torch.float32)
    outputs = model.forward_training(value)
    shrink_target = torch.zeros_like(outputs[0])
    shrink_target[:, :, 10:18, 20:44] = 1.0
    shrink_mask = torch.ones_like(outputs[0])
    threshold_target = torch.full_like(outputs[0], 0.3)
    threshold_mask = torch.zeros_like(outputs[0])
    threshold_mask[:, :, 7:21, 17:47] = 1.0
    losses = db_objective_loss(outputs, shrink_target, shrink_mask, threshold_target, threshold_mask)
    assert all(torch.isfinite(value) for value in losses)
    losses[0].backward()
    assert model.shrink_head.weight.grad is not None
    assert model.threshold_head.weight.grad is not None


def test_p2_boundary_margin_is_one_sided_and_confined_to_ignored_band() -> None:
    shrink = torch.full((1, 1, 8, 12), 0.1, dtype=torch.float32)
    threshold = torch.full_like(shrink, 0.5)
    binary = torch.sigmoid(50.0 * (shrink - threshold))
    target = torch.zeros_like(shrink)
    target[:, :, 3:5, 4:8] = 1.0
    mask = torch.ones_like(shrink)
    mask[:, :, 2:6, 3:9] = 0.0
    mask[target > 0.5] = 1.0
    threshold_target = torch.full_like(shrink, 0.3)
    threshold_mask = torch.zeros_like(shrink)
    threshold_mask[:, :, 1:7, 2:10] = 1.0

    below = db_objective_loss_with_boundary_margin(
        (shrink, threshold, binary),
        target,
        mask,
        threshold_target,
        threshold_mask,
        boundary_probability_ceiling=0.25,
        boundary_margin_loss_weight=1.0,
    )
    changed = shrink.clone()
    changed[:, :, 2:6, 3:9] = 0.5
    changed[target > 0.5] = 0.1
    above = db_objective_loss_with_boundary_margin(
        (changed, threshold, binary),
        target,
        mask,
        threshold_target,
        threshold_mask,
        boundary_probability_ceiling=0.25,
        boundary_margin_loss_weight=1.0,
    )
    assert float(below[4]) == 0.0
    assert float(above[4]) > 0.0
    assert torch.equal(below[1], above[1])
    assert torch.equal(below[2], above[2])
    assert torch.equal(below[3], above[3])


def test_frozen_evidence_and_source_hashes_are_exact() -> None:
    paths = {
        name: REVISION_ROOT / name
        for name in ("PROTOCOL.json", "SELECTION_MANIFEST.json", "SEALED_PUBLIC_TEST_SEAL.json", "P1_PREREGISTRATION.json", "training/p1.json")
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("Split has not been frozen yet")
    protocol = json.loads(paths["PROTOCOL.json"].read_text(encoding="utf-8"))
    selection = json.loads(paths["SELECTION_MANIFEST.json"].read_text(encoding="utf-8"))
    seal = json.loads(paths["SEALED_PUBLIC_TEST_SEAL.json"].read_text(encoding="utf-8"))
    preregistration = json.loads(paths["P1_PREREGISTRATION.json"].read_text(encoding="utf-8"))
    training = json.loads(paths["training/p1.json"].read_text(encoding="utf-8"))
    assert protocol["split_generator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    assert training["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    assert selection["training_split_fingerprint"] == training_split_fingerprint()
    validation = build_validation_split()
    assert selection["validation_split_fingerprint"] == split_fingerprint(validation)
    assert training["expected_optimizer_steps"] == EPOCHS * (TRAIN_SAMPLE_COUNT // BATCH_SIZE) == 2880
    assert preregistration["candidate_config_sha256"] == sha256_file(paths["training/p1.json"])
    assert preregistration["sealed_public_test_seal_sha256"] == sha256_file(paths["SEALED_PUBLIC_TEST_SEAL.json"])
    assert seal["truth_hidden_from_training_runner"] is True
    assert seal["public_release_eligible"] is False


def test_p1_result_and_p2_preregistration_are_checksum_bound_and_public_closed() -> None:
    result_path = REVISION_ROOT / "P1_RESULT.json"
    preregistration_path = REVISION_ROOT / "P2_PREREGISTRATION.json"
    config_path = REVISION_ROOT / "training/p2.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_selection"
    assert result["selection_metrics"]["exact_fixture_count"] == 47
    assert result["selection_metrics"]["false_region_count"] == 95
    assert result["diagnosis"]["diagnostic_runs"] == 1
    assert result["diagnosis"]["threshold_sweeps"] == 0
    assert preregistration["p1_result_sha256"] == sha256_file(result_path)
    assert preregistration["candidate_config_sha256"] == sha256_file(config_path)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS)
    assert config["boundary_probability_ceiling"] == 0.25
    assert config["boundary_margin_loss_weight"] == 1.0
    assert preregistration["public_gate_authorized"] is False
    assert preregistration["sealed_public_archive_opened"] is False


def test_p2_result_is_checksum_bound_and_failed_closed() -> None:
    result_path = REVISION_ROOT / "P2_RESULT.json"
    report_path = REVISION_ROOT / "artifacts/P2-run/candidate-report.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_selection"
    assert result["selection_gate_passed"] is False
    assert result["probability_contract_passed"] is True
    assert result["onnx_parity_passed"] is True
    assert result["selection_metrics"]["exact_fixture_count"] == 84
    assert result["selection_metrics"]["false_region_count"] == 48
    assert result["selection_metrics"]["exclusion_false_region_count"] == 0
    assert result["selection_metrics"]["text_missed_fixture_count"] == 51
    assert result["selection_report_sha256"] == sha256_file(report_path)
    assert result["selection_metrics"] == {
        key: value for key, value in report["selection_metrics"].items() if key != "records"
    }
    assert result["sealed_public_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["production_approval"] is False
    assert result["release_eligible"] is False


def test_p2_diagnosis_is_bound_to_the_consumed_payload_and_single_output() -> None:
    result = json.loads((REVISION_ROOT / "P2_RESULT.json").read_text(encoding="utf-8"))
    assert P2_DIAGNOSIS_REPORT_SHA256 == result["selection_report_sha256"]
    assert P2_DIAGNOSIS_ONNX_SHA256 == result["onnx_sha256"]
    assert P2_DIAGNOSIS_OUTPUT_PATH.as_posix() == (
        "ml/ocr/graph_text_db_objective_v5/artifacts/P2-run/selection-diagnosis.json"
    )


def test_canonical_budget_consumes_failed_p2_and_keeps_p3_unregistered() -> None:
    ledger = json.loads((REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["task"] == "ocr-detection" and item["revision"] == REVISION)
    assert entry["status"] == "candidate_2_failed_selection"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P3"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["p1_selection_exact_fixture_count"] == 47
    assert entry["p1_diagnostic_runs"] == 1
    assert entry["p2_boundary_probability_ceiling"] == 0.25
    assert entry["p2_boundary_margin_loss_weight"] == 1.0
    assert entry["p2_result_sha256"] == sha256_file(REVISION_ROOT / "P2_RESULT.json")
    assert entry["p2_training_report_sha256"] == sha256_file(
        REVISION_ROOT / "artifacts/P2-run/candidate-report.json"
    )
    assert entry["p2_selection_exact_fixture_count"] == 84
    assert entry["p2_selection_false_region_count"] == 48
    assert entry["p2_selection_exclusion_false_region_count"] == 0
    assert entry["p2_selection_gate_passed"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
