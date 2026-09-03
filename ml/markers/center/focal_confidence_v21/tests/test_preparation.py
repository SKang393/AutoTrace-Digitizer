# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import ast
import json
from pathlib import Path

import torch
from torch.nn import functional

from ml.markers.center.focal_confidence_v21 import protocol
from ml.markers.center.focal_confidence_v21.focal_loss import binary_focal_loss_with_logits, regression_terms, v21_loss
from ml.markers.center.focal_confidence_v21.train import RUNNER_SOURCE_PATHS, build_train_scenes as build_v21_train_scenes
from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.tail_coverage_v20.training_families import build_train_scenes as build_v20_train_scenes
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


ROOT = Path(__file__).parents[5]
CONFIG_PATH = ROOT / "ml/markers/center/focal_confidence_v21/training/p1.json"


def test_binds_exact_v20_result_and_diagnostic_one_candidate_budget() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = ROOT / protocol.V20_RESULT_PATH
    diagnostic = ROOT / protocol.V20_DIAGNOSTIC_PATH
    assert sha256_file(result) == protocol.V20_RESULT_SHA256
    assert sha256_file(diagnostic) == protocol.V20_DIAGNOSTIC_SHA256
    assert config["v20_result_sha256"] == protocol.V20_RESULT_SHA256
    assert config["v20_diagnostic_sha256"] == protocol.V20_DIAGNOSTIC_SHA256
    assert config["geometry_veto_guard"] == protocol.GEOMETRY_VETO_GUARD
    diagnostic_payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    counts = diagnostic_payload["miss_stage"]["counts"]
    assert counts["marker_geometry_veto"] == 2
    assert counts["unmasked_artifact_veto"] == 1
    assert config["sealed_candidate_budget"] == 1
    assert config["sealed_runs"] == 0
    assert config["public_gate_evaluations"] == 0


def test_v20_scene_tensors_are_reused_byte_identically() -> None:
    first = build_v20_train_scenes()
    second = build_v21_train_scenes()
    assert len(first) == len(second) == 21
    assert [scene.scene_id for scene in first] == [scene.scene_id for scene in second]
    assert all(left.tensor.numpy().tobytes() == right.tensor.numpy().tobytes() for left, right in zip(first, second, strict=True))
    assert [scene.radii for scene in first] == [scene.radii for scene in second]


def test_focal_loss_is_deterministic_on_fixed_logits() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0, -5.0], dtype=torch.float32)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    expected = torch.tensor([0.00135267, 0.04332170, 0.00045089, 0.0000002256])
    first = binary_focal_loss_with_logits(logits, targets, alpha=protocol.FOCAL_ALPHA, gamma=protocol.FOCAL_GAMMA)
    second = binary_focal_loss_with_logits(logits, targets, alpha=protocol.FOCAL_ALPHA, gamma=protocol.FOCAL_GAMMA)
    assert torch.equal(first, second)
    assert torch.allclose(first, expected, atol=1e-6, rtol=0.0)


def test_regression_terms_match_frozen_v16_formula() -> None:
    raw = torch.tensor([[0.4, 0.2, -0.3, 0.7], [-1.0, 0.8, 0.1, -0.4], [0.2, -0.2, 0.5, 0.0]])
    labels = torch.tensor([1.0, 0.0, 1.0])
    offsets = torch.tensor([[0.1, -0.2], [0.0, 0.0], [-0.3, 0.4]])
    radii = torch.tensor([4.0, 3.0, 7.5])
    actual_offset, actual_radius = regression_terms(raw, labels, offsets, radii)
    positive = labels > 0.5
    expected_offset = functional.smooth_l1_loss(torch.tanh(raw[positive, 1:3]) * 0.75, offsets[positive])
    expected_radius = functional.smooth_l1_loss(2.5 + torch.sigmoid(raw[positive, 3]) * 5.5, radii[positive].clamp(2.5, 8.0))
    assert torch.equal(actual_offset, expected_offset)
    assert torch.equal(actual_radius, expected_radius)
    hard = torch.tensor([False, True, False])
    combined = v21_loss(raw, labels, offsets, radii, hard, positive_weight=16.0, hard_weight=5.0)
    classification = (binary_focal_loss_with_logits(raw[:, 0], labels) * torch.tensor([16.0, 5.0, 16.0])).mean()
    assert torch.equal(combined, classification + 1.25 * expected_offset + 0.25 * expected_radius)


def test_v16_runtime_contract_and_v21_guards_remain_unchanged() -> None:
    model = ScaleClassifierNet(ModelConfig(seed=20260902)).eval()
    for count in (1, 8, 37):
        with torch.inference_mode():
            output = model(torch.zeros((count, 3, 33, 33)))
        assert tuple(output.shape) == (count, 4)
        assert float(output[:, 0].min()) >= 0.0
        assert float(output[:, 0].max()) <= 1.0
        assert float(output[:, 3].min()) >= 2.5
        assert float(output[:, 3].max()) <= 8.0
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["fixed_confidence_threshold"] == 0.25
    assert config["label_positive_distance_px"] == 3.0
    assert config["onnx_dynamic_candidate_counts"] == [1, 8, 37]


def test_runner_has_no_private_or_sealed_read() -> None:
    source = (ROOT / "ml/markers/center/focal_confidence_v21/train.py").read_text(encoding="utf-8")
    assert "data/manual data" not in source
    assert 'build_selection_scenes("sealed")' not in source
    ast.parse(source)


def test_frozen_v20_sources_and_runner_source_hash_are_current() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert sha256_file(ROOT / "ml/markers/center/tail_coverage_v20/protocol.py") == protocol.V20_PROTOCOL_SHA256
    assert sha256_file(ROOT / "ml/markers/center/tail_coverage_v20/training_families.py") == protocol.V20_TRAINING_FAMILIES_SHA256
    assert sha256_file(ROOT / "ml/markers/center/tail_coverage_v20/train.py") == protocol.V20_TRAIN_RUNNER_SHA256
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    result = json.loads((Path(__file__).parents[1] / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["dev_metrics"]["precision"] == 1.0
    assert result["dev_metrics"]["recall"] == 0.9166666666666666
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
