# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.role_anchor_set_v23.dataset import proposal_summary, render_scene
from ml.ocr.role_anchor_set_v23.model import RoleAnchorSetNet
from ml.ocr.role_anchor_set_v23.protocol import (
    CANDIDATE_LIMIT,
    FEATURE_COUNT,
    ROLE_ORDER,
    V22_RESULT_PATH,
    V22_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.role_anchor_set_v23.train_p1 import (
    CONFIG_PATH,
    RUNNER_SOURCE_PATHS,
    _balanced_class_weights,
    _proposal_objective,
    preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_fresh_bounded_and_fail_closed() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (REPO_ROOT / "ml/ocr/role_anchor_set_v23/PROTOCOL.json").read_text(
            encoding="utf-8"
        )
    )
    assert tracked == json.loads(json.dumps(protocol))
    assert protocol["candidate_budget"]["candidate_limit"] == CANDIDATE_LIMIT == 3
    assert protocol["candidate_budget"]["optimizer_steps_maximum"] == 1280
    assert protocol["candidate_budget"]["public_execution_limit"] == 1
    assert protocol["fixture_identity_frozen"] is False
    assert protocol["training_authorized"] is False
    assert protocol["public_execution_authorized"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["private_validation_authorized"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert "Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]


def test_v22_trigger_is_exact_aggregate_only_terminal_record() -> None:
    path = REPO_ROOT / V22_RESULT_PATH
    assert sha256_file(path) == V22_RESULT_SHA256
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["candidate_id"] == "P3"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["case_level_details_emitted"] is False
    assert result["selection_metrics"]["false_positives"] == 0
    assert result["selection_metrics"]["false_negatives"] == 1
    assert result["selection_metrics"]["role_accuracy"] == 0.9765625
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert "cases" not in result and "predictions" not in result


def test_canonical_budget_authorizes_only_committed_p1() -> None:
    ledger = json.loads(
        (
            REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item["task"] == "ocr-detection-recognition"
        and item["revision"] == "graph-text-role-anchor-set-v23"
    )
    protocol_path = REPO_ROOT / "ml/ocr/role_anchor_set_v23/PROTOCOL.json"
    config_path = REPO_ROOT / CONFIG_PATH
    seal_path = REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json"
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["protocol_sha256"] == sha256_file(protocol_path)
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["split_materialized"] is True
    assert entry["split_seal_sha256"] == sha256_file(seal_path)
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["p1_expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    )
    assert entry["selection_evaluations"] == 0
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_p1_config_binds_runner_fixtures_and_locked_public_gate() -> None:
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    seal = json.loads(
        (REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["expected_optimizer_steps"] == 1280
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    )
    assert config["split_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json"
    )
    assert config["train_fixture_archive_sha256"] == seal["splits"]["train"]["archive_sha256"]
    assert config["selection_fixture_archive_sha256"] == seal["splits"]["validation"]["archive_sha256"]
    assert config["public_fixture_archive_sha256"] == seal["splits"]["sealed_public"]["archive_sha256"]
    assert config["selection_evaluation_limit"] == 1
    assert config["public_execution_authorized"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["marker_creation_evaluated"] is False
    assert config["private_or_article_images"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False


def test_p1_preflight_validates_exact_committed_inputs_without_execution() -> None:
    evidence = preflight()
    assert evidence["seal"]["optimizer_steps_at_freeze"] == 0
    assert evidence["seal"]["selection_evaluations"] == 0
    assert evidence["seal"]["public_evaluations"] == 0
    assert evidence["config"]["public_execution_authorized"] is False


def test_p1_balanced_losses_and_scene_extrema_margins_are_preregistered() -> None:
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    targets = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    weights = _balanced_class_weights(targets, 2, "proposal")
    assert torch.allclose(weights, torch.tensor([0.5, 1.5]))
    objective_targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    passing = torch.tensor([[2.0, -2.0], [1.5, -1.5], [-2.0, 2.0], [-1.5, 1.5]])
    failing_negative = passing.clone()
    failing_negative[1] = torch.tensor([-2.0, 2.0])
    failing_positive = passing.clone()
    failing_positive[3] = torch.tensor([2.0, -2.0])
    _, passing_negative, passing_positive = _proposal_objective(
        passing, objective_targets, torch.ones(2), config,
    )
    _, bad_negative, _ = _proposal_objective(
        failing_negative, objective_targets, torch.ones(2), config,
    )
    _, _, bad_positive = _proposal_objective(
        failing_positive, objective_targets, torch.ones(2), config,
    )
    assert float(passing_negative) == 0.0
    assert float(passing_positive) == 0.0
    assert float(bad_negative) > 0.0
    assert float(bad_positive) > 0.0


def test_split_families_and_seed_offsets_are_disjoint() -> None:
    registrations = [split_registration(name) for name in ("train", "validation", "sealed_public")]
    assert len({item.seed_offset for item in registrations}) == 3
    renderer_sets = [set(item.renderer_families) for item in registrations]
    degradation_sets = [set(item.degradation_families) for item in registrations]
    assert not any(
        renderer_sets[left] & renderer_sets[right]
        for left in range(3) for right in range(left + 1, 3)
    )
    assert not any(
        degradation_sets[left] & degradation_sets[right]
        for left in range(3) for right in range(left + 1, 3)
    )


def test_renderer_produces_one_proposal_for_every_role_truth() -> None:
    scenes = tuple(render_scene(split, 0) for split in ("train", "validation", "sealed_public"))
    for scene in scenes:
        summary = proposal_summary((scene,))
        assert summary["positive_proposal_count"] == len(ROLE_ORDER)
        assert summary["negative_proposal_count"] > 0
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert scene.raster.shape == (320, 640)


def test_role_anchor_model_is_dynamic_and_permutation_equivariant() -> None:
    torch.manual_seed(2301)
    model = RoleAnchorSetNet().eval()
    values = torch.randn(1, 11, FEATURE_COUNT)
    order = torch.tensor([7, 1, 9, 3, 0, 10, 6, 2, 8, 5, 4])
    with torch.inference_mode():
        expected = model(values)
        permuted = model(values.index_select(1, order))
    assert expected.shape == (1, 11, 2 + len(ROLE_ORDER))
    assert model(values[:, :3]).shape == (1, 3, 2 + len(ROLE_ORDER))
    assert torch.allclose(permuted, expected.index_select(1, order), atol=1e-6, rtol=1e-6)


def test_role_anchor_model_exports_with_dynamic_cpu_parity(tmp_path: Path) -> None:
    model = RoleAnchorSetNet().eval()
    example = torch.linspace(-1.0, 1.0, 11 * FEATURE_COUNT).reshape(1, 11, FEATURE_COUNT)
    path = tmp_path / "role-anchor-set-v23.onnx"
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["proposal_evidence"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    for count in (3, 11):
        values = example[:, :count].numpy().astype(np.float32)
        with torch.inference_mode():
            expected = model(torch.from_numpy(values)).numpy()
        actual = np.asarray(
            session.run(None, {session.get_inputs()[0].name: values})[0],
            dtype=np.float32,
        )
        assert actual.shape == (1, count, 2 + len(ROLE_ORDER))
        assert float(np.max(np.abs(expected - actual))) <= 1e-5


def test_any_frozen_split_remains_fail_closed() -> None:
    seal_path = REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json"
    if not seal_path.exists():
        assert protocol_configuration()["fixture_identity_frozen"] is False
        return
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["optimizer_steps_at_freeze"] == 0
    assert seal["selection_evaluations"] == 0
    assert seal["public_evaluations"] == 0
    assert seal["training_authorized"] is False
    assert seal["public_execution_authorized"] is False
    assert seal["private_data"] is False
    assert seal["chandler_used"] is False
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
