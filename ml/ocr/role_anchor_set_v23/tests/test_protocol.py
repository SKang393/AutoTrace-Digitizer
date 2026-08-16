# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
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
from ml.ocr.role_anchor_set_v23.train_p2 import (
    CANONICAL_OUTPUT as P2_CANONICAL_OUTPUT,
    CONFIG_PATH as P2_CONFIG_PATH,
    P1_RESULT_PATH,
    P1_RESULT_SHA256,
    PROTOCOL_PATH as P2_PROTOCOL_PATH,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _proposal_objective as _p2_proposal_objective,
    preflight as p2_preflight,
)
from ml.ocr.role_anchor_set_v23.train_p3 import (
    CANONICAL_OUTPUT as P3_CANONICAL_OUTPUT,
    CONFIG_PATH as P3_CONFIG_PATH,
    P2_RESULT_PATH,
    P2_RESULT_SHA256,
    PROTOCOL_PATH as P3_PROTOCOL_PATH,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
    TRAINABLE_PARAMETER_NAMES,
    _proposal_head_objective,
    preflight as p3_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
P3_RESULT_PATH = Path("ml/ocr/role_anchor_set_v23/P3_RESULT.json")
P3_RESULT_SHA256 = "83d7a3be46e082be3550144cb4bb1b0a287ada29fadbdcca231d2e27d7ad7422"


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


def test_canonical_budget_records_exhausted_p1_p2_p3_and_locks_execution() -> None:
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
    p2_config_path = REPO_ROOT / P2_CONFIG_PATH
    p3_config_path = REPO_ROOT / P3_CONFIG_PATH
    seal_path = REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json"
    assert entry["status"] == "exhausted_selection_failed"
    assert entry["protocol_sha256"] == sha256_file(protocol_path)
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["split_materialized"] is True
    assert entry["split_seal_sha256"] == sha256_file(seal_path)
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(p2_config_path)
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(p3_config_path)
    assert entry["p1_expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    )
    assert entry["p1_result_sha256"] == P1_RESULT_SHA256
    assert entry["p2_protocol_sha256"] == sha256_file(REPO_ROOT / P2_PROTOCOL_PATH)
    assert entry["p2_expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS,
    )
    assert entry["p2_result_sha256"] == P2_RESULT_SHA256
    assert entry["p3_protocol_sha256"] == sha256_file(REPO_ROOT / P3_PROTOCOL_PATH)
    assert entry["p3_expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P3_RUNNER_SOURCE_PATHS,
    )
    assert entry["p3_result_sha256"] == P3_RESULT_SHA256
    assert entry["selection_evaluations"] == 3
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
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


def test_p1_preflight_refuses_consumed_output() -> None:
    with pytest.raises(RuntimeError, match="P1 output already exists"):
        preflight()


def test_p1_terminal_result_binds_report_payload_and_single_use_seals() -> None:
    result_path = REPO_ROOT / P1_RESULT_PATH
    assert sha256_file(result_path) == P1_RESULT_SHA256
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["candidate_id"] == "P1"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["selection_gate_passed"] is False
    assert result["case_level_details_emitted"] is False
    assert result["selection_metrics"]["direct_stored_fixture_byte_execution"] is True
    assert result["selection_metrics"]["false_positives"] == 3
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["prohibited_structure_hits"] == 3
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["marker_creation_evaluated"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
    assert "cases" not in result and "predictions" not in result

    report_path = REPO_ROOT / "ml/ocr/role_anchor_set_v23/artifacts/P1-run/candidate-report.json"
    if report_path.is_file():
        assert sha256_file(report_path) == result["report_sha256"]
    seal_root = (
        REPO_ROOT
        / "ml/markers/training-seals/ocr-detection-recognition"
        / "graph-text-role-anchor-set-v23/P1"
    )
    assert sha256_file(seal_root / "opened.json") == result["training_opened_seal_sha256"]
    assert sha256_file(seal_root / "result.json") == result["training_result_seal_sha256"]


def test_p2_protocol_uses_only_aggregate_p1_trigger_and_remains_fail_closed() -> None:
    protocol = json.loads((REPO_ROOT / P2_PROTOCOL_PATH).read_text(encoding="utf-8"))
    trigger = protocol["trigger_aggregate"]
    assert trigger["p1_result_path"] == P1_RESULT_PATH.as_posix()
    assert trigger["p1_result_sha256"] == P1_RESULT_SHA256
    assert trigger["p1_false_positives"] == 3
    assert trigger["p1_false_negatives"] == 0
    assert trigger["p1_prohibited_structure_hits"] == 3
    assert trigger["validation_case_identity_or_pixels_used"] is False
    assert protocol["case_level_evidence_used_for_design"] is False
    assert protocol["validation_or_public_pixels_used_for_design"] is False
    assert protocol["public_execution_authorized"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert "cases" not in protocol and "predictions" not in protocol


def test_p2_config_binds_protocol_runner_fixtures_and_p1_result() -> None:
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    seal = json.loads(
        (REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["candidate_id"] == "P2"
    assert config["p1_result_sha256"] == P1_RESULT_SHA256
    assert config["protocol_sha256"] == sha256_file(REPO_ROOT / P2_PROTOCOL_PATH)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS,
    )
    assert config["split_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json"
    )
    assert config["train_fixture_archive_sha256"] == seal["splits"]["train"]["archive_sha256"]
    assert config["selection_fixture_archive_sha256"] == seal["splits"]["validation"]["archive_sha256"]
    assert config["public_fixture_archive_sha256"] == seal["splits"]["sealed_public"]["archive_sha256"]
    assert config["selection_evaluation_limit"] == 1
    assert config["public_execution_authorized"] is False
    assert config["marker_creation_evaluated"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False


def test_p2_preflight_refuses_consumed_output() -> None:
    assert (REPO_ROOT / P2_CANONICAL_OUTPUT).exists()
    with pytest.raises(RuntimeError, match="P2 output already exists"):
        p2_preflight()


def test_p2_terminal_result_binds_report_payload_and_single_use_seals() -> None:
    result_path = REPO_ROOT / P2_RESULT_PATH
    assert sha256_file(result_path) == P2_RESULT_SHA256
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["candidate_id"] == "P2"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["selection_gate_passed"] is False
    assert result["case_level_details_emitted"] is False
    assert result["selection_metrics"]["direct_stored_fixture_byte_execution"] is True
    assert result["selection_metrics"]["false_positives"] == 2
    assert result["selection_metrics"]["false_negatives"] == 23
    assert result["selection_metrics"]["prohibited_structure_hits"] == 2
    assert result["selection_metrics"]["role_accuracy"] == 0.87890625
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["marker_creation_evaluated"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
    assert "cases" not in result and "predictions" not in result

    report_path = REPO_ROOT / "ml/ocr/role_anchor_set_v23/artifacts/P2-run/candidate-report.json"
    assert sha256_file(report_path) == result["report_sha256"]
    seal_root = (
        REPO_ROOT
        / "ml/markers/training-seals/ocr-detection-recognition"
        / "graph-text-role-anchor-set-v23/P2"
    )
    assert sha256_file(seal_root / "opened.json") == result["training_opened_seal_sha256"]
    assert sha256_file(seal_root / "result.json") == result["training_result_seal_sha256"]


def test_p2_tightens_only_preregistered_worst_negative_margin() -> None:
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    assert config["negative_acceptance_probability_maximum"] == 0.01
    assert config["negative_scene_extrema_margin_weight"] == 4.0
    assert config["positive_acceptance_probability_minimum"] == 0.9
    assert config["positive_scene_extrema_margin_weight"] == 0.25
    assert config["role_loss_weight"] == 0.75
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    passing = torch.tensor([[3.0, -3.0], [2.5, -2.5], [-3.0, 3.0], [-2.5, 2.5]])
    failing_negative = passing.clone()
    failing_negative[1] = torch.tensor([-3.0, 3.0])
    failing_positive = passing.clone()
    failing_positive[3] = torch.tensor([3.0, -3.0])
    weights = torch.ones(2)
    _, passing_negative, passing_positive = _p2_proposal_objective(
        passing, targets, weights, config,
    )
    _, bad_negative, _ = _p2_proposal_objective(
        failing_negative, targets, weights, config,
    )
    _, _, bad_positive = _p2_proposal_objective(
        failing_positive, targets, weights, config,
    )
    assert float(passing_negative) == 0.0
    assert float(passing_positive) == 0.0
    assert float(bad_negative) > 0.0
    assert float(bad_positive) > 0.0


def test_p3_protocol_uses_only_aggregate_p1_p2_triggers_and_remains_fail_closed() -> None:
    protocol = json.loads((REPO_ROOT / P3_PROTOCOL_PATH).read_text(encoding="utf-8"))
    trigger = protocol["trigger_aggregate"]
    assert trigger["p1_result_sha256"] == P1_RESULT_SHA256
    assert trigger["p2_result_sha256"] == P2_RESULT_SHA256
    assert trigger["p1_false_negatives"] == 0
    assert trigger["p1_false_positives"] == 3
    assert trigger["p2_false_negatives"] == 23
    assert trigger["p2_false_positives"] == 2
    assert trigger["validation_case_identity_or_pixels_used"] is False
    assert protocol["case_level_evidence_used_for_design"] is False
    assert protocol["validation_or_public_pixels_used_for_design"] is False
    assert protocol["training"]["proposal_head_only"] is True
    assert protocol["public_execution_authorized"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert "cases" not in protocol and "predictions" not in protocol


def test_p3_config_binds_parent_protocol_runner_fixtures_and_results() -> None:
    config = json.loads((REPO_ROOT / P3_CONFIG_PATH).read_text(encoding="utf-8"))
    seal = json.loads(
        (REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["candidate_id"] == "P3"
    assert config["p1_result_sha256"] == P1_RESULT_SHA256
    assert config["p2_result_sha256"] == P2_RESULT_SHA256
    assert config["protocol_sha256"] == sha256_file(REPO_ROOT / P3_PROTOCOL_PATH)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P3_RUNNER_SOURCE_PATHS,
    )
    assert config["trainable_parameter_names"] == list(TRAINABLE_PARAMETER_NAMES)
    assert config["split_seal_sha256"] == sha256_file(
        REPO_ROOT / "ml/ocr/role_anchor_set_v23/SPLIT_SEAL.json"
    )
    assert config["train_fixture_archive_sha256"] == seal["splits"]["train"]["archive_sha256"]
    assert config["selection_fixture_archive_sha256"] == seal["splits"]["validation"]["archive_sha256"]
    assert config["public_fixture_archive_sha256"] == seal["splits"]["sealed_public"]["archive_sha256"]
    assert config["expected_optimizer_steps"] == 512
    assert config["selection_evaluation_limit"] == 1
    assert config["public_execution_authorized"] is False
    assert config["marker_creation_evaluated"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False


def test_p3_preflight_refuses_consumed_output() -> None:
    assert (REPO_ROOT / P3_CANONICAL_OUTPUT).exists()
    with pytest.raises(RuntimeError, match="P3 output already exists"):
        p3_preflight()


def test_p3_terminal_result_binds_report_payload_and_single_use_seals() -> None:
    result_path = REPO_ROOT / P3_RESULT_PATH
    assert sha256_file(result_path) == P3_RESULT_SHA256
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["candidate_id"] == "P3"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["selection_gate_passed"] is False
    assert result["selection_metrics"]["direct_stored_fixture_byte_execution"] is True
    assert result["selection_metrics"]["false_positives"] == 3
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["duplicate_region_count"] == 0
    assert result["selection_metrics"]["prohibited_structure_hits"] == 3
    assert result["p1_teacher_role_maximum_absolute_error"] == 0.0
    assert result["p1_teacher_role_preserved"] is True
    assert (
        result["frozen_parameter_stream_sha256_before"]
        == result["frozen_parameter_stream_sha256_after"]
    )
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["marker_creation_evaluated"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
    assert "cases" not in result and "predictions" not in result

    report_path = REPO_ROOT / "ml/ocr/role_anchor_set_v23/artifacts/P3-run/candidate-report.json"
    assert sha256_file(report_path) == result["report_sha256"]
    seal_root = (
        REPO_ROOT
        / "ml/markers/training-seals/ocr-detection-recognition"
        / "graph-text-role-anchor-set-v23/P3"
    )
    assert sha256_file(seal_root / "opened.json") == result["training_opened_seal_sha256"]
    assert sha256_file(seal_root / "result.json") == result["training_result_seal_sha256"]


def test_p3_teacher_signed_margin_and_scene_separation_are_fixed() -> None:
    config = json.loads((REPO_ROOT / P3_CONFIG_PATH).read_text(encoding="utf-8"))
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    teacher = torch.tensor([[0.0, -1.0], [0.0, -0.5], [0.0, 1.0], [0.0, 2.0]])
    passing = torch.tensor([[0.0, -2.0], [0.0, -1.5], [0.0, 1.0], [0.0, 2.0]])
    failing_positive = passing.clone()
    failing_positive[2, 1] = 0.5
    failing_negative = passing.clone()
    failing_negative[1, 1] = 0.5
    _, positive, negative, separation = _proposal_head_objective(
        passing, teacher, targets, torch.ones(2), config,
    )
    _, bad_positive, _, _ = _proposal_head_objective(
        failing_positive, teacher, targets, torch.ones(2), config,
    )
    _, _, bad_negative, bad_separation = _proposal_head_objective(
        failing_negative, teacher, targets, torch.ones(2), config,
    )
    assert float(positive) == 0.0
    assert float(negative) == 0.0
    assert float(separation) == 0.0
    assert float(bad_positive) > 0.0
    assert float(bad_negative) > 0.0
    assert float(bad_separation) > 0.0


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
