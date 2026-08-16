# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import sha256_file
from ml.ocr.scene_evidence_attention_v22.dataset import proposal_summary, render_scene
from ml.ocr.scene_evidence_attention_v22.model import SceneEvidenceAttentionNet
from ml.ocr.scene_evidence_attention_v22.protocol import (
    CANDIDATE_LIMIT,
    FEATURE_COUNT,
    ROLE_ORDER,
    V20_RESULT_PATH,
    V20_RESULT_SHA256,
    V21_RESULT_PATH,
    V21_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.scene_evidence_attention_v22.train_p1 import (
    CANONICAL_OUTPUT,
    CONFIG_PATH,
    RUNNER_SOURCE_PATHS,
    _calibrated_records,
    _feature_groups,
    preflight,
)
from ml.ocr.scene_evidence_attention_v22.train_p2 import (
    CONFIG_PATH as P2_CONFIG_PATH,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _proposal_objective,
    preflight as p2_preflight,
)
from ml.ocr.margin_calibrator_v20.pipeline import ProposalRecord
from ml.markers.gate_seal import source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_fresh_bounded_and_fail_closed() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/PROTOCOL.json").read_text(
            encoding="utf-8"
        )
    )
    assert tracked == json.loads(json.dumps(protocol))
    assert protocol["candidate_budget"]["candidate_limit"] == CANDIDATE_LIMIT == 3
    assert protocol["candidate_budget"]["optimizer_steps_maximum"] == 1280
    assert protocol["candidate_budget"]["public_execution_limit"] == 1
    assert protocol["training_authorized"] is False
    assert protocol["public_execution_authorized"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["private_validation_authorized"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert "Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]


def test_trigger_results_are_exact_aggregate_only_records() -> None:
    assert sha256_file(REPO_ROOT / V20_RESULT_PATH) == V20_RESULT_SHA256
    assert sha256_file(REPO_ROOT / V21_RESULT_PATH) == V21_RESULT_SHA256
    v20 = json.loads((REPO_ROOT / V20_RESULT_PATH).read_text(encoding="utf-8"))
    v21 = json.loads((REPO_ROOT / V21_RESULT_PATH).read_text(encoding="utf-8"))
    assert v20["case_level_details_emitted"] is False
    assert v20["public_gate_evaluations"] == 0
    assert "cases" not in v21 and "predictions" not in v21
    assert v21["public_archive_opened"] is False
    assert v21["public_evaluation_count"] == 0


def test_split_families_and_seed_offsets_are_disjoint() -> None:
    registrations = [split_registration(name) for name in ("train", "validation", "sealed_public")]
    assert len({item.seed_offset for item in registrations}) == 3
    renderer_sets = [set(item.renderer_families) for item in registrations]
    degradation_sets = [set(item.degradation_families) for item in registrations]
    assert not any(renderer_sets[left] & renderer_sets[right] for left in range(3) for right in range(left + 1, 3))
    assert not any(degradation_sets[left] & degradation_sets[right] for left in range(3) for right in range(left + 1, 3))


def test_renderer_produces_one_proposal_for_every_role_truth() -> None:
    scenes = tuple(render_scene(split, 0) for split in ("train", "validation", "sealed_public"))
    for scene in scenes:
        summary = proposal_summary((scene,))
        assert summary["positive_proposal_count"] == len(ROLE_ORDER)
        assert summary["negative_proposal_count"] > 0
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert scene.raster.shape == (320, 640)


def test_attention_model_is_dynamic_and_permutation_equivariant() -> None:
    torch.manual_seed(2201)
    model = SceneEvidenceAttentionNet().eval()
    values = torch.randn(1, 11, FEATURE_COUNT)
    order = torch.tensor([7, 1, 9, 3, 0, 10, 6, 2, 8, 5, 4])
    with torch.inference_mode():
        expected = model(values)
        permuted = model(values.index_select(1, order))
    assert expected.shape == (1, 11, 2 + len(ROLE_ORDER))
    assert model(values[:, :3]).shape == (1, 3, 2 + len(ROLE_ORDER))
    assert torch.allclose(permuted, expected.index_select(1, order), atol=1e-6, rtol=1e-6)


def test_attention_model_exports_with_dynamic_cpu_parity(tmp_path: Path) -> None:
    model = SceneEvidenceAttentionNet().eval()
    example = torch.linspace(-1.0, 1.0, 11 * FEATURE_COUNT).reshape(1, 11, FEATURE_COUNT)
    path = tmp_path / "scene-evidence-attention-v22.onnx"
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


def test_p1_result_records_consumed_failed_selection_without_public_execution() -> None:
    seal_path = REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/SPLIT_SEAL.json"
    if seal_path.exists():
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
    result = json.loads(
        (
            REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/P1_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert result["candidate_id"] == "P1"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["selection_gate_passed"] is False
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["marker_creation_evaluated"] is False
    assert result["private_or_article_images"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False


def test_p1_config_binds_runner_fixtures_and_fail_closed_outputs() -> None:
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/SPLIT_SEAL.json"
    assert config["expected_optimizer_steps"] == 1280
    assert config["proposal_selection"] == "all_frozen_production_proposals_no_detector_prefilter"
    assert config["complete_proposal_negative_cap_per_scene"] == 100000
    assert config["detector_prefilter_applied"] is False
    assert config["selection_evaluation_limit"] == 1
    assert config["public_execution_authorized"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["marker_creation_evaluated"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False
    assert config["split_seal_sha256"] == sha256_file(seal_path)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    )
    result = json.loads(
        (
            REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/P1_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        report = json.loads(
            (REPO_ROOT / CANONICAL_OUTPUT / "candidate-report.json").read_text(
                encoding="utf-8"
            )
        )
        assert sha256_file(REPO_ROOT / CANONICAL_OUTPUT / "candidate-report.json") == result["report_sha256"]
        assert report["status"] == result["status"]
        assert report["selected_threshold"] == result["selected_threshold"]
        assert report["selection_metrics"]["false_positives"] == 8
        assert report["selection_metrics"]["prohibited_structure_hits"] == 8
        assert report["public_gate_archive_opened"] is False
        assert report["public_gate_evaluations"] == 0
        assert report["production_approval"] is False
        assert report["release_eligible"] is False
    else:
        evidence = preflight()
        assert evidence["config"] == config
        assert evidence["seal"]["public_evaluations"] == 0


def test_canonical_budget_authorizes_only_preregistered_p2_after_consumed_p1() -> None:
    ledger = json.loads(
        (
            REPO_ROOT
            / "ml/markers/training-budgets/production-repair-v1.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item["task"] == "ocr-detection-recognition"
        and item["revision"] == "graph-text-scene-evidence-attention-v22"
    )
    config_path = REPO_ROOT / CONFIG_PATH
    result_path = REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/P1_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    p2_config_path = REPO_ROOT / P2_CONFIG_PATH
    assert entry["status"] == "candidate_2_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P2"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P3"]
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P2"
    assert entry["execution_blocker"] is None
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(p2_config_path)
    assert entry["p1_expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    )
    assert entry["p1_result_sha256"] == sha256_file(result_path)
    assert entry["p1_candidate_report_sha256"] == result["report_sha256"]
    assert entry["p1_checkpoint_sha256"] == result["checkpoint_sha256"]
    assert entry["p1_onnx_sha256"] == result["onnx_sha256"]
    assert entry["p1_training_opened_seal_sha256"] == result["training_opened_seal_sha256"]
    assert entry["p1_training_result_seal_sha256"] == result["training_result_seal_sha256"]
    assert entry["p2_expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS,
    )
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_p2_config_binds_aggregate_trigger_margin_objective_and_locked_public_gate() -> None:
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    assert config["candidate_id"] == "P2"
    assert config["objective"] == "class_balanced_cross_entropy_plus_scene_extrema_acceptance_margin_v1"
    assert config["negative_acceptance_probability_maximum"] == 0.10
    assert config["positive_acceptance_probability_minimum"] == 0.90
    assert config["expected_optimizer_steps"] == 1280
    assert config["p1_result_sha256"] == sha256_file(
        REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/P1_RESULT.json"
    )
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS,
    )
    assert config["public_execution_authorized"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["production_approval"] is False
    assert config["release_eligible"] is False
    evidence = p2_preflight()
    assert evidence["config"] == config
    assert evidence["p1_result"]["status"] == "failed_selection"


def test_p2_scene_extrema_objective_penalizes_worst_negative_and_preserves_truth() -> None:
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    weights = torch.ones(2, dtype=torch.float32)
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    passing = torch.tensor([[2.0, -2.0], [1.5, -1.5], [-2.0, 2.0], [-1.5, 1.5]])
    failing_negative = passing.clone()
    failing_negative[1] = torch.tensor([-2.0, 2.0])
    failing_positive = passing.clone()
    failing_positive[3] = torch.tensor([2.0, -2.0])
    _, passing_negative, passing_positive = _proposal_objective(
        passing, targets, weights, config,
    )
    _, bad_negative, _ = _proposal_objective(failing_negative, targets, weights, config)
    _, _, bad_positive = _proposal_objective(failing_positive, targets, weights, config)
    assert float(passing_negative) == 0.0
    assert float(passing_positive) == 0.0
    assert float(bad_negative) > 0.0
    assert float(bad_positive) > 0.0


def test_scene_grouping_and_calibrated_role_projection_are_exact() -> None:
    records = (
        ProposalRecord(0, 0, 0, "1", "Other"),
        ProposalRecord(0, 1, -1, "", "Other"),
        ProposalRecord(1, 0, 0, "2", "Other"),
    )
    groups = _feature_groups(records, 2)
    assert [group.tolist() for group in groups] == [[0, 1], [2]]
    output = np.zeros((3, 2 + len(ROLE_ORDER)), dtype=np.float32)
    output[0, 2 + ROLE_ORDER.index("XTick")] = 3.0
    output[1, 2 + ROLE_ORDER.index("Other")] = 2.0
    output[2, 2 + ROLE_ORDER.index("YTick")] = 4.0
    projected = _calibrated_records(records, output)
    assert [record.predicted_role for record in projected] == ["XTick", "Other", "YTick"]
