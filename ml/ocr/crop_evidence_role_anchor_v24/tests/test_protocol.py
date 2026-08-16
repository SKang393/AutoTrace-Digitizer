# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.crop_evidence_role_anchor_v24.dataset import (
    encode_scene, proposal_summary, render_scene,
)
from ml.ocr.crop_evidence_role_anchor_v24.model import CropEvidenceRoleAnchorNet
from ml.ocr.crop_evidence_role_anchor_v24.model_p2 import (
    FrozenRoleAnchorCropResidualNet,
)
from ml.ocr.crop_evidence_role_anchor_v24.pipeline import proposal_crops
from ml.ocr.crop_evidence_role_anchor_v24.protocol import (
    CANDIDATE_LIMIT,
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    FEATURE_COUNT,
    ROLE_ORDER,
    V23_RESULT_PATH,
    V23_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.margin_calibrator_v20.pipeline import ProposalRecord
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    CONFIG_PATH as P1_CONFIG_PATH,
    _balanced_class_weights,
    _proposal_objective,
)
from ml.ocr.crop_evidence_role_anchor_v24.train_p2 import (
    CONFIG_PATH as P2_CONFIG_PATH,
    P1_RESULT_PATH,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _proposal_residual_objective,
    preflight as p2_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_fresh_bounded_and_fail_closed() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (REPO_ROOT / "ml/ocr/crop_evidence_role_anchor_v24/PROTOCOL.json").read_text(
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


def test_v23_trigger_is_exact_aggregate_only_terminal_record() -> None:
    path = REPO_ROOT / V23_RESULT_PATH
    assert sha256_file(path) == V23_RESULT_SHA256
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["candidate_id"] == "P3"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["case_level_details_emitted"] is False
    assert result["selection_metrics"]["false_positives"] == 3
    assert result["selection_metrics"]["false_negatives"] == 0
    assert result["selection_metrics"]["prohibited_structure_hits"] == 3
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert "cases" not in result and "predictions" not in result


def test_canonical_budget_consumes_p1_and_binds_one_unused_p2_authorization() -> None:
    ledger = json.loads(
        (
            REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item["task"] == "ocr-detection-recognition"
        and item["revision"] == "graph-text-crop-evidence-role-anchor-v24"
    )
    protocol_path = REPO_ROOT / "ml/ocr/crop_evidence_role_anchor_v24/PROTOCOL.json"
    p1_config_path = REPO_ROOT / P1_CONFIG_PATH
    p2_config_path = REPO_ROOT / P2_CONFIG_PATH
    seal_path = REPO_ROOT / "ml/ocr/crop_evidence_role_anchor_v24/SPLIT_SEAL.json"
    assert entry["status"] == "candidate_2_preregistered"
    assert entry["protocol_sha256"] == sha256_file(protocol_path)
    assert entry["preregistered_candidate_ids"] == ["P2"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P3"]
    assert entry["split_materialized"] is True
    assert entry["split_seal_sha256"] == sha256_file(seal_path)
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(p1_config_path)
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(p2_config_path)
    assert entry["p1_result_sha256"] == sha256_file(REPO_ROOT / P1_RESULT_PATH)
    assert entry["p2_expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS,
    )
    assert entry["selection_evaluations"] == 1
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P2"
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


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


def test_renderer_produces_fresh_complete_proposal_scenes() -> None:
    scenes = tuple(render_scene(split, 0) for split in ("train", "validation", "sealed_public"))
    for scene in scenes:
        summary = proposal_summary((scene,))
        assert summary["positive_proposal_count"] == len(ROLE_ORDER)
        assert summary["negative_proposal_count"] > 0
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert scene.raster.shape == (320, 640)
        assert scene.scene_id.startswith("crop-evidence-role-anchor-v24-")


def test_crop_stream_uses_exact_production_candidate_indices() -> None:
    scene = render_scene("train", 1)
    encoded, candidates, _, _ = encode_scene(scene)
    indices = (0, len(candidates) // 2, len(candidates) - 1)
    records = tuple(
        ProposalRecord(0, index, -1, "", ROLE_ORDER[0]) for index in indices
    )
    actual = proposal_crops((scene,), records)
    expected = encoded[np.asarray(indices), :, :, :CROP_WIDTH]
    assert actual.shape == (len(indices), CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH)
    assert np.array_equal(actual, expected)


def test_crop_evidence_model_is_dynamic_and_permutation_equivariant() -> None:
    torch.manual_seed(2401)
    model = CropEvidenceRoleAnchorNet().eval()
    evidence = torch.randn(1, 7, FEATURE_COUNT)
    crops = torch.randn(1, 7, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH)
    order = torch.tensor([6, 1, 4, 0, 5, 3, 2])
    with torch.inference_mode():
        expected = model(evidence, crops)
        permuted = model(evidence.index_select(1, order), crops.index_select(1, order))
    assert expected.shape == (1, 7, 2 + len(ROLE_ORDER))
    assert model(evidence[:, :3], crops[:, :3]).shape == (1, 3, 2 + len(ROLE_ORDER))
    assert torch.allclose(permuted, expected.index_select(1, order), atol=1e-6, rtol=1e-6)


def test_crop_evidence_model_exports_with_dynamic_cpu_parity(tmp_path: Path) -> None:
    model = CropEvidenceRoleAnchorNet().eval()
    evidence = torch.linspace(-1.0, 1.0, 5 * FEATURE_COUNT).reshape(1, 5, FEATURE_COUNT)
    crops = torch.linspace(
        -1.0, 1.0, 5 * CROP_CHANNELS * CROP_HEIGHT * CROP_WIDTH,
    ).reshape(1, 5, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH)
    path = tmp_path / "crop-evidence-role-anchor-v24.onnx"
    torch.onnx.export(
        model,
        (evidence, crops),
        path,
        input_names=["proposal_evidence", "proposal_crops"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    for count in (3, 5):
        evidence_values = evidence[:, :count].numpy().astype(np.float32)
        crop_values = crops[:, :count].numpy().astype(np.float32)
        with torch.inference_mode():
            expected = model(
                torch.from_numpy(evidence_values), torch.from_numpy(crop_values),
            ).numpy()
        actual = np.asarray(session.run(None, {
            "proposal_evidence": evidence_values,
            "proposal_crops": crop_values,
        })[0], dtype=np.float32)
        assert actual.shape == (1, count, 2 + len(ROLE_ORDER))
        assert float(np.max(np.abs(expected - actual))) <= 1e-5


def test_p1_balanced_losses_and_scene_extrema_margins_are_bound() -> None:
    config = json.loads((REPO_ROOT / P1_CONFIG_PATH).read_text(encoding="utf-8"))
    targets = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    assert torch.allclose(
        _balanced_class_weights(targets, 2, "proposal"), torch.tensor([0.5, 1.5]),
    )
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


def test_p2_zero_residual_preserves_every_parent_output_and_freezes_backbone() -> None:
    torch.manual_seed(2402)
    model = FrozenRoleAnchorCropResidualNet().eval()
    evidence = torch.randn(1, 6, FEATURE_COUNT)
    crops = torch.randn(1, 6, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH)
    with torch.inference_mode():
        parent = model.backbone(evidence)
        actual = model(evidence, crops)
    assert torch.equal(actual, parent)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.trainable_parameters()


def test_p2_model_exports_dynamic_cpu_with_parent_roles_preserved(tmp_path: Path) -> None:
    model = FrozenRoleAnchorCropResidualNet().eval()
    evidence = torch.linspace(-1.0, 1.0, 5 * FEATURE_COUNT).reshape(
        1, 5, FEATURE_COUNT,
    )
    crops = torch.linspace(
        -1.0, 1.0, 5 * CROP_CHANNELS * CROP_HEIGHT * CROP_WIDTH,
    ).reshape(1, 5, CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH)
    path = tmp_path / "frozen-role-anchor-crop-residual-v24.onnx"
    torch.onnx.export(
        model,
        (evidence, crops),
        path,
        input_names=["proposal_evidence", "proposal_crops"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for count in (3, 5):
        evidence_values = evidence[:, :count].numpy().astype(np.float32)
        crop_values = crops[:, :count].numpy().astype(np.float32)
        with torch.inference_mode():
            parent = model.backbone(torch.from_numpy(evidence_values)).numpy()
            expected = model(
                torch.from_numpy(evidence_values), torch.from_numpy(crop_values),
            ).numpy()
        actual = np.asarray(session.run(None, {
            "proposal_evidence": evidence_values,
            "proposal_crops": crop_values,
        })[0], dtype=np.float32)
        assert np.array_equal(expected[:, :, 2:], parent[:, :, 2:])
        assert actual.shape == (1, count, 2 + len(ROLE_ORDER))
        assert float(np.max(np.abs(expected - actual))) <= 1e-5


def test_p2_teacher_residual_objective_penalizes_positive_drop_and_negative_regression() -> None:
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    teacher = torch.tensor([[2.0, -2.0], [1.0, -1.0], [-2.0, 2.0], [-1.0, 1.0]])
    passing = teacher.clone()
    passing[:2, 0] += 1.0
    bad_positive = passing.clone()
    bad_positive[3] = torch.tensor([2.0, -2.0])
    bad_negative = passing.clone()
    bad_negative[1] = torch.tensor([-2.0, 2.0])
    _, positive, negative, _ = _proposal_residual_objective(
        passing, teacher, targets, torch.ones(2), config,
    )
    _, bad_positive_margin, _, _ = _proposal_residual_objective(
        bad_positive, teacher, targets, torch.ones(2), config,
    )
    _, _, bad_negative_margin, _ = _proposal_residual_objective(
        bad_negative, teacher, targets, torch.ones(2), config,
    )
    assert float(positive) == 0.0
    assert float(negative) == 0.0
    assert float(bad_positive_margin) > 0.0
    assert float(bad_negative_margin) > 0.0


def test_v24_p1_result_and_p2_preregistration_remain_fail_closed() -> None:
    root = REPO_ROOT / "ml/ocr/crop_evidence_role_anchor_v24"
    seal = json.loads((root / "SPLIT_SEAL.json").read_text(encoding="utf-8"))
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    p1 = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
    assert seal["optimizer_steps_at_freeze"] == 0
    assert seal["selection_evaluations"] == 0
    assert seal["public_evaluations"] == 0
    assert seal["cross_split_source_overlap_counts"] == {
        "train_validation": 0,
        "train_sealed_public": 0,
        "validation_sealed_public": 0,
    }
    assert config["split_seal_sha256"] == sha256_file(root / "SPLIT_SEAL.json")
    assert p1["candidate_consumed"] is True
    assert p1["status"] == "failed_selection"
    assert p1["selection_metrics"]["false_positives"] == 0
    assert p1["selection_metrics"]["false_negatives"] == 22
    assert p1["selection_metrics"]["per_role_accuracy"]["PhaseHeading"] == 0.421875
    assert p1["onnx_parity_passed"] is False
    assert p1["public_gate_archive_opened"] is False
    assert p1["case_level_details_emitted"] is False
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS,
    )
    evidence = p2_preflight()
    assert evidence["seal"] == seal
    assert not (root / "artifacts/P2-run").exists()
    assert sha256_file(REPO_ROOT / "artifacts/production-validation/ocr-v24-train.zip") == config[
        "train_fixture_archive_sha256"
    ]
    assert sha256_file(REPO_ROOT / "artifacts/production-validation/ocr-v24-selection.zip") == config[
        "selection_fixture_archive_sha256"
    ]
    assert sha256_file(REPO_ROOT / "artifacts/production-validation/ocr-v24-public.zip") == config[
        "public_fixture_archive_sha256"
    ]
    assert config["public_execution_authorized"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["marker_creation_evaluated"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False
