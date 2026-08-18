# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration tests for OCR V27."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.morphology_veto_proposal_v27.dataset import proposal_summary, render_scene
from ml.ocr.morphology_veto_proposal_v27.features import structure_features
from ml.ocr.morphology_veto_proposal_v27.model import FrozenV26MorphologyVetoNet
from ml.ocr.morphology_veto_proposal_v27.model_p2 import (
    FrozenP1FinalVetoProposalNet,
)
from ml.ocr.morphology_veto_proposal_v27.prepare_split import (
    ARCHIVE_PATHS,
    SOURCE_PATHS,
)
from ml.ocr.morphology_veto_proposal_v27.protocol import (
    CANDIDATE_LIMIT,
    FEATURE_COUNT,
    STRUCTURE_FEATURE_COUNT,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.scene_topology_proposal_v26.dataset import render_scene as render_v26_scene
from ml.ocr.morphology_veto_proposal_v27.train_p1 import (
    _proposal_objective,
)
from ml.ocr.morphology_veto_proposal_v27.train_p2 import (
    CONFIG_PATH as P2_CONFIG_PATH,
    P1_CHECKPOINT_PATH,
    P1_CHECKPOINT_SHA256,
    P1_ONNX_PATH,
    P1_ONNX_SHA256,
    P1_REPORT_PATH,
    P1_REPORT_SHA256,
    P1_RESULT_PATH,
    P1_RESULT_SHA256,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    _load_p1_state,
    _objective as p2_objective,
    preflight as p2_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _png_sha256(raster: np.ndarray) -> str:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(
        stream, format="PNG", optimize=False, compress_level=9,
    )
    return sha256(stream.getvalue()).hexdigest()


def test_protocol_is_fresh_bounded_and_fail_closed() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (
            REPO_ROOT
            / "ml/ocr/morphology_veto_proposal_v27/PROTOCOL.json"
        ).read_text(encoding="utf-8")
    )
    assert tracked == json.loads(json.dumps(protocol))
    assert protocol["candidate_budget"]["candidate_limit"] == CANDIDATE_LIMIT == 3
    assert protocol["candidate_budget"]["optimizer_steps_maximum"] == 1024
    assert protocol["fixture_identity_frozen"] is False
    assert protocol["training_authorized"] is False
    assert protocol["public_execution_authorized"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["private_validation_authorized"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert protocol["architecture"]["runtime_numeric_precision"] == "float32"
    assert protocol["architecture"]["output_logit_scale"] == 0.5
    assert protocol["selection_gates"]["onnx_parity_maximum_absolute_error"] == 1e-5
    assert "Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]


def test_v26_trigger_is_exact_aggregate_only_terminal_record() -> None:
    path = REPO_ROOT / TRIGGER_RESULT_PATH
    assert sha256_file(path) == TRIGGER_RESULT_SHA256
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["selection_metrics"]
    assert result["candidate_id"] == "P3"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["case_level_details_emitted"] is False
    assert metrics["scene_count"] == 128
    assert metrics["exact_scene_count"] == 122
    assert metrics["true_positives"] == 1024
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == 0
    assert metrics["prohibited_structure_hits"] == 1
    assert result["onnx_parity_maximum_absolute_error"] > 1e-5
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert "cases" not in result and "predictions" not in result


def test_canonical_budget_records_p1_and_authorizes_only_checksum_bound_p2() -> None:
    ledger = json.loads(
        (
            REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item["revision"] == "graph-text-morphology-veto-proposal-v27"
    )
    assert entry["status"] == "candidate_2_preregistered"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P2"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P3"]
    assert entry["protocol_sha256"] == sha256_file(REPO_ROOT / entry["protocol_path"])
    assert entry["split_materialized"] is True
    assert entry["split_seal_sha256"] == sha256_file(
        REPO_ROOT / entry["split_seal_path"]
    )
    assert entry["selection_evaluations"] == 1
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P2"
    assert entry["execution_authorization"]
    assert entry["manifest_created"] is False
    assert entry["model_store_promoted"] is False
    assert entry["private_validation"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    for split, path in entry["split_archive_paths"].items():
        assert sha256_file(REPO_ROOT / path) == entry["split_archive_sha256"][split]
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(
        REPO_ROOT / entry["candidate_config_paths"]["P1"]
    )
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(
        REPO_ROOT / entry["candidate_config_paths"]["P2"]
    )
    assert entry["p1_result_sha256"] == sha256_file(REPO_ROOT / P1_RESULT_PATH)
    assert entry["p1_candidate_report_sha256"] == sha256_file(
        REPO_ROOT / P1_REPORT_PATH
    )
    assert entry["p1_checkpoint_sha256"] == sha256_file(
        REPO_ROOT / P1_CHECKPOINT_PATH
    )
    assert entry["p1_onnx_sha256"] == sha256_file(REPO_ROOT / P1_ONNX_PATH)
    assert entry["candidate_1_selection_metrics"]["passing_threshold_window"] == []
    assert entry["candidate_1_selection_metrics"]["false_positives"] == 3
    assert entry["candidate_1_selection_metrics"]["prohibited_structure_hits"] == 3


def test_fresh_split_families_offsets_and_sample_bytes_are_disjoint() -> None:
    registrations = [
        split_registration(name) for name in ("train", "validation", "sealed_public")
    ]
    renderer_families = [set(value.renderer_families) for value in registrations]
    degradation_families = [set(value.degradation_families) for value in registrations]
    assert len({value.seed_offset for value in registrations}) == 3
    assert all(
        not renderer_families[left] & renderer_families[right]
        for left in range(3) for right in range(left + 1, 3)
    )
    assert all(
        not degradation_families[left] & degradation_families[right]
        for left in range(3) for right in range(left + 1, 3)
    )
    v27_hashes: set[str] = set()
    v26_hashes: set[str] = set()
    for split in ("train", "validation", "sealed_public"):
        scenes = tuple(render_scene(split, index) for index in range(4))
        summary = proposal_summary(scenes)
        assert summary["positive_proposal_count"] == 4 * 8
        assert summary["role_truth_counts"] == {
            "Annotation": 4,
            "AxisTitle": 4,
            "LegendText": 4,
            "Other": 4,
            "Participant": 4,
            "PhaseHeading": 4,
            "XTick": 4,
            "YTick": 4,
        }
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert all(scene.scene_id.startswith("morphology-veto-v27-") for scene in scenes)
        v27_hashes.update(_png_sha256(scene.raster) for scene in scenes)
        v26_hashes.update(
            _png_sha256(render_v26_scene(split, index).raster) for index in range(4)
        )
    assert len(v27_hashes) == 12
    assert not v27_hashes & v26_hashes


def test_structure_features_are_deterministic_normalized_and_sensitive() -> None:
    crops = np.zeros((2, 2, 32, 128), dtype=np.float32)
    crops[0, 0, 15:17, 8:120] = 1.0
    crops[0, 1, 5:27, 62:66] = 1.0
    crops[1, :, 9:23, 39:89] = 0.8
    first = structure_features(crops)
    second = structure_features(crops.copy())
    assert first.shape == (2, STRUCTURE_FEATURE_COUNT)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.all(first >= 0.0) and np.all(first <= 1.0)
    assert not np.array_equal(first[0], first[1])


def test_scaled_model_preserves_parent_role_argmax_and_strict_cpu_parity(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2701)
    model = FrozenV26MorphologyVetoNet().eval()
    evidence = torch.linspace(-1.0, 1.0, 5 * FEATURE_COUNT).reshape(
        1, 5, FEATURE_COUNT
    )
    crops = torch.linspace(
        0.0, 1.0, 5 * 2 * 32 * 128
    ).reshape(1, 5, 2, 32, 128)
    structure = torch.linspace(
        0.0, 1.0, 5 * STRUCTURE_FEATURE_COUNT
    ).reshape(1, 5, STRUCTURE_FEATURE_COUNT)
    with torch.no_grad():
        expected = model(evidence, crops, structure)
        parent = model.parent(evidence, crops)
    assert torch.equal(
        expected[:, :, 2:].argmax(dim=2), parent[:, :, 2:].argmax(dim=2)
    )
    assert torch.equal(expected[:, :, :2], parent[:, :, :2] * 0.5)
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())
    assert model.trainable_parameters()

    path = tmp_path / "morphology-veto-v27.onnx"
    torch.onnx.export(
        model,
        (evidence, crops, structure),
        path,
        input_names=["proposal_evidence", "proposal_crops", "structure_features"],
        output_names=["proposal_and_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "structure_features": {1: "proposal_count"},
            "proposal_and_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None,
        {
            "proposal_evidence": evidence.numpy(),
            "proposal_crops": crops.numpy(),
            "structure_features": structure.numpy(),
        },
    )[0]
    assert actual.dtype == np.float32
    assert float(np.max(np.abs(expected.numpy() - actual))) <= 1e-5


def test_freeze_source_inventory_and_archives_are_v27_owned() -> None:
    assert set(ARCHIVE_PATHS) == {"train", "validation", "sealed_public"}
    assert all("ocr-v27-" in path.name for path in ARCHIVE_PATHS.values())
    paths = {path.as_posix() for path in SOURCE_PATHS}
    assert {
        "ml/ocr/morphology_veto_proposal_v27/PROTOCOL.json",
        "ml/ocr/morphology_veto_proposal_v27/prepare_split.py",
        "ml/ocr/morphology_veto_proposal_v27/sealed_gate.py",
        "ml/ocr/morphology_veto_proposal_v27/train_p1.py",
        "ml/ocr/scene_topology_proposal_v26/model_p3.py",
    }.issubset(paths)
    assert all((REPO_ROOT / path).is_file() for path in SOURCE_PATHS)


def test_asymmetric_objective_weights_negative_errors_more_heavily() -> None:
    config = protocol_configuration()["candidate_p1"]
    weights = torch.ones(2)
    targets = torch.tensor([0, 1])
    false_positive_logits = torch.tensor([[-4.0, 4.0], [-4.0, 4.0]])
    false_negative_logits = torch.tensor([[4.0, -4.0], [4.0, -4.0]])
    _, false_positive_components = _proposal_objective(
        false_positive_logits, targets, weights, config,
    )
    _, false_negative_components = _proposal_objective(
        false_negative_logits, targets, weights, config,
    )
    assert (
        false_positive_components["asymmetric_cross_entropy"]
        > false_negative_components["asymmetric_cross_entropy"]
    )
    assert config["false_positive_weight"] == 4.0


def test_frozen_split_and_p1_config_keep_public_truth_unopened() -> None:
    seal = json.loads(
        (REPO_ROOT / "ml/ocr/morphology_veto_proposal_v27/SPLIT_SEAL.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        (REPO_ROOT / "ml/ocr/morphology_veto_proposal_v27/training/p1.json").read_text(
            encoding="utf-8"
        )
    )
    assert seal["optimizer_steps_at_freeze"] == 0
    assert seal["selection_evaluations"] == 0
    assert seal["public_evaluations"] == 0
    assert seal["training_authorized"] is False
    assert seal["public_execution_authorized"] is False
    assert config["selection_evaluation_limit"] == 1
    assert config["public_execution_authorized"] is False


def test_p1_result_is_exact_aggregate_only_failed_selection_evidence() -> None:
    assert sha256_file(REPO_ROOT / P1_RESULT_PATH) == P1_RESULT_SHA256
    assert sha256_file(REPO_ROOT / P1_CHECKPOINT_PATH) == P1_CHECKPOINT_SHA256
    assert sha256_file(REPO_ROOT / P1_ONNX_PATH) == P1_ONNX_SHA256
    assert sha256_file(REPO_ROOT / P1_REPORT_PATH) == P1_REPORT_SHA256
    result = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
    metrics = result["selection_metrics"]
    assert result["status"] == "failed_selection"
    assert result["candidate_id"] == "P1"
    assert result["candidate_consumed"] is True
    assert result["case_level_details_emitted"] is False
    assert result["passing_threshold_window"] == []
    assert result["onnx_parity_passed"] is True
    assert result["parent_role_argmax_preserved"] is True
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert metrics["scene_count"] == 128
    assert metrics["exact_scene_count"] == 123
    assert metrics["true_positives"] == 1024
    assert metrics["false_positives"] == 3
    assert metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == 0
    assert metrics["prohibited_structure_hits"] == 3
    assert "cases" not in result and "predictions" not in result


def test_p2_freezes_p1_except_final_veto_linear_and_preserves_roles(
    tmp_path: Path,
) -> None:
    model = FrozenP1FinalVetoProposalNet()
    model.load_p1_state_dict(_load_p1_state())
    trainable = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    assert trainable == ["veto_head.4.bias", "veto_head.4.weight"]
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())
    assert all(
        not parameter.requires_grad
        for layer in (model.veto_head[0], model.veto_head[2])
        for parameter in layer.parameters()
    )
    evidence = torch.linspace(-1.0, 1.0, 4 * FEATURE_COUNT).reshape(
        1, 4, FEATURE_COUNT
    )
    crops = torch.linspace(0.0, 1.0, 4 * 2 * 32 * 128).reshape(
        1, 4, 2, 32, 128
    )
    structure = torch.linspace(
        0.0, 1.0, 4 * STRUCTURE_FEATURE_COUNT
    ).reshape(1, 4, STRUCTURE_FEATURE_COUNT)
    with torch.inference_mode():
        output = model(evidence, crops, structure)
        parent = model.parent(evidence, crops)
    assert torch.equal(output[:, :, 2:].argmax(dim=2), parent[:, :, 2:].argmax(dim=2))
    path = tmp_path / "morphology-veto-v27-p2-preflight.onnx"
    torch.onnx.export(
        model.eval(),
        (evidence, crops, structure),
        path,
        input_names=["proposal_evidence", "proposal_crops", "structure_features"],
        output_names=["proposal_and_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "structure_features": {1: "proposal_count"},
            "proposal_and_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {
        "proposal_evidence": evidence.numpy(),
        "proposal_crops": crops.numpy(),
        "structure_features": structure.numpy(),
    })[0]
    assert float(np.max(np.abs(output.numpy() - actual))) <= 1e-5
    assert np.array_equal(
        np.argmax(actual[:, :, 2:], axis=2),
        np.argmax(parent.numpy()[:, :, 2:], axis=2),
    )


def test_p2_objective_targets_top_hard_negatives_and_anchors_positives() -> None:
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    logits = torch.tensor([
        [-3.0, 3.0],
        [2.0, -2.0],
        [4.0, -4.0],
        [-2.0, 2.0],
    ])
    p1_logits = logits.clone()
    p1_logits[3] = torch.tensor([-1.5, 1.5])
    targets = torch.tensor([0, 0, 0, 1])
    total, components = p2_objective(
        logits,
        p1_logits,
        targets,
        torch.ones(2),
        config,
    )
    assert total > 0
    assert components["hard_negative"] > 0
    assert components["positive_anchor"] > 0
    assert config["false_positive_weight"] == 8.0
    assert config["hard_negative_fraction"] == 0.1
    assert config["hard_negative_weight"] == 6.0
    assert config["positive_anchor_weight"] == 4.0


def test_p2_preflight_binds_exact_p1_and_keeps_public_archive_locked() -> None:
    config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS
    )
    evidence = p2_preflight()
    assert evidence["config"] == config
    assert evidence["seal"]["public_evaluations"] == 0
    assert evidence["seal"]["public_execution_authorized"] is False
    assert config["selection_evaluation_limit"] == 1
    assert config["case_level_predecessor_evidence_used"] is False
    assert config["validation_or_public_pixels_used_for_training"] is False
    assert config["public_execution_authorized"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["production_approval"] is False
    assert config["release_eligible"] is False
