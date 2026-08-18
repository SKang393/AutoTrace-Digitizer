# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration tests for OCR V28."""

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
from ml.ocr.margin_calibrator_v20.pipeline import ProposalRecord
from ml.ocr.morphology_veto_proposal_v27.dataset import render_scene as render_v27_scene
from ml.ocr.relational_neighborhood_proposal_v28.dataset import (
    proposal_summary,
    proposals,
    render_scene,
)
from ml.ocr.relational_neighborhood_proposal_v28.model import (
    RelationalNeighborhoodProposalNet,
)
from ml.ocr.relational_neighborhood_proposal_v28.prepare_split import SOURCE_PATHS
from ml.ocr.relational_neighborhood_proposal_v28.protocol import (
    CANDIDATE_LIMIT,
    FEATURE_COUNT,
    RELATION_FEATURE_COUNT,
    ROLE_ORDER,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.relational_neighborhood_proposal_v28.relations import (
    proposal_relation_features,
)
from ml.ocr.relational_neighborhood_proposal_v28.train_p1 import (
    RUNNER_SOURCE_PATHS,
    _proposal_objective,
    preflight,
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
            / "ml/ocr/relational_neighborhood_proposal_v28/PROTOCOL.json"
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
    assert protocol["architecture"]["proposal_weights_initialized_from_scratch"] is True
    assert protocol["selection_gates"]["onnx_parity_maximum_absolute_error"] == 1e-5
    assert "Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]


def test_v27_trigger_is_exact_aggregate_only_terminal_record() -> None:
    path = REPO_ROOT / TRIGGER_RESULT_PATH
    assert sha256_file(path) == TRIGGER_RESULT_SHA256
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["selection_metrics"]
    assert result["candidate_id"] == "P3"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["case_level_details_emitted"] is False
    assert metrics["scene_count"] == 128
    assert metrics["exact_scene_count"] == 123
    assert metrics["true_positives"] == 1024
    assert metrics["false_positives"] == 3
    assert metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == 0
    assert metrics["prohibited_structure_hits"] == 3
    assert result["onnx_parity_maximum_absolute_error"] <= 1e-5
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert "cases" not in result and "predictions" not in result


def test_canonical_budget_authorizes_only_checksum_bound_materialized_p1() -> None:
    ledger = json.loads(
        (
            REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item for item in ledger["revisions"]
        if item["revision"] == "graph-text-relational-neighborhood-proposal-v28"
    )
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["protocol_sha256"] == sha256_file(
        REPO_ROOT / entry["protocol_path"]
    )
    assert entry["split_materialized"] is True
    seal_path = REPO_ROOT / entry["split_seal_path"]
    assert sha256_file(seal_path) == entry["split_seal_sha256"]
    assert entry["split_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, SOURCE_PATHS,
    )
    for split, archive_path in entry["split_archive_paths"].items():
        assert sha256_file(REPO_ROOT / archive_path) == entry["split_archive_sha256"][split]
    config_path = REPO_ROOT / entry["candidate_config_paths"]["P1"]
    assert sha256_file(config_path) == entry["candidate_config_sha256"]["P1"]
    assert entry["selection_evaluations"] == 0
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert "exactly once" in entry["execution_authorization"]
    assert "public archive remains unauthorized and unopened" in entry[
        "execution_authorization"
    ]
    assert entry["manifest_created"] is False
    assert entry["model_store_promoted"] is False
    assert entry["private_validation"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_frozen_split_and_candidate_preflight_remain_fail_closed() -> None:
    seal_path = (
        REPO_ROOT / "ml/ocr/relational_neighborhood_proposal_v28/SPLIT_SEAL.json"
    )
    config_path = (
        REPO_ROOT / "ml/ocr/relational_neighborhood_proposal_v28/training/p1.json"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert seal["schema"] == "graphreader.ocr-relational-neighborhood-split-seal.v1"
    assert seal["source_commit"] == "d49a7b469ea787d2c991383608dd93e6565e4439"
    assert seal["source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, SOURCE_PATHS,
    )
    assert seal["cross_split_source_overlap_counts"] == {
        "train_sealed_public": 0,
        "train_validation": 0,
        "validation_sealed_public": 0,
    }
    assert seal["optimizer_steps_at_freeze"] == 0
    assert seal["selection_evaluations"] == 0
    assert seal["public_evaluations"] == 0
    assert seal["training_authorized"] is False
    assert seal["public_execution_authorized"] is False
    assert seal["private_data"] is False
    assert seal["chandler_used"] is False
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
    assert config["split_seal_sha256"] == sha256_file(seal_path)
    assert config["expected_runner_source_bundle_sha256"] == seal[
        "source_bundle_sha256"
    ]
    assert config["public_execution_authorized"] is False
    assert config["public_gate_evaluations"] == 0
    assert config["private_or_article_images"] is False
    assert config["chandler_included"] is False
    state = preflight()
    assert state["seal"] == seal


def test_freeze_inventory_binds_the_exact_candidate_runner_and_relational_sources() -> None:
    assert RUNNER_SOURCE_PATHS == SOURCE_PATHS
    assert len(SOURCE_PATHS) == len(set(SOURCE_PATHS))
    required = {
        "ml/ocr/relational_neighborhood_proposal_v28/model.py",
        "ml/ocr/relational_neighborhood_proposal_v28/relations.py",
        "ml/ocr/relational_neighborhood_proposal_v28/train_p1.py",
        "ml/ocr/relational_neighborhood_proposal_v28/prepare_split.py",
    }
    assert required <= {path.as_posix() for path in SOURCE_PATHS}
    assert all((REPO_ROOT / path).is_file() for path in SOURCE_PATHS)


def test_p1_objective_directly_penalizes_hard_false_regions() -> None:
    config = protocol_configuration()["candidate_p1"]
    targets = torch.tensor([1, 1, 0, 0, 0], dtype=torch.int64)
    class_weights = torch.ones(2, dtype=torch.float32)
    safer = torch.tensor([
        [-2.0, 3.0], [-1.5, 2.5], [3.0, -2.0], [2.5, -1.5], [2.0, -1.0],
    ])
    unsafe = safer.clone()
    unsafe[2:, 0] = -2.0
    unsafe[2:, 1] = 3.0
    safe_loss, safe_parts = _proposal_objective(
        safer, targets, class_weights, config,
    )
    unsafe_loss, unsafe_parts = _proposal_objective(
        unsafe, targets, class_weights, config,
    )
    assert torch.isfinite(safe_loss) and torch.isfinite(unsafe_loss)
    assert unsafe_loss > safe_loss
    assert unsafe_parts["hard_negative"] > safe_parts["hard_negative"]


def test_fresh_split_families_offsets_and_sample_bytes_are_disjoint() -> None:
    registrations = [
        split_registration(name) for name in ("train", "validation", "sealed_public")
    ]
    renderers = [set(value.renderer_families) for value in registrations]
    degradations = [set(value.degradation_families) for value in registrations]
    assert len({value.seed_offset for value in registrations}) == 3
    assert all(
        not renderers[left] & renderers[right]
        for left in range(3) for right in range(left + 1, 3)
    )
    assert all(
        not degradations[left] & degradations[right]
        for left in range(3) for right in range(left + 1, 3)
    )
    v28_hashes: set[str] = set()
    v27_hashes: set[str] = set()
    for split in ("train", "validation", "sealed_public"):
        scenes = tuple(render_scene(split, index) for index in range(3))
        summary = proposal_summary(scenes)
        assert summary["positive_proposal_count"] == 3 * len(ROLE_ORDER)
        assert summary["role_truth_counts"] == {role: 3 for role in ROLE_ORDER}
        for index, scene in enumerate(scenes):
            assert scene.scene_id.startswith("relational-neighborhood-v28-")
            v28_hashes.add(_png_sha256(scene.raster))
            v27_hashes.add(_png_sha256(render_v27_scene(split, index).raster))
    assert len(v28_hashes) == 9
    assert not v28_hashes & v27_hashes


def test_pairwise_features_are_truth_independent_bounded_and_permutation_equivariant() -> None:
    scene = render_scene("train", 0)
    count = min(7, len(proposals(scene.raster)))
    records = tuple(
        ProposalRecord(0, index, -1, "", "Other") for index in range(count)
    )
    values = proposal_relation_features(scene, records)
    assert values.shape == (count, count, RELATION_FEATURE_COUNT)
    assert np.isfinite(values).all()
    assert float(values.min()) >= -1.0 and float(values.max()) <= 1.0
    assert np.array_equal(np.diag(values[:, :, -1]), np.ones(count, dtype=np.float32))
    reverse = tuple(reversed(records))
    reversed_values = proposal_relation_features(scene, reverse)
    assert np.array_equal(reversed_values, values[::-1, ::-1])


def test_model_has_dynamic_scene_contract_and_preserves_frozen_role_argmax() -> None:
    torch.manual_seed(41)
    model = RelationalNeighborhoodProposalNet().eval()
    assert model.trainable_parameters()
    assert all(not parameter.requires_grad for parameter in model.role_parent.parameters())
    for count in (3, 7):
        evidence = torch.randn(1, count, FEATURE_COUNT)
        crops = torch.randn(1, count, 2, 32, 128)
        relations = torch.randn(1, count, count, RELATION_FEATURE_COUNT)
        with torch.no_grad():
            output = model(evidence, crops, relations)
            parent = model.role_parent(evidence, crops)
        assert output.shape == (1, count, 2 + len(ROLE_ORDER))
        assert torch.equal(output[:, :, 2:].argmax(dim=2), parent[:, :, 2:].argmax(dim=2))


def test_random_weight_onnx_contract_is_dynamic_and_within_strict_parity(tmp_path: Path) -> None:
    torch.manual_seed(43)
    model = RelationalNeighborhoodProposalNet().eval()
    evidence = torch.randn(1, 5, FEATURE_COUNT)
    crops = torch.randn(1, 5, 2, 32, 128)
    relations = torch.randn(1, 5, 5, RELATION_FEATURE_COUNT)
    path = tmp_path / "v28-contract.onnx"
    torch.onnx.export(
        model,
        (evidence, crops, relations),
        path,
        input_names=["proposal_evidence", "proposal_crops", "proposal_relations"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "proposal_relations": {1: "proposal_count", 2: "neighbor_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=17,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"],
    )
    for count in (3, 5):
        values = evidence[:, :count].numpy().astype(np.float32)
        crop_values = crops[:, :count].numpy().astype(np.float32)
        relation_values = relations[:, :count, :count].numpy().astype(np.float32)
        with torch.no_grad():
            expected = model(
                torch.from_numpy(values),
                torch.from_numpy(crop_values),
                torch.from_numpy(relation_values),
            ).numpy()
        actual = session.run(None, {
            "proposal_evidence": values,
            "proposal_crops": crop_values,
            "proposal_relations": relation_values,
        })[0]
        assert float(np.max(np.abs(expected - actual))) <= 1e-5
