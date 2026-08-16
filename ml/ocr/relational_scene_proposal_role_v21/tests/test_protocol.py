# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.ocr.relational_scene_proposal_role_v21.dataset import (
    encode_scene,
    label_vocabulary,
    proposal_summary,
    render_scene,
    split_fingerprint,
)
from ml.ocr.relational_scene_proposal_role_v21.model import RelationalSceneProposalRoleNet
from ml.ocr.relational_scene_proposal_role_v21.prepare_split import SOURCE_PATHS
from ml.ocr.relational_scene_proposal_role_v21.train_p1 import (
    _choose_threshold,
    _evaluate_threshold,
    _gate_passed,
    source_bundle_sha256,
)
from ml.ocr.relational_scene_proposal_role_v21.train_p2 import (
    _proposal_class_weights,
)
from ml.ocr.relational_scene_proposal_role_v21.protocol import (
    CANDIDATE_LIMIT,
    ENCODED_WIDTH,
    ROLE_ORDER,
    THRESHOLDS,
    protocol_configuration,
    split_registration,
)


def test_protocol_is_fresh_synthetic_and_fail_closed() -> None:
    config = protocol_configuration()
    assert config["candidate_budget"]["candidate_limit"] == CANDIDATE_LIMIT == 3
    assert config["predecessor_aggregate_only"]["case_level_evidence_used"] is False
    assert config["predecessor_aggregate_only"]["fixture_bytes_truth_or_scene_ids_reused"] is False
    assert config["chandler_included"] is False
    assert config["generalization_label_included"] is False
    assert config["private_or_article_images"] is False
    assert config["external_training_data"] is False
    assert config["fixture_identity_frozen"] is False
    assert config["training_authorized"] is False
    assert config["public_execution_authorized"] is False
    assert config["marker_creation_evaluated"] is False
    assert config["artifact_mask_production_approval"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False


def test_protocol_json_matches_the_executable_preregistration() -> None:
    path = Path(__file__).resolve().parents[1] / "PROTOCOL.json"
    expected = json.loads(json.dumps(protocol_configuration()))
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_p1_config_and_separate_authorization_are_fixed_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "P1_CONFIG.json").read_text(encoding="utf-8"))
    assert config["candidate_id"] == "P1"
    assert config["expected_optimizer_steps"] == 1536
    assert config["epochs"] == 4
    assert config["thresholds"] == list(THRESHOLDS)
    assert config["split_seal_sha256"] == "085c93c73731ca97bc85d4eed52841547e6faab28effa56ca14db90d999b3047"
    assert config["training_authorized"] is False
    assert config["public_execution_authorized"] is False
    authorization = json.loads((root / "P1_TRAINING_AUTHORIZATION.json").read_text(encoding="utf-8"))
    assert authorization["authorized_source_commit"] == "d9d5ed2eda4f53da54660f47ef1de594b5e628b7"
    assert authorization["candidate_config_sha256"] == "e3fdbb0208a49b890ae4eebda0bf3db9b52417c31ecdbef5d9521fd327be5fca"
    assert authorization["runner_source_bundle_sha256"] == "f6a090b2611d41ddd045939f1c4e918d464297e99b080a9bee696a3ddc26a4a1"
    assert authorization["execution_limit"] == 1
    assert authorization["execution_count"] == 0
    assert authorization["training_authorized"] is True
    assert authorization["public_execution_authorized"] is False
    assert authorization["private_validation_authorized"] is False
    assert authorization["production_approval"] is False
    assert authorization["release_eligible"] is False
    result = json.loads((root / "P1_SELECTION_RESULT.json").read_text(encoding="utf-8"))
    assert result["p1_consumed"] is True
    assert result["optimizer_steps"] == 1536
    assert result["selection_gate_passed"] is False
    assert result["scene_count"] == 128
    assert result["exact_scene_count"] == 107
    assert result["true_positives"] == 1004
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 20
    assert result["duplicate_regions"] == 0
    assert result["prohibited_structure_hits"] == 1
    assert result["onnx_parity_passed"] is False
    assert result["public_archive_opened"] is False
    assert result["public_evaluation_count"] == 0
    assert result["production_approval"] is False
    assert result["release_eligible"] is False


def test_p2_is_preregistered_as_one_asymmetric_continuation_and_not_authorized() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "P2_CONFIG.json").read_text(encoding="utf-8"))
    assert config["candidate_id"] == "P2"
    assert config["candidate_type"] == "bounded-checkpoint-continuation"
    assert config["continuation_epochs"] == 1
    assert config["expected_candidate_optimizer_steps"] == 384
    assert config["expected_total_optimizer_steps"] == 1920
    assert config["positive_proposal_loss_multiplier"] == 2.0
    assert config["p1_checkpoint_sha256"] == "9c279efbb5980091d30b25def3aa99147e7faa04655e1e194aa093ba112f7a28"
    assert config["p1_report_sha256"] == "f4f5f24ea01148b311c89639e4e76040b8728cb96c667ca1d794e586092d9dc8"
    assert config["p1_selection_result_sha256"] == "d6f55fd369e4aade05f449de6431fb1c94ee15cbaa28c2898bfe6dcdd8c5c967"
    assert config["thresholds"] == list(THRESHOLDS)
    assert config["training_authorized"] is False
    assert config["public_execution_authorized"] is False
    assert not (root / "P2_TRAINING_AUTHORIZATION.json").exists()


def test_p2_positive_multiplier_changes_only_the_positive_class_pressure() -> None:
    labels = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    baseline = _proposal_class_weights(labels, 1.000001)
    asymmetric = _proposal_class_weights(labels, 2.0)
    assert asymmetric[1] / asymmetric[0] > baseline[1] / baseline[0]
    assert asymmetric.sum() == pytest.approx(2.0)


def test_runner_source_bundle_is_order_independent_and_path_bound() -> None:
    left = source_bundle_sha256({"b.py": "2" * 64, "a.py": "1" * 64})
    right = source_bundle_sha256({"a.py": "1" * 64, "b.py": "2" * 64})
    different_path = source_bundle_sha256({"c.py": "1" * 64, "b.py": "2" * 64})
    assert left == right
    assert left != different_path


def test_split_vocabularies_are_disjoint_and_exclude_private_labels() -> None:
    split_values: list[set[str]] = []
    for split in ("train", "validation", "sealed_public"):
        vocabulary = label_vocabulary(split)
        assert set(vocabulary) == set(ROLE_ORDER)
        values = {value for role_values in vocabulary.values() for value in role_values}
        assert "Generalization" not in values
        assert "Chandler" not in values
        split_values.append(values)
    assert all(
        split_values[left].isdisjoint(split_values[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_split_families_and_seed_offsets_are_disjoint() -> None:
    registrations = [split_registration(name) for name in ("train", "validation", "sealed_public")]
    assert len({item.seed_offset for item in registrations}) == 3
    renderer_sets = [set(item.renderer_families) for item in registrations]
    degradation_sets = [set(item.degradation_families) for item in registrations]
    assert all(renderer_sets[left].isdisjoint(renderer_sets[right]) for left in range(3) for right in range(left + 1, 3))
    assert all(degradation_sets[left].isdisjoint(degradation_sets[right]) for left in range(3) for right in range(left + 1, 3))
    assert all(item.scene_count > 0 for item in registrations)


def test_split_seal_binds_transitive_generator_sources_and_renderer_fonts() -> None:
    expected = {
        "ml/ocr/relational_scene_proposal_role_v21/PROTOCOL.json",
        "ml/ocr/relational_scene_proposal_role_v21/dataset.py",
        "ml/ocr/relational_scene_proposal_role_v21/model.py",
        "ml/ocr/relational_scene_proposal_role_v21/prepare_split.py",
        "ml/ocr/relational_scene_proposal_role_v21/protocol.py",
        "ml/ocr/cross_model_consensus_v9_p3/P3_SELECTION_RESULT.json",
        "ml/ocr/layout_conditioned_proposal_role_v15/dataset.py",
        "ml/ocr/layout_conditioned_proposal_role_v15/protocol.py",
        "ml/ocr/component_context_detector_v7/dataset.py",
        "ml/ocr/component_context_detector_v7/protocol.py",
        "ml/ocr/component_region_detector_v6/dataset.py",
        "ml/ocr/component_region_detector_v6/protocol.py",
        "ml/markers/gate_seal.py",
        "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf",
        "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf",
        "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf",
    }
    assert {path.as_posix() for path in SOURCE_PATHS} == expected


def test_dynamic_relational_model_preserves_proposal_order_and_output_contract() -> None:
    torch.manual_seed(9)
    model = RelationalSceneProposalRoleNet().eval()
    for proposal_count in (3, 11):
        value = torch.zeros((1, proposal_count, 2, 32, ENCODED_WIDTH), dtype=torch.float32)
        with torch.inference_mode():
            output = model(value)
        assert output.shape == (1, proposal_count, 2 + len(ROLE_ORDER))
        assert torch.isfinite(output).all()


def test_dynamic_relational_model_exports_and_matches_cpu_onnx(tmp_path: Path) -> None:
    torch.manual_seed(10)
    model = RelationalSceneProposalRoleNet().eval()
    source = torch.rand((1, 5, 2, 32, ENCODED_WIDTH), dtype=torch.float32)
    output_path = tmp_path / "relational-scene-proposal-role-v21.onnx"
    torch.onnx.export(
        model,
        source,
        str(output_path),
        input_names=["proposals"],
        output_names=["logits"],
        dynamic_axes={"proposals": {1: "proposal_count"}, "logits": {1: "proposal_count"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert session.get_providers()[0] == "CPUExecutionProvider"
    for proposal_count in (3, 11):
        value = torch.rand((1, proposal_count, 2, 32, ENCODED_WIDTH), dtype=torch.float32)
        with torch.inference_mode():
            expected = model(value).numpy()
        actual = session.run(["logits"], {"proposals": value.numpy()})[0]
        assert actual.shape == expected.shape
        assert float(np.max(np.abs(expected - actual))) <= 0.00001


def test_selection_metrics_require_every_role_and_reject_structure_acceptance() -> None:
    proposal_truth = (np.asarray([1] * len(ROLE_ORDER) + [0], dtype=np.int64),)
    role_truth = (np.asarray(list(range(len(ROLE_ORDER))) + [-1], dtype=np.int64),)
    predicted_roles = (np.asarray(list(range(len(ROLE_ORDER))) + [0], dtype=np.int64),)
    passing_probabilities = (np.asarray([0.9] * len(ROLE_ORDER) + [0.1], dtype=np.float32),)
    passing = _evaluate_threshold(passing_probabilities, predicted_roles, proposal_truth, role_truth, 0.5)
    assert passing.exact_scene_count == 1
    assert passing.true_positives == len(ROLE_ORDER)
    assert passing.false_positives == 0
    assert passing.false_negatives == 0
    assert passing.role_accuracy == pytest.approx(1.0)
    assert min(passing.per_role_accuracy.values()) == pytest.approx(1.0)
    assert _gate_passed(passing)

    failing_probabilities = (np.asarray([0.9] * (len(ROLE_ORDER) + 1), dtype=np.float32),)
    failing = _evaluate_threshold(failing_probabilities, predicted_roles, proposal_truth, role_truth, 0.5)
    assert failing.false_positives == 1
    assert failing.prohibited_structure_hits == 1
    assert not _gate_passed(failing)
    assert _choose_threshold((failing, passing)) == passing


def test_fresh_probe_scenes_use_production_proposals_and_all_roles() -> None:
    scenes = tuple(render_scene(split, 0) for split in ("train", "validation", "sealed_public"))
    assert len({split_fingerprint((scene,)) for scene in scenes}) == 3
    assert all(len(scene.truths) == len(ROLE_ORDER) for scene in scenes)
    assert all({truth.role for truth in scene.truths} == set(ROLE_ORDER) for scene in scenes)
    assert all("Generalization" not in truth.text for scene in scenes for truth in scene.truths)
    for scene in scenes:
        encoded, candidates, proposal_labels, role_labels = encode_scene(scene)
        assert encoded.shape[0] == len(candidates) == len(proposal_labels) == len(role_labels)
        assert int((proposal_labels == 1).sum()) == len(ROLE_ORDER)
        assert set(role_labels[role_labels >= 0].tolist()) == set(range(len(ROLE_ORDER)))


def test_probe_proposal_summary_is_complete_and_contains_structure_negatives() -> None:
    scenes = tuple(render_scene("train", index) for index in range(4))
    summary = proposal_summary(scenes)
    assert summary["scene_count"] == 4
    assert summary["positive_proposal_count"] == 4 * len(ROLE_ORDER)
    assert summary["negative_proposal_count"] > 0
    assert summary["exactly_one_production_proposal_per_truth"] is True
