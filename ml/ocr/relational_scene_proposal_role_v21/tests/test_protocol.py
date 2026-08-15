# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.ocr.relational_scene_proposal_role_v21.dataset import (
    encode_scene,
    label_vocabulary,
    proposal_summary,
    render_scene,
    split_fingerprint,
)
from ml.ocr.relational_scene_proposal_role_v21.model import RelationalSceneProposalRoleNet
from ml.ocr.relational_scene_proposal_role_v21.protocol import (
    CANDIDATE_LIMIT,
    ENCODED_WIDTH,
    ROLE_ORDER,
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
