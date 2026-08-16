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


def test_freeze_state_remains_fail_closed_before_training() -> None:
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
    assert not (REPO_ROOT / "ml/ocr/scene_evidence_attention_v22/P1_RESULT.json").exists()
    assert not (
        REPO_ROOT
        / "ml/ocr/scene_evidence_attention_v22/artifacts/P1-run"
    ).exists()
