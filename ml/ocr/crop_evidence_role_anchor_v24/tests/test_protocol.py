# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import sha256_file
from ml.ocr.crop_evidence_role_anchor_v24.dataset import (
    encode_scene, proposal_summary, render_scene,
)
from ml.ocr.crop_evidence_role_anchor_v24.model import CropEvidenceRoleAnchorNet
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


def test_canonical_budget_preregisters_v24_without_opening_execution() -> None:
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
    assert entry["status"] == "preregistered"
    assert entry["protocol_sha256"] == sha256_file(protocol_path)
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["split_materialized"] is False
    assert entry["selection_evaluations"] == 0
    assert entry["execution_authorized"] is False
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


def test_v24_has_not_materialized_or_opened_any_gate() -> None:
    root = REPO_ROOT / "ml/ocr/crop_evidence_role_anchor_v24"
    assert not (root / "SPLIT_SEAL.json").exists()
    assert not (root / "training").exists()
    assert not (root / "artifacts").exists()
    assert not (REPO_ROOT / "artifacts/production-validation/ocr-v24-train.zip").exists()
    assert not (REPO_ROOT / "artifacts/production-validation/ocr-v24-selection.zip").exists()
    assert not (REPO_ROOT / "artifacts/production-validation/ocr-v24-public.zip").exists()
