# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed tests for OCR evidence-rescue V25 preregistration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    source_bundle_sha256,
)
from ml.ocr.evidence_rescue_v25.dataset import proposal_summary, render_scene
from ml.ocr.evidence_rescue_v25.model import FrozenCropResidualCtcRescueNet
from ml.ocr.evidence_rescue_v25.sealed_gate import (
    EVALUATOR_SOURCE_PATHS,
    GATE_CONFIG,
    PUBLIC_REVISION,
)
from ml.ocr.evidence_rescue_v25.protocol import (
    CANDIDATE_LIMIT,
    FEATURE_COUNT,
    PARENT_RESULT_PATH,
    PARENT_RESULT_SHA256,
    ROLE_ORDER,
    protocol_configuration,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_fresh_bounded_and_fail_closed() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (REPO_ROOT / "ml/ocr/evidence_rescue_v25/PROTOCOL.json").read_text(
            encoding="utf-8"
        )
    )
    assert tracked == json.loads(json.dumps(protocol))
    assert protocol["candidate_budget"]["candidate_limit"] == CANDIDATE_LIMIT == 3
    assert protocol["candidate_budget"]["optimizer_steps_maximum"] == 0
    assert protocol["fixture_identity_frozen"] is False
    assert protocol["training_authorized"] is False
    assert protocol["public_execution_authorized"] is False
    assert protocol["marker_creation_evaluated"] is False
    assert protocol["private_validation_authorized"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert "Chandler" in protocol["data_scope"]
    assert "Generalization" in protocol["data_scope"]


def test_v24_trigger_is_exact_aggregate_only_terminal_record() -> None:
    path = REPO_ROOT / PARENT_RESULT_PATH
    assert sha256_file(path) == PARENT_RESULT_SHA256
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["candidate_id"] == "P2"
    assert result["candidate_consumed"] is True
    assert result["status"] == "failed_selection"
    assert result["case_level_details_emitted"] is False
    assert result["selection_metrics"]["false_positives"] == 0
    assert result["selection_metrics"]["false_negatives"] == 2
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert "cases" not in result and "predictions" not in result


def test_canonical_budget_authorizes_only_p1_while_public_gate_stays_locked() -> None:
    ledger = json.loads(
        (REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item for item in ledger["revisions"]
        if item["task"] == "ocr-detection-recognition"
        and item["revision"] == "graph-text-evidence-rescue-v25"
    )
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["protocol_sha256"] == sha256_file(
        REPO_ROOT / "ml/ocr/evidence_rescue_v25/PROTOCOL.json"
    )
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["split_materialized"] is True
    split_seal = REPO_ROOT / entry["split_seal_path"]
    assert entry["split_seal_sha256"] == sha256_file(split_seal)
    candidate_config = REPO_ROOT / entry["candidate_config_paths"]["P1"]
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(candidate_config)
    assert entry["public_evaluator_preregistered"] is True
    public_config = REPO_ROOT / entry["public_gate_config_path"]
    assert entry["public_gate_config_sha256"] == sha256_file(public_config)
    assert entry["selection_evaluations"] == 0
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_public_evaluator_is_preregistered_without_opening_hidden_truth() -> None:
    config_path = REPO_ROOT / "ml/ocr/evidence_rescue_v25/gates/sealed-public-v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_seal = json.loads(
        (REPO_ROOT / config["split_seal_path"]).read_text(encoding="utf-8")
    )
    public = split_seal["splits"]["sealed_public"]
    assert config["task"] == "ocr-detection-recognition"
    assert config["revision"] == PUBLIC_REVISION
    assert config["expected_dataset_manifest_sha256"] == public["manifest_sha256"]
    assert config["public_fixture_archive_path"] == public["archive_path"]
    assert config["public_fixture_archive_sha256"] == public["archive_sha256"]
    assert sha256_file(REPO_ROOT / public["archive_path"]) == public["archive_sha256"]
    assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert config["expected_gate_config_sha256"] == sha256_bytes(
        canonical_json_bytes(GATE_CONFIG)
    )
    assert config["public_execution_authorized"] is False
    assert config["public_evaluations"] == 0
    assert config["public_archive_opened"] is False
    gate_root = REPO_ROOT / "ml/markers/gate-seals/ocr-detection-recognition"
    for opened_path in gate_root.glob("*/opened.json") if gate_root.is_dir() else ():
        opened = json.loads(opened_path.read_text(encoding="utf-8"))
        assert opened.get("binding", {}).get("revision") != PUBLIC_REVISION


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
    hashes = {scene.raster.tobytes() for scene in scenes}
    assert len(hashes) == 3
    for scene in scenes:
        summary = proposal_summary((scene,))
        assert summary["positive_proposal_count"] == len(ROLE_ORDER)
        assert summary["negative_proposal_count"] > 0
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert scene.raster.shape == (320, 640)
        assert scene.scene_id.startswith("evidence-rescue-v25-")


class _FakeAnchor(nn.Module):
    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        count = evidence.shape[1]
        proposal = torch.tensor([-2.0, 2.0], dtype=evidence.dtype).reshape(1, 1, 2)
        roles = torch.arange(len(ROLE_ORDER), dtype=evidence.dtype).reshape(1, 1, -1)
        return torch.cat((proposal.expand(1, count, 2), roles.expand(1, count, -1)), dim=2)


class _FakeParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeAnchor()

    def forward(self, evidence: torch.Tensor, crops: torch.Tensor) -> torch.Tensor:
        del crops
        count = evidence.shape[1]
        proposal = torch.tensor(
            [[[-2.0, 2.0], [2.0, -2.0], [2.0, -2.0]]], dtype=evidence.dtype,
        )[:, :count]
        roles = torch.arange(len(ROLE_ORDER), dtype=evidence.dtype).reshape(1, 1, -1)
        return torch.cat((proposal, roles.expand(1, count, -1)), dim=2)


def test_rescue_retains_parent_acceptance_and_requires_all_ctc_evidence() -> None:
    model = FrozenCropResidualCtcRescueNet()
    model.parent = _FakeParent()
    evidence = torch.zeros(1, 3, FEATURE_COUNT)
    evidence[:, :, 10] = 0.90
    evidence[:, :, 11] = 0.90
    evidence[:, :, 12] = 0.50
    evidence[:, :, 13] = 0.20
    evidence[:, :, 14] = 0.20
    evidence[:, :, 15] = 0.25
    evidence[:, :, 16] = 1.0
    evidence[:, 2, 10] = 0.10
    crops = torch.zeros(1, 3, 2, 32, 128)
    output = model(evidence, crops)
    assert (output[:, :, 1] > output[:, :, 0]).tolist() == [[True, True, False]]
    expected_roles = model.parent(evidence, crops)[:, :, 2:]
    assert torch.equal(output[:, :, 2:], expected_roles)


def test_model_exports_dynamic_cpu_with_exact_role_preservation(tmp_path: Path) -> None:
    torch.manual_seed(2501)
    model = FrozenCropResidualCtcRescueNet().eval()
    evidence = torch.linspace(-1.0, 1.0, 5 * FEATURE_COUNT).reshape(1, 5, FEATURE_COUNT)
    crops = torch.linspace(-1.0, 1.0, 5 * 2 * 32 * 128).reshape(1, 5, 2, 32, 128)
    path = tmp_path / "evidence-rescue-v25.onnx"
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
        values = evidence[:, :count].numpy().astype(np.float32)
        crop_values = crops[:, :count].numpy().astype(np.float32)
        with torch.inference_mode():
            expected = model(torch.from_numpy(values), torch.from_numpy(crop_values)).numpy()
        actual = np.asarray(session.run(None, {
            "proposal_evidence": values,
            "proposal_crops": crop_values,
        })[0], dtype=np.float32)
        assert actual.shape == (1, count, 2 + len(ROLE_ORDER))
        assert float(np.max(np.abs(expected - actual))) <= 1e-5
