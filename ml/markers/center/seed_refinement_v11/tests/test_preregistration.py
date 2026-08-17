# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Pre-execution checks for the marker-center V11 defect class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.markers.center.decoupled_heads_v10.dataset import (
    DEGRADATION_FAMILIES as V10_DEGRADATION_FAMILIES,
    RENDERER_FAMILIES as V10_RENDERER_FAMILIES,
)
from ml.markers.center.seed_refinement_v11.dataset import (
    DEGRADATION_FAMILIES,
    PUBLIC_SCENE_COUNT,
    RENDERER_FAMILIES,
    TRAIN_SCENE_COUNT,
    VALIDATION_SCENE_COUNT,
    read_archive,
    render_scene,
)
from ml.markers.center.seed_refinement_v11.model import create_model
from ml.markers.center.seed_refinement_v11.protocol import (
    DESIGN_SOURCE_PATHS,
    EXPERIMENT_BUDGET,
    ONNX_PARITY_TOLERANCE,
    TRIGGER_RESULT_SHA256,
)
from ml.markers.center.seed_refinement_v11.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    run as run_public_gate,
)
from ml.markers.center.seed_refinement_v11.train_p1 import (
    RUNNER_SOURCE_PATHS,
    _evaluate,
    _export,
    _onnx_output,
    _torch_output,
    _verify_config_and_inputs,
)
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/seed_refinement_v11"
V10_ROOT = REPO_ROOT / "ml/markers/center/decoupled_heads_v10"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes(manifest: dict[str, object], names: tuple[str, ...]) -> set[str]:
    result: set[str] = set()
    for name in names:
        with np.load(REPO_ROOT / manifest[name]["archive_path"], allow_pickle=False) as archive:
            result.update(str(value) for value in archive["source_sha256"])
    return result


def test_protocol_is_bounded_and_does_not_weaken_any_gate() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    assert EXPERIMENT_BUDGET == 3
    assert ONNX_PARITY_TOLERANCE == 1e-5
    assert TRIGGER_RESULT_SHA256 == "c0b580c68346124b878521dc6ef46f1e3ed4fe587c29be35be66d5bb8992b62f"
    assert protocol["selection_gates"]["minimum_artifact_precision"] == 0.9
    assert protocol["selection_gates"]["minimum_artifact_recall"] == 0.95
    assert protocol["selection_gates"]["false_positive_maximum"] == 0
    assert protocol["selection_gates"]["false_negative_maximum"] == 0
    assert protocol["selection_gates"]["duplicate_maximum"] == 0
    assert protocol["selection_gates"]["prohibited_structure_hit_maximum"] == 0
    assert protocol["selection_gates"]["marker_artifact_hit_maximum"] == 0


def test_fresh_families_and_source_hashes_are_disjoint_from_v10() -> None:
    assert (TRAIN_SCENE_COUNT, VALIDATION_SCENE_COUNT, PUBLIC_SCENE_COUNT) == (512, 128, 160)
    for families, prior in (
        (RENDERER_FAMILIES, V10_RENDERER_FAMILIES),
        (DEGRADATION_FAMILIES, V10_DEGRADATION_FAMILIES),
    ):
        current_values = [set(families[name]) for name in ("train", "validation", "sealed_public")]
        assert not (current_values[0] & current_values[1])
        assert not (current_values[0] & current_values[2])
        assert not (current_values[1] & current_values[2])
        assert not set().union(*current_values) & set().union(*map(set, prior.values()))
    current = _json(ROOT / "SELECTION_MANIFEST.json")
    prior = _json(V10_ROOT / "SELECTION_MANIFEST.json")
    current_visible = _source_hashes(current, ("train", "validation"))
    prior_visible = _source_hashes(prior, ("train", "validation"))
    current_public = {
        item["image_sha256"] for item in _json(ROOT / "PUBLIC_DATASET_MANIFEST.json")["fixtures"]
    }
    prior_public = {
        item["image_sha256"] for item in _json(V10_ROOT / "PUBLIC_DATASET_MANIFEST.json")["fixtures"]
    }
    assert len(current_visible | current_public) == 800
    assert not (current_visible | current_public) & (prior_visible | prior_public)


def test_model_can_remove_and_add_seed_pixels_without_changing_dense_shape() -> None:
    model = create_model()
    value = torch.zeros((1, 3, 24, 32), dtype=torch.float32)
    value[:, 2, 5, 7] = 1.0
    with torch.no_grad():
        model.artifact_head.weight.zero_()
        model.artifact_head.bias.fill_(-10.0)
        removed = model(value)
        model.artifact_head.bias.fill_(10.0)
        added = model(torch.zeros_like(value))
    assert removed.shape == (1, 3, 24, 32)
    assert torch.all(removed[:, 1:2] == 2.5)
    assert removed[0, 2, 5, 7].item() < 0.35
    assert added[0, 2, 5, 7].item() > 0.35


def test_untrained_architecture_exports_dynamic_onnx_and_matches_cpu(tmp_path: Path) -> None:
    model = create_model().eval()
    value = np.zeros((1, 3, 28, 36), dtype=np.float32)
    value[:, 2, 4:8, 5:10] = 1.0
    path = tmp_path / "seed-refinement-v11.onnx"
    _export(model, torch.from_numpy(value), path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = _torch_output(model, value)
    actual = _onnx_output(session, value)
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert actual.shape == (1, 3, 28, 36)
    assert float(np.max(np.abs(expected - actual))) <= ONNX_PARITY_TOLERANCE


def test_refined_artifact_output_drives_selection_without_seed_union() -> None:
    selection = _json(ROOT / "SELECTION_MANIFEST.json")
    archive = read_archive(REPO_ROOT / selection["validation"]["archive_path"])
    scene_count = archive["inputs"].shape[0]
    one = {
        key: value[:1] if hasattr(value, "shape") and value.shape and value.shape[0] == scene_count else value
        for key, value in archive.items()
    }
    output = np.zeros((1, 3, *one["inputs"].shape[-2:]), dtype=np.float32)
    output[:, 0:1] = one["center_targets"]
    output[:, 1:2] = 2.5
    output[:, 2:3] = one["artifact_targets"]
    result = _evaluate(one, [output], 0.3)
    assert result["passed"] is True
    assert result["seed_removed_pixels"] > 0
    assert result["seed_added_pixels"] > 0
    assert result["artifact_precision"] == 1.0
    assert result["artifact_recall"] == 1.0


def test_frozen_sources_configs_and_ledger_match_exact_bytes() -> None:
    protocol = _json(ROOT / "PROTOCOL.json")
    freeze = _json(ROOT / "SPLIT_FREEZE_REPORT.json")
    config = _json(ROOT / "training/p1.json")
    gate = _json(ROOT / "gates/sealed-public-v1.json")
    ledger = _json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == protocol["revision"])
    assert protocol["state"] == "split_frozen_runner_and_public_evaluator_preregistered_execution_blocked"
    assert protocol["execution_authorized"] is False
    assert protocol["authorized_candidate_id"] is None
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert sha256_file(ROOT / "PROTOCOL.json") == entry["protocol_sha256"]
    assert sha256_file(ROOT / "SPLIT_FREEZE_REPORT.json") == protocol["split_freeze_report_sha256"]
    assert sha256_file(ROOT / "SELECTION_MANIFEST.json") == protocol["selection_manifest_sha256"]
    assert sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json") == protocol["sealed_public_test_seal_sha256"]
    assert sha256_file(ROOT / "PUBLIC_DATASET_MANIFEST.json") == protocol["public_dataset_manifest_sha256"]
    assert sha256_file(ROOT / "training/p1.json") == protocol["candidate_config_sha256"]
    assert sha256_file(ROOT / "gates/sealed-public-v1.json") == protocol["public_gate_config_sha256"]
    assert source_bundle_sha256(REPO_ROOT, DESIGN_SOURCE_PATHS) == freeze["generator_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS) == gate["expected_evaluator_source_bundle_sha256"]
    assert freeze["model_execution_count_at_freeze"] == 0
    assert freeze["optimizer_step_count_at_freeze"] == 0
    assert freeze["public_gate_archive_opened"] is False
    assert freeze["public_gate_evaluations"] == 0


def test_p1_preflight_binds_v10_feasibility_and_frozen_v11_archives() -> None:
    config = _json(ROOT / "training/p1.json")
    selection, train_path, validation_path = _verify_config_and_inputs(config)
    assert sha256_file(train_path) == selection["train"]["archive_sha256"]
    assert sha256_file(validation_path) == selection["validation"]["archive_sha256"]
    assert config["aggregate_only_evidence"] is True
    assert config["case_detail_or_pixels_inspected"] is False
    assert config["prior_checkpoint_reused"] is False
    assert config["prior_fixture_bytes_reused"] is False
    assert config["runtime_postprocess_profile"] == "nonmonotonic_seed_refinement_v1"


def test_public_gate_refuses_unapproved_candidate_before_model_or_archive_execution(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate-report.json"
    output_path = tmp_path / "public-report.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "P1",
                "selection_gate_passed": True,
                "onnx_path": "does-not-exist.onnx",
                "onnx_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not separately authorized"):
        run_public_gate(candidate_path, output_path)
    assert not output_path.exists()


def test_visible_scene_uses_new_identity_without_private_data() -> None:
    scene = render_scene("validation", 0)
    assert scene.scene_id == "marker-seed-refinement-v11-validation-0000"
    assert scene.renderer_family in RENDERER_FAMILIES["validation"]
    assert scene.degradation_family in DEGRADATION_FAMILIES["validation"]
