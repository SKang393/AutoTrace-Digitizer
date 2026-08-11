# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_region_detector_v6 import sealed_gate as public_gate
from ml.ocr.component_region_detector_v6.dataset import (
    build_split,
    encode_proposal,
    load_sealed_public_archive,
    proposal_examples,
    proposal_labels,
    proposal_summary,
    proposals,
    split_fingerprint,
)
from ml.ocr.component_region_detector_v6.prepare_split import SPLIT_SOURCE_PATHS
from ml.ocr.component_region_detector_v6.protocol import ENCODED_WIDTH, REVISION, TASK, protocol_configuration
from ml.ocr.component_region_detector_v6.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.component_region_detector_v6.train_p1 import CONFIG_PATH, RUNNER_SOURCE_PATHS, _export
from ml.ocr.component_region_detector_v6.model import ComponentRegionNet


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_region_detector_v6"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def test_fresh_splits_are_deterministic_disjoint_and_proposal_complete() -> None:
    train = build_split("train")
    validation = build_split("validation")
    assert split_fingerprint(train) == split_fingerprint(build_split("train"))
    assert split_fingerprint(train) != split_fingerprint(validation)
    assert {scene.renderer_family for scene in train}.isdisjoint({scene.renderer_family for scene in validation})
    assert all(scene.raster.shape == (256, 512) and scene.raster.dtype == np.uint8 for scene in validation)
    summary = proposal_summary(validation)
    assert summary["scene_count"] == 48
    assert summary["truth_region_count"] == 144
    assert summary["positive_proposal_count"] == 144
    assert summary["negative_proposal_count"] > 0


def test_proposal_encoding_and_order_are_frozen() -> None:
    scene = build_split("validation")[0]
    candidates = proposals(scene.raster)
    assert list(candidates) == sorted(candidates, key=lambda item: (item.top, item.left, item.bottom, item.right))
    assert proposal_labels(scene, candidates).sum() == len(scene.truths)
    encoded = encode_proposal(scene.raster, candidates[0])
    assert encoded.shape == (1, 32, ENCODED_WIDTH)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()
    assert float(encoded.min()) >= 0.0
    assert float(encoded.max()) <= 1.0


def test_frozen_split_and_gate_hashes_are_directly_bound() -> None:
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
    expected_protocol = json.loads(json.dumps(protocol_configuration()))
    expected_protocol["split_generator_source_paths"] = [p.as_posix() for p in SPLIT_SOURCE_PATHS]
    expected_protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    assert protocol == expected_protocol
    seal_path = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["chandler_included"] is False
    assert seal["generalization_label_included"] is False
    assert seal["predecessor_public_archive_reused"] is False
    assert seal["truth_hidden_from_training_runner"] is True
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    scenes = load_sealed_public_archive(REPO_ROOT / seal["fixture_archive_path"])
    assert split_fingerprint(scenes) == seal["split_fingerprint"]
    assert proposal_summary(scenes)["positive_proposal_count"] == seal["truth_region_count"]
    gate = json.loads((ROOT / "gates/sealed-public-v1.json").read_text(encoding="utf-8"))
    assert gate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    assert gate["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))


def test_budget_records_consumed_p1_and_authorizes_only_public_gate() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["task"] == TASK and item["revision"] == REVISION)
    result_path = ROOT / "P1_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output = ROOT / "artifacts/P1-run"
    seal_root = REPO_ROOT / "ml/markers/training-seals/ocr-detection/graph-text-component-region-v6/P1"
    assert entry["status"] == "candidate_1_selected_public_gate_pending"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized_on_selection_pass"] is True
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(REPO_ROOT / CONFIG_PATH)
    assert entry["candidate_checkpoint_sha256"]["P1"] == result["checkpoint_sha256"]
    assert entry["candidate_onnx_sha256"]["P1"] == result["onnx_sha256"]
    assert entry["p1_result_sha256"] == sha256_file(result_path)
    assert entry["p1_training_report_sha256"] == sha256_file(output / "candidate-report.json")
    assert entry["p1_training_opened_seal_sha256"] == sha256_file(seal_root / "opened.json")
    assert entry["p1_training_result_seal_sha256"] == sha256_file(seal_root / "result.json")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["selection_manifest_sha256"] == sha256_file(ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert entry["public_gate_config_sha256"] == sha256_file(ROOT / "gates/sealed-public-v1.json")
    assert result["status"] == "selected_public_gate_pending"
    assert result["selected_threshold"] == 0.65
    assert result["selection_exact_scene_count"] == result["selection_scene_count"] == 48
    assert result["selection_true_positives"] == 144
    assert result["selection_false_positives"] == 0
    assert result["selection_false_negatives"] == 0
    assert result["selection_duplicate_region_count"] == 0
    assert result["selection_prohibited_structure_hits"] == 0
    assert result["onnx_parity_passed"] is True
    assert sha256_file(output / "graph-text-component-region-v6-p1.onnx") == result["onnx_sha256"]
    assert sha256_file(output / "graph-text-component-region-v6-p1.pt") == result["checkpoint_sha256"]
    assert result["sealed_public_archive_opened_by_training"] is False
    assert result["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False


def test_public_gate_refuses_nonselected_report_before_opening_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(public_gate, "require_committed_sources", lambda *_args, **_kwargs: None)
    source = ROOT / "artifacts/P1-run/candidate-report.json"
    tampered = json.loads(source.read_text(encoding="utf-8"))
    tampered["selected_threshold"] = 0.55
    report_path = tmp_path / "tampered-selection.json"
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    output_path = tmp_path / "public-report.json"
    with pytest.raises(RuntimeError, match="not authorized by the canonical ledger"):
        public_gate.evaluate_candidate(
            onnx_path=ROOT / "artifacts/P1-run/graph-text-component-region-v6-p1.onnx",
            selection_report_path=report_path,
            output_path=output_path,
        )
    assert not output_path.exists()


def test_p1_model_exports_dynamic_cpu_onnx_before_training(tmp_path: Path) -> None:
    values, _ = proposal_examples(build_split("validation"))
    model = ComponentRegionNet().eval()
    source = torch.from_numpy(values[:8])
    with torch.inference_mode():
        expected = model(source).numpy()
    assert expected.shape == (8, 2)
    path = tmp_path / "component-region-preflight.onnx"
    _export(model, source, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {"region_proposals": values[:8]})[0], dtype=np.float32)
    assert actual.shape == expected.shape
    assert float(np.max(np.abs(actual - expected))) <= 1e-5
    dynamic = np.asarray(session.run(None, {"region_proposals": values[:3]})[0], dtype=np.float32)
    assert dynamic.shape == (3, 2)
