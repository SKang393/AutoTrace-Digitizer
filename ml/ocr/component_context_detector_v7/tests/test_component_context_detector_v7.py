# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import (
    build_split,
    encode_proposal,
    proposal_examples,
    proposal_labels,
    proposal_summary,
    proposals,
    split_fingerprint,
)
from ml.ocr.component_context_detector_v7.model import ComponentContextNet
from ml.ocr.component_context_detector_v7.prepare_split import SPLIT_SOURCE_PATHS
from ml.ocr.component_context_detector_v7.protocol import ENCODED_WIDTH, INPUT_CHANNELS, REVISION, TASK, protocol_configuration
from ml.ocr.component_context_detector_v7.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.component_context_detector_v7.train_p1 import (
    CONFIG_PATH as P1_CONFIG_PATH,
    RUNNER_SOURCE_PATHS as P1_RUNNER_SOURCE_PATHS,
    _export,
)
from ml.ocr.component_context_detector_v7.train_p2 import (
    CONFIG_PATH as P2_CONFIG_PATH,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
)
from ml.ocr.component_context_detector_v7.train_p3 import (
    CONFIG_PATH as P3_CONFIG_PATH,
    RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_context_detector_v7"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def test_fresh_splits_are_deterministic_disjoint_and_proposal_complete() -> None:
    train = build_split("train")
    validation = build_split("validation")
    assert split_fingerprint(train) == split_fingerprint(build_split("train"))
    assert split_fingerprint(train) != split_fingerprint(validation)
    assert {scene.renderer_family for scene in train}.isdisjoint({scene.renderer_family for scene in validation})
    assert all(scene.raster.shape == (320, 640) and scene.raster.dtype == np.uint8 for scene in validation)
    summary = proposal_summary(validation)
    assert summary == {
        "scene_count": 64,
        "truth_region_count": 256,
        "proposal_count": 795,
        "positive_proposal_count": 256,
        "negative_proposal_count": 539,
    }


def test_dual_context_encoding_and_order_are_frozen() -> None:
    scene = build_split("validation")[0]
    candidates = proposals(scene.raster)
    assert list(candidates) == sorted(candidates, key=lambda item: (item.top, item.left, item.bottom, item.right))
    assert proposal_labels(scene, candidates).sum() == len(scene.truths)
    encoded = encode_proposal(scene.raster, candidates[0])
    assert encoded.shape == (INPUT_CHANNELS, 32, ENCODED_WIDTH)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()
    assert float(encoded.min()) >= 0.0
    assert float(encoded.max()) <= 1.0
    assert not np.array_equal(encoded[0, :, :128], encoded[1, :, :128])


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
    assert seal["predecessor_public_sample_or_pixel_inspection_used"] is False
    assert seal["truth_hidden_from_training_runner"] is True
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    gate = json.loads((ROOT / "gates/sealed-public-v1.json").read_text(encoding="utf-8"))
    assert gate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    assert gate["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))


def test_failed_p1_and_p2_are_consumed_and_p3_is_the_final_preregistered_candidate() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["task"] == TASK and item["revision"] == REVISION)
    p1_config = json.loads((REPO_ROOT / P1_CONFIG_PATH).read_text(encoding="utf-8"))
    p2_config = json.loads((REPO_ROOT / P2_CONFIG_PATH).read_text(encoding="utf-8"))
    p3_config = json.loads((REPO_ROOT / P3_CONFIG_PATH).read_text(encoding="utf-8"))
    p1_result = json.loads((ROOT / "P1_RESULT.json").read_text(encoding="utf-8"))
    p2_result = json.loads((ROOT / "P2_RESULT.json").read_text(encoding="utf-8"))
    assert entry["status"] == "candidate_3_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P3"]
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P3"
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(REPO_ROOT / P1_CONFIG_PATH)
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(REPO_ROOT / P2_CONFIG_PATH)
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(REPO_ROOT / P3_CONFIG_PATH)
    assert p1_config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, P1_RUNNER_SOURCE_PATHS)
    assert p2_config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS)
    assert p3_config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS)
    assert p1_result["status"] == "failed_selection"
    assert p1_result["selection_exact_scene_count"] == 62
    assert p1_result["selection_false_positives"] == 2
    assert p1_result["selection_false_negatives"] == 0
    assert p1_result["public_gate_archive_opened"] is False
    assert sha256_file(ROOT / "P1_RESULT.json") == entry["p1_result_sha256"]
    assert p2_result["status"] == "failed_selection"
    assert p2_result["selection_exact_scene_count"] == 63
    assert p2_result["selection_false_positives"] == 1
    assert p2_result["selection_false_negatives"] == 0
    assert p2_result["public_gate_archive_opened"] is False
    assert sha256_file(ROOT / "P2_RESULT.json") == entry["p2_result_sha256"]
    assert p3_config["source_checkpoint_sha256"] == p1_result["checkpoint_sha256"]
    assert p3_config["p2_result_sha256"] == sha256_file(ROOT / "P2_RESULT.json")
    assert p3_config["label_smoothing"] == 0.05
    assert p3_config["seed"] == p2_config["seed"]
    assert p3_config["selection_thresholds"] == p2_config["selection_thresholds"]
    assert not (ROOT / "P3_RESULT.json").exists()
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_p1_model_exports_dynamic_cpu_onnx_before_training(tmp_path: Path) -> None:
    values, _ = proposal_examples(build_split("validation"))
    model = ComponentContextNet().eval()
    source = torch.from_numpy(values[:8])
    with torch.inference_mode():
        expected = model(source).numpy()
    assert expected.shape == (8, 2)
    path = tmp_path / "component-context-preflight.onnx"
    _export(model, source, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {"region_proposals": values[:8]})[0], dtype=np.float32)
    assert actual.shape == expected.shape
    assert float(np.max(np.abs(actual - expected))) <= 1e-5
    dynamic = np.asarray(session.run(None, {"region_proposals": values[:3]})[0], dtype=np.float32)
    assert dynamic.shape == (3, 2)
