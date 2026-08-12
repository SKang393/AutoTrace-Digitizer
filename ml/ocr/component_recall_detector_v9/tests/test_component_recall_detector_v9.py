# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration tests for OCR component-recall V9."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.component_recall_detector_v9.dataset import (
    build_split,
    encode_proposal,
    proposal_summary,
    proposals,
    split_fingerprint,
)
from ml.ocr.component_recall_detector_v9.model import ComponentRecallNet
from ml.ocr.component_recall_detector_v9.protocol import REVISION, SPLITS, protocol_configuration
from ml.ocr.component_recall_detector_v9.sealed_gate import EVALUATOR_SOURCE_PATHS, SPLIT_CONFIG_PATH
from ml.ocr.component_recall_detector_v9.train_p1 import CONFIG_PATH, RUNNER_SOURCE_PATHS, _export


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_recall_detector_v9"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fresh_splits_are_disjoint_and_proposal_complete() -> None:
    selection = _load(ROOT / "SELECTION_MANIFEST.json")
    seal = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    fingerprints: set[str] = set()
    for registration in SPLITS:
        scenes = build_split(registration.split)
        summary = proposal_summary(scenes)
        fingerprint = split_fingerprint(scenes)
        assert len(scenes) == registration.scene_count
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        assert summary["positive_proposal_count"] == summary["truth_region_count"]
        expected = selection[registration.split] if registration.split != "sealed_public" else seal
        assert summary == {key: expected[key] for key in summary}
        assert fingerprint == expected["split_fingerprint"]


def test_v9_retains_exact_proposal_tensor_and_v8_architecture_contract(tmp_path: Path) -> None:
    scene = build_split("validation")[0]
    encoded = encode_proposal(scene.raster, proposals(scene.raster)[0])
    assert encoded.shape == (2, 32, 140)
    assert encoded.dtype == np.float32
    model = ComponentRecallNet().eval()
    path = tmp_path / "v9-preflight.onnx"
    _export(model, torch.from_numpy(np.stack((encoded, encoded))), path)
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [value.name for value in session.get_inputs()] == ["region_proposals"]
    assert [value.name for value in session.get_outputs()] == ["region_logits"]
    output = session.run(None, {"region_proposals": np.stack((encoded, encoded)).astype(np.float32)})[0]
    assert output.shape == (2, 2)


def test_p1_is_the_only_checksum_bound_authorized_candidate() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    expected = json.loads(json.dumps(protocol_configuration()))
    expected["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected
    config = _load(REPO_ROOT / CONFIG_PATH)
    ledger = _load(LEDGER_PATH)
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(REPO_ROOT / CONFIG_PATH)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)


def test_public_gate_is_frozen_hidden_and_unapproved() -> None:
    seal = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    gate = _load(REPO_ROOT / SPLIT_CONFIG_PATH)
    assert seal["truth_hidden_from_training_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert not (ROOT / "P1_RESULT.json").exists()
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*component*recall*v9*.json"))
