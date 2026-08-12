# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration checks for OCR component-fusion V8."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.component_fusion_detector_v8.dataset import (
    build_split,
    encode_proposal,
    proposal_summary,
    proposals,
    split_fingerprint,
)
from ml.ocr.component_fusion_detector_v8.model import ComponentFusionNet
from ml.ocr.component_fusion_detector_v8.model_p2 import ComponentFusionP2Net
from ml.ocr.component_fusion_detector_v8.protocol import REVISION, SPLITS, protocol_configuration
from ml.ocr.component_fusion_detector_v8.train_p1 import _balanced_order
from ml.ocr.component_fusion_detector_v8.train_p2 import CONFIG_PATH, RUNNER_SOURCE_PATHS, _export


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_fusion_detector_v8"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fresh_splits_are_disjoint_deterministic_and_proposal_complete() -> None:
    selection = _load(ROOT / "SELECTION_MANIFEST.json")
    summaries: dict[str, dict[str, int]] = {}
    fingerprints: set[str] = set()
    for registration in SPLITS:
        scenes = build_split(registration.split)
        assert len(scenes) == registration.scene_count
        summary = proposal_summary(scenes)
        summaries[registration.split] = summary
        fingerprint = split_fingerprint(scenes)
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        assert summary["positive_proposal_count"] == summary["truth_region_count"]
    assert summaries["train"] == {key: selection["train"][key] for key in summaries["train"]}
    assert summaries["validation"] == {key: selection["validation"][key] for key in summaries["validation"]}


def test_encoding_and_separated_model_contract_are_fixed() -> None:
    scene = build_split("validation")[0]
    candidate = proposals(scene.raster)[0]
    encoded = encode_proposal(scene.raster, candidate)
    assert encoded.shape == (2, 32, 140)
    assert encoded.dtype == np.float32
    assert np.array_equal(encoded[0, 0, 128:], encoded[1, 31, 128:])
    model = ComponentFusionNet()
    model.eval()
    value = torch.from_numpy(np.stack((encoded, encoded.copy())).astype(np.float32))
    value[1, :, :, 128:] = 0
    with torch.inference_mode():
        output = model(value)
    assert tuple(output.shape) == (2, 2)
    assert not torch.equal(output[0], output[1])


def test_balanced_sampler_is_deterministic_and_retains_both_classes() -> None:
    labels = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.int64)
    first = _balanced_order(labels, torch.Generator().manual_seed(17))
    second = _balanced_order(labels, torch.Generator().manual_seed(17))
    assert torch.equal(first, second)
    assert len(first) == 8
    selected = labels.index_select(0, first)
    assert int((selected == 0).sum()) == int((selected == 1).sum()) == 4
    assert set(first.tolist()) >= set(range(len(labels)))


def test_p2_fixed_pool_exports_dynamic_cpu_onnx(tmp_path: Path) -> None:
    model = ComponentFusionP2Net().eval()
    path = tmp_path / "p2-preflight.onnx"
    _export(model, torch.zeros((8, 2, 32, 140), dtype=torch.float32), path)
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"region_proposals": np.zeros((3, 2, 32, 140), dtype=np.float32)})[0]
    assert output.shape == (3, 2)


def test_preregistered_hashes_bind_runner_splits_and_single_candidate() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    selection = _load(ROOT / "SELECTION_MANIFEST.json")
    seal = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    config = _load(REPO_ROOT / CONFIG_PATH)
    ledger = _load(LEDGER_PATH)
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    expected_protocol = json.loads(json.dumps(protocol_configuration()))
    expected_protocol["split_generator_source_paths"] = protocol["split_generator_source_paths"]
    expected_protocol["split_generator_source_bundle_sha256"] = protocol["split_generator_source_bundle_sha256"]
    assert protocol == expected_protocol
    assert selection["sealed_public_truth_available_to_training"] is False
    assert seal["truth_hidden_from_training_runner"] is True
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    assert entry["status"] == "candidate_2_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P2"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["execution_authorized"] is True
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_archive_opened"] is False


def test_preregistration_cannot_be_discovered_as_a_production_model() -> None:
    p1 = _load(ROOT / "P1_RESULT.json")
    assert p1["status"] == "failed_runner"
    assert p1["optimizer_steps"] == 0
    assert p1["public_gate_archive_opened"] is False
    assert not (ROOT / "P2_RESULT.json").exists()
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*component*fusion*v8*.json"))
    index = _load(REPO_ROOT / "artifacts/production-model-store/production-model-index.json")
    serialized = json.dumps(index, sort_keys=True)
    assert "graph-text-component-fusion-v8" not in serialized
    assert "61cdd4661d4bbd1878c32f4e9642836ebb5f8712cff0e33c4d527bcba940b5cd" not in serialized
