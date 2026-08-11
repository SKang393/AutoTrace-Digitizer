# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.graph_text_detector_v1 import dataset, protocol
from ml.ocr.graph_text_detector_v1.model import GraphTextRegionNet
from ml.ocr.graph_text_detector_v1.train_p1 import RUNNER_SOURCE_PATHS


ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_fail_closed_and_fixes_db_thresholds() -> None:
    value = protocol.protocol_configuration()
    assert value["production_approval"] is False
    assert value["release_eligible"] is False
    assert value["experiment_budget"] == 3
    assert value["postprocessing"] == {
        "algorithm": "db_postprocess_v1",
        "score_mode": "fast",
        "probability_threshold": 0.30,
        "box_confidence_threshold": 0.60,
        "unclip_ratio": 1.5,
        "minimum_side_length": 3,
        "maximum_regions": 1000,
    }


def test_training_renderer_is_deterministic_and_excludes_private_labels() -> None:
    first = dataset.render_training_patch(17)
    repeated = dataset.render_training_patch(17)
    different = dataset.render_training_patch(18)
    assert np.array_equal(first.bgr, repeated.bgr)
    assert np.array_equal(first.target, repeated.target)
    assert not np.array_equal(first.bgr, different.bgr)
    assert first.bgr.shape == (protocol.PATCH_HEIGHT, protocol.PATCH_WIDTH, 3)
    assert first.target.shape == (protocol.PATCH_HEIGHT, protocol.PATCH_WIDTH)
    assert "Chandler" not in dataset.GENERIC_TEXT
    assert "Generalization" not in dataset.GENERIC_TEXT


def test_validation_split_is_fixed_and_distinct_from_diagnostic_dimensions() -> None:
    frames = dataset.build_validation_split()
    assert len(frames) == protocol.VALIDATION_TEXT_COUNT + protocol.VALIDATION_EXCLUSION_COUNT
    assert sum(frame.kind == "text" for frame in frames) == protocol.VALIDATION_TEXT_COUNT
    assert sum(frame.kind == "exclusion" for frame in frames) == protocol.VALIDATION_EXCLUSION_COUNT
    assert all(len(frame.detector_bgr) == dataset.FRAME_WIDTH * dataset.FRAME_HEIGHT * 3 for frame in frames)
    assert (dataset.FRAME_WIDTH, dataset.FRAME_HEIGHT) != (384, 192) or all(
        frame.renderer_family != "offset-lattice-graph-diagnostic-v1" for frame in frames
    )


def test_model_returns_same_size_finite_probabilities() -> None:
    model = GraphTextRegionNet().eval()
    with torch.inference_mode():
        output = model(torch.zeros((2, 3, 128, 256), dtype=torch.float32))
    assert output.shape == (2, 1, 128, 256)
    assert torch.isfinite(output).all()
    assert torch.all((output >= 0) & (output <= 1))


def test_model_rejects_non_multiple_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible by eight"):
        GraphTextRegionNet()(torch.zeros((1, 3, 127, 256), dtype=torch.float32))


def test_frozen_metadata_remains_unapproved_when_present() -> None:
    for relative in (
        "ml/ocr/graph_text_detector_v1/PROTOCOL.json",
        "ml/ocr/graph_text_detector_v1/SELECTION_MANIFEST.json",
        "ml/ocr/graph_text_detector_v1/SEALED_PUBLIC_TEST_SEAL.json",
        "ml/ocr/graph_text_detector_v1/training/p1.json",
    ):
        path = ROOT / relative
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            assert value.get("production_approval", value.get("public_release_eligible")) is False


def test_canonical_budget_authorizes_only_the_frozen_p1_candidate() -> None:
    ledger = json.loads(
        (ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entries = [
        item
        for item in ledger["revisions"]
        if item["task"] == protocol.TASK and item["revision"] == protocol.REVISION
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["execution_authorized"] is True
    assert entry["public_gate_authorized"] is False
    config_path = Path(entry["candidate_config_paths"]["P1"])
    assert sha256_file(ROOT / config_path) == entry["candidate_config_sha256"]["P1"]
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS) == entry["expected_runner_source_bundle_sha256"]
    assert sha256_file(ROOT / entry["trigger_evidence_path"]) == entry["trigger_evidence_sha256"]
