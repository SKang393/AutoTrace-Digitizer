# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed checks for the preregistered stride-4 detector."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from ml.markers.gate_seal import source_bundle_sha256
from ml.ocr.graph_text_ignore_band_v3.dataset import build_validation_split as build_v3_validation_split
from ml.ocr.graph_text_stride4_v4.dataset import (
    GENERIC_TEXT,
    build_validation_split,
    render_training_tiles,
    split_fingerprint,
    training_split_fingerprint,
)
from ml.ocr.graph_text_stride4_v4.model import Stride4TextRegionNet
from ml.ocr.graph_text_stride4_v4.prepare_split import SPLIT_SOURCE_PATHS, freeze_split
from ml.ocr.graph_text_stride4_v4.protocol import (
    BOUNDARY_MARGIN_LOSS_WEIGHT,
    BOUNDARY_PROBABILITY_CEILING,
    EXPERIMENT_BUDGET,
    REVISION,
    SPLITS,
    TILES_PER_SOURCE,
    TRAIN_SAMPLE_COUNT,
    TRAIN_SOURCE_COUNT,
    VALIDATION_EXCLUSION_COUNT,
    VALIDATION_TEXT_COUNT,
    protocol_configuration,
)
from ml.ocr.graph_text_stride4_v4.train_p1 import _p3_loss
from ml.ocr.graph_text_stride4_v4.train_p1 import RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[4]
REVISION_ROOT = REPO_ROOT / "ml/ocr/graph_text_stride4_v4"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_protocol_freezes_distinct_recall_defect_class() -> None:
    protocol = protocol_configuration()
    assert protocol["revision"] == REVISION == "graph-text-stride4-v4"
    assert protocol["experiment_budget"] == EXPERIMENT_BUDGET == 3
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["trigger"]["prior_revision"] == "graph-text-ignore-band-v3"
    assert protocol["trigger"]["prior_exact_fixture_count"] == 92
    assert protocol["trigger"]["prior_text_missed_fixture_count"] == 19
    assert protocol["architecture"] == "fine-skip-stride4-probability-map-v1"
    assert protocol["training"]["sample_count"] == TRAIN_SAMPLE_COUNT == 1920
    assert protocol["training"]["boundary_probability_ceiling"] == BOUNDARY_PROBABILITY_CEILING == 0.25
    assert protocol["training"]["boundary_margin_loss_weight"] == BOUNDARY_MARGIN_LOSS_WEIGHT == 1.0
    assert protocol["postprocessing"]["probability_threshold"] == 0.30
    assert protocol["postprocessing"]["box_confidence_threshold"] == 0.60
    assert protocol["selection_gates"]["false_region_count"] == 0
    assert protocol["selection_gates"]["exclusion_false_region_count"] == 0
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    assert len({item.seed_offset for item in SPLITS}) == 3


def test_renderer_is_deterministic_and_generic() -> None:
    first = render_training_tiles(19)
    repeated = render_training_tiles(19)
    exclusion = render_training_tiles(TRAIN_SOURCE_COUNT - 1)
    assert len(first) == len(repeated) == len(exclusion) == TILES_PER_SOURCE == 3
    for actual, again in zip(first, repeated, strict=True):
        assert actual.tile_id == again.tile_id
        assert actual.left == again.left
        assert actual.top == again.top
        assert np.array_equal(actual.bgr, again.bgr)
        assert np.array_equal(actual.target, again.target)
        assert np.array_equal(actual.supervision_mask, again.supervision_mask)
    assert first[0].kind == "text"
    assert any(np.count_nonzero(tile.target) > 0 for tile in first)
    assert any(np.count_nonzero(tile.supervision_mask == 0) > 0 for tile in first)
    assert all(tile.kind == "exclusion" for tile in exclusion)
    assert all(np.count_nonzero(tile.target) == 0 for tile in exclusion)
    assert all(np.all(tile.supervision_mask == 255) for tile in exclusion)
    assert all(value.casefold() not in {"chandler", "generalization"} for value in GENERIC_TEXT)


def test_validation_is_deterministic_and_disjoint_from_v3_selection() -> None:
    first = build_validation_split()
    repeated = build_validation_split()
    prior = build_v3_validation_split()
    assert len(first) == VALIDATION_TEXT_COUNT + VALIDATION_EXCLUSION_COUNT == 136
    assert split_fingerprint(first) == split_fingerprint(repeated) == "b660c9720f9393b277c54840d0b28fa0ac2c0ca73622d67174cfecf41aae829c"
    assert training_split_fingerprint() == "325ead235c7e15971e57721e89554f7f2568127ca947a5eb6a0c714a6a156134"
    assert {sample.source_sha256 for sample in first}.isdisjoint(sample.source_sha256 for sample in prior)
    assert {sample.detector_bgr_sha256 for sample in first}.isdisjoint(sample.detector_bgr_sha256 for sample in prior)
    assert all(sample.renderer_family.endswith("-v4") for sample in first)
    assert all(":validation_" in sample.degradation_family for sample in first)


def test_model_preserves_full_probability_map_with_fourfold_bottleneck() -> None:
    model = Stride4TextRegionNet().eval()
    values = torch.zeros((2, 3, 64, 128), dtype=torch.float32)
    with torch.inference_mode():
        output = model(values)
        level1 = model.encoder1(values)
        level2 = model.encoder2(level1)
    assert output.shape == (2, 1, 64, 128)
    assert level1.shape[-2:] == (32, 64)
    assert level2.shape[-2:] == (16, 32)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    with pytest.raises(ValueError, match=r"\[batch,3,H,W\]"):
        model(torch.zeros((1, 1, 64, 128)))
    with pytest.raises(ValueError, match="divisible by four"):
        model(torch.zeros((1, 3, 63, 128)))


def test_retained_p3_margin_is_one_sided_and_boundary_only() -> None:
    target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target[:, :, 3:5, 3:5] = 1.0
    supervision = torch.ones_like(target)
    supervision[:, :, 2:6, 2:6] = 0.0
    supervision[target > 0.5] = 1.0
    below = torch.full_like(target, 0.20)
    below[target > 0.5] = 0.80
    at_ceiling = below.clone()
    at_ceiling[supervision <= 0.5] = BOUNDARY_PROBABILITY_CEILING
    above = below.clone()
    above[supervision <= 0.5] = 0.35
    losses = [
        _p3_loss(
            probabilities,
            target,
            supervision,
            boundary_probability_ceiling=BOUNDARY_PROBABILITY_CEILING,
            boundary_margin_loss_weight=BOUNDARY_MARGIN_LOSS_WEIGHT,
        )
        for probabilities in (below, at_ceiling, above)
    ]
    assert float(losses[0][3]) == 0.0
    assert float(losses[1][3]) == 0.0
    assert float(losses[2][3]) > 0.0
    assert torch.equal(losses[0][1], losses[1][1])
    assert torch.equal(losses[1][1], losses[2][1])


def test_frozen_hashes_and_ledger_authorize_only_unused_p1() -> None:
    expected_hashes = {
        "PROTOCOL.json": "8e205a7f6cfc2252294948cfb576b6045421f338580dc0532165cc97371b0bd0",
        "SELECTION_MANIFEST.json": "2b839e9775082aa04eac6a4d34fcf9532b2013f7d107e5f1c18501e375886aeb",
        "SEALED_PUBLIC_TEST_SEAL.json": "83771a24f66740ef201550a7bb7a74ed9b8710ba42a4678f4c178311f0cb295b",
        "P1_PREREGISTRATION.json": "ab42cedac1bb1292fc71749837f459227e2ed66a8f673cccc16317fb3ce5c803",
        "training/p1.json": "71eabe488cbbfb0b605986feb944017a8fd37e9abbab71b221a026d00c5d479e",
    }
    for name, expected in expected_hashes.items():
        assert _sha256(REVISION_ROOT / name) == expected
    protocol = _json(REVISION_ROOT / "PROTOCOL.json")
    assert source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS) == protocol["split_generator_source_bundle_sha256"]
    config = _json(REVISION_ROOT / "training/p1.json")
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    seal = _json(REVISION_ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert seal["truth_hidden_from_training_runner"] is True
    assert seal["public_release_eligible"] is False
    assert _sha256(REPO_ROOT / str(seal["fixture_archive_path"])) == seal["fixture_archive_sha256"] == "b6498f2b37bfce66878c2fcba1da1764ec07323022e6d173f215d1ed039edd1b"
    ledger = _json(LEDGER_PATH)
    entries = [entry for entry in ledger["revisions"] if entry["revision"] == REVISION]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["candidate_config_sha256"]["P1"] == expected_hashes["training/p1.json"]
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_freeze_refuses_to_overwrite_before_reading_sealed_data(tmp_path: Path) -> None:
    root = REPO_ROOT / "ml/ocr/graph_text_stride4_v4/artifacts/test-freeze-refusal" / tmp_path.name
    protocol_path = root.joinpath("protocol.json").relative_to(REPO_ROOT)
    private_root = root.joinpath("private").relative_to(REPO_ROOT)
    root.mkdir(parents=True, exist_ok=False)
    (REPO_ROOT / protocol_path).write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="refuses to overwrite evidence"):
            freeze_split(
                private_root=private_root,
                protocol_path=protocol_path,
                selection_path=root.joinpath("selection.json").relative_to(REPO_ROOT),
                seal_path=root.joinpath("seal.json").relative_to(REPO_ROOT),
                preregistration_path=root.joinpath("preregistration.json").relative_to(REPO_ROOT),
                training_path=root.joinpath("training.json").relative_to(REPO_ROOT),
                trigger_path=Path("ml/ocr/graph_text_ignore_band_v3/P3_RESULT.json"),
            )
    finally:
        shutil.rmtree(root)
