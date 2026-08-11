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
from ml.ocr.graph_text_stride4_v4.model_p2 import Stride4ResizeConvTextRegionNet
from ml.ocr.graph_text_stride4_v4.model_p3 import Stride4FullResolutionRefinementNet
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
from ml.ocr.graph_text_stride4_v4.train_p1 import RUNNER_SOURCE_PATHS as P1_RUNNER_SOURCE_PATHS, _p3_loss
from ml.ocr.graph_text_stride4_v4.train_p2 import RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS
from ml.ocr.graph_text_stride4_v4.train_p3 import RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS


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


def test_p2_changes_only_to_resize_convolution_upsampling() -> None:
    model = Stride4ResizeConvTextRegionNet().eval()
    values = torch.zeros((2, 3, 64, 128), dtype=torch.float32)
    with torch.inference_mode():
        output = model(values)
        level1 = model.encoder1(values)
        level2 = model.encoder2(level1)
    assert output.shape == (2, 1, 64, 128)
    assert level1.shape[-2:] == (32, 64)
    assert level2.shape[-2:] == (16, 32)
    assert isinstance(model.decoder, torch.nn.Conv2d)
    assert isinstance(model.output, torch.nn.Conv2d)
    assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules())
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    with pytest.raises(ValueError, match=r"\[batch,3,H,W\]"):
        model(torch.zeros((1, 1, 64, 128)))
    with pytest.raises(ValueError, match="divisible by four"):
        model(torch.zeros((1, 3, 63, 128)))


def test_p3_adds_only_full_resolution_input_detail_fusion() -> None:
    model = Stride4FullResolutionRefinementNet().eval()
    values = torch.zeros((2, 3, 64, 128), dtype=torch.float32)
    with torch.inference_mode():
        output = model(values)
        level1 = model.encoder1(values)
        level2 = model.encoder2(level1)
        input_detail = model.input_detail(values)
    assert output.shape == (2, 1, 64, 128)
    assert level1.shape[-2:] == (32, 64)
    assert level2.shape[-2:] == (16, 32)
    assert input_detail.shape == (2, 8, 64, 128)
    assert model.fusion.in_channels == 20
    assert isinstance(model.decoder, torch.nn.Conv2d)
    assert isinstance(model.output, torch.nn.Conv2d)
    assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules())
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


def test_frozen_hashes_and_ledger_record_exhausted_v4_without_opening_public_gate() -> None:
    expected_hashes = {
        "PROTOCOL.json": "8e205a7f6cfc2252294948cfb576b6045421f338580dc0532165cc97371b0bd0",
        "SELECTION_MANIFEST.json": "2b839e9775082aa04eac6a4d34fcf9532b2013f7d107e5f1c18501e375886aeb",
        "SEALED_PUBLIC_TEST_SEAL.json": "83771a24f66740ef201550a7bb7a74ed9b8710ba42a4678f4c178311f0cb295b",
        "P1_PREREGISTRATION.json": "ab42cedac1bb1292fc71749837f459227e2ed66a8f673cccc16317fb3ce5c803",
        "P1_RESULT.json": "545211c5dcc62d1e9590e7c54c084c67750c30eb8077f1054fdf24089edbdf50",
        "P2_PREREGISTRATION.json": "8027a7ab0a131577c23cc5eafa9cf97d1176ac1a9255b6857055bfe732a32138",
        "P2_RESULT.json": "7e9053717af8a47e5170dafc3b88669a2ab24e010e6226db9d770cdafe9cf80f",
        "P3_PREREGISTRATION.json": "c5c8fb1d0b0181c86fa375e78a58f48efae09abf60f9e1fe5c25d7e6605eb21f",
        "P3_RESULT.json": "eba985a812d08788230622adb1623e5f1e50e556c85e123512e1cac05e26d410",
        "training/p1.json": "71eabe488cbbfb0b605986feb944017a8fd37e9abbab71b221a026d00c5d479e",
        "training/p2.json": "7f0d93326ff57078cc0ca1442ed8b335542cd30e4e3d30da22feb22cf2f492f5",
        "training/p3.json": "7392076bd4c51683ec8ac3544fc30e8a43531c41d43c6e1fccf8e610341619e6",
    }
    for name, expected in expected_hashes.items():
        assert _sha256(REVISION_ROOT / name) == expected
    protocol = _json(REVISION_ROOT / "PROTOCOL.json")
    assert source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS) == protocol["split_generator_source_bundle_sha256"]
    p1_config = _json(REVISION_ROOT / "training/p1.json")
    p2_config = _json(REVISION_ROOT / "training/p2.json")
    p3_config = _json(REVISION_ROOT / "training/p3.json")
    assert source_bundle_sha256(REPO_ROOT, P1_RUNNER_SOURCE_PATHS) == p1_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P2_RUNNER_SOURCE_PATHS) == p2_config["expected_runner_source_bundle_sha256"]
    assert source_bundle_sha256(REPO_ROOT, P3_RUNNER_SOURCE_PATHS) == p3_config["expected_runner_source_bundle_sha256"]
    p2_result = _json(REVISION_ROOT / "P2_RESULT.json")
    assert p2_result["selection_report_sha256"] == "04a4c6aa6f541dcba1f95c1aac2e02ad07dde3d90e2b742a5f842f11322359f4"
    assert p2_result["diagnosis_sha256"] == "7f1609ae8d38e27f4eb44f5057d6e346dc8a9919eaec4a71c69f263289a21cfc"
    assert p2_result["diagnosis"]["diagnostic_runs"] == 1
    assert p2_result["diagnosis"]["threshold_sweeps"] == 0
    assert p2_result["sealed_public_archive_opened"] is False
    assert p2_result["public_gate_evaluations"] == 0
    assert p2_result["production_approval"] is False
    p2_seal_root = REPO_ROOT / "ml/markers/training-seals/ocr-detection/graph-text-stride4-v4/P2"
    assert _sha256(p2_seal_root / "opened.json") == "24661d855572e9d998131e017b49d15a95a32575937e4e1800660c2e58b91b01"
    assert _sha256(p2_seal_root / "result.json") == "3bda7660218eff7903d21374c8edf39c20acfe5b74264c14d5decdb2e28f5fe8"
    p3_preregistration = _json(REVISION_ROOT / "P3_PREREGISTRATION.json")
    assert p3_preregistration["p2_result_sha256"] == expected_hashes["P2_RESULT.json"]
    assert p3_preregistration["p2_diagnosis_sha256"] == p2_result["diagnosis_sha256"]
    assert p3_preregistration["public_gate_authorized"] is False
    assert p3_preregistration["sealed_public_archive_opened"] is False
    p3_result = _json(REVISION_ROOT / "P3_RESULT.json")
    assert p3_result["selection_report_sha256"] == "e17abd79d869d0912c3b1537e4c84850fc2b4e921fb4b52f0e3afdc0ce443bdb"
    assert p3_result["selection_metrics"]["exact_fixture_count"] == 82
    assert p3_result["selection_metrics"]["text_missed_fixture_count"] == 50
    assert p3_result["public_gate_evaluations"] == 0
    assert p3_result["sealed_public_archive_opened"] is False
    assert p3_result["production_approval"] is False
    p3_seal_root = REPO_ROOT / "ml/markers/training-seals/ocr-detection/graph-text-stride4-v4/P3"
    assert _sha256(p3_seal_root / "opened.json") == "4709dd28d90c698d14e127f22fd4822c459105a865423c9d7cae575f730241f3"
    assert _sha256(p3_seal_root / "result.json") == "00f3ae0fa7d4d2ecba8fcbe68a6e604e6e8a74a84586c513792e13343a56d660"
    seal = _json(REVISION_ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert seal["truth_hidden_from_training_runner"] is True
    assert seal["public_release_eligible"] is False
    assert _sha256(REPO_ROOT / str(seal["fixture_archive_path"])) == seal["fixture_archive_sha256"] == "b6498f2b37bfce66878c2fcba1da1764ec07323022e6d173f215d1ed039edd1b"
    ledger = _json(LEDGER_PATH)
    entries = [entry for entry in ledger["revisions"] if entry["revision"] == REVISION]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "exhausted_failed_selection"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert "three-candidate budget is exhausted" in entry["execution_blocker"]
    assert entry["candidate_config_sha256"]["P1"] == expected_hashes["training/p1.json"]
    assert entry["candidate_config_sha256"]["P2"] == expected_hashes["training/p2.json"]
    assert entry["candidate_config_sha256"]["P3"] == expected_hashes["training/p3.json"]
    assert entry["p1_training_report_sha256"] == "0390c78e06f54f0808495da9202f7cb558e4184493dd8f0d568cc645fe15948d"
    assert entry["p1_selection_exact_fixture_count"] == 101
    assert entry["p1_selection_false_region_count"] == 14
    assert entry["p1_selection_exclusion_false_region_count"] == 0
    assert entry["p1_text_missed_fixture_count"] == 35
    assert entry["p1_selection_gate_passed"] is False
    assert entry["p2_training_report_sha256"] == "04a4c6aa6f541dcba1f95c1aac2e02ad07dde3d90e2b742a5f842f11322359f4"
    assert entry["p2_selection_exact_fixture_count"] == 108
    assert entry["p2_selection_false_region_count"] == 15
    assert entry["p2_selection_exclusion_false_region_count"] == 1
    assert entry["p2_text_missed_fixture_count"] == 24
    assert entry["p2_text_multi_region_fixture_count"] == 3
    assert entry["p2_diagnostic_runs"] == 1
    assert entry["p2_threshold_sweeps"] == 0
    assert entry["p2_selection_gate_passed"] is False
    assert entry["p3_training_report_sha256"] == "e17abd79d869d0912c3b1537e4c84850fc2b4e921fb4b52f0e3afdc0ce443bdb"
    assert entry["p3_selection_exact_fixture_count"] == 82
    assert entry["p3_selection_false_region_count"] == 39
    assert entry["p3_selection_exclusion_false_region_count"] == 1
    assert entry["p3_text_missed_fixture_count"] == 50
    assert entry["p3_text_multi_region_fixture_count"] == 8
    assert entry["p3_selection_gate_passed"] is False
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
