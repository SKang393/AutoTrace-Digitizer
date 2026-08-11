# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed checks for the preregistered ignore-band detector."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.markers.gate_seal import source_bundle_sha256
from ml.ocr.graph_text_ignore_band_v3.dataset import (
    GENERIC_TEXT,
    build_validation_split,
    render_training_patch,
    split_fingerprint,
    training_split_fingerprint,
)
from ml.ocr.graph_text_ignore_band_v3.model import IgnoreBandTextRegionNet
from ml.ocr.graph_text_ignore_band_v3.prepare_split import freeze_split
from ml.ocr.graph_text_ignore_band_v3.protocol import (
    EMPTY_TARGET_NEGATIVE_PIXELS,
    EXPERIMENT_BUDGET,
    IGNORE_BAND_EXPANSION_PIXELS,
    NEGATIVE_TO_POSITIVE_RATIO,
    REVISION,
    SPLITS,
    TRAIN_SAMPLE_COUNT,
    VALIDATION_EXCLUSION_COUNT,
    VALIDATION_TEXT_COUNT,
)
from ml.ocr.graph_text_ignore_band_v3.train_p1 import RUNNER_SOURCE_PATHS, _loss


REPO_ROOT = Path(__file__).resolve().parents[4]
REVISION_ROOT = REPO_ROOT / "ml/ocr/graph_text_ignore_band_v3"
LEDGER_PATH = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_protocol_freezes_distinct_fail_closed_defect_class() -> None:
    protocol = _json(REVISION_ROOT / "PROTOCOL.json")
    assert protocol["revision"] == REVISION == "graph-text-ignore-band-v3"
    assert protocol["experiment_budget"] == EXPERIMENT_BUDGET == 3
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["trigger"]["prior_revision"] == "graph-text-balanced-recall-v2"
    assert protocol["trigger"]["prior_exact_fixture_count"] == 30
    assert protocol["training"]["ignore_band_expansion_pixels"] == IGNORE_BAND_EXPANSION_PIXELS == 1
    assert protocol["training"]["negative_to_positive_ratio"] == NEGATIVE_TO_POSITIVE_RATIO == 3
    assert protocol["training"]["empty_target_negative_pixels"] == EMPTY_TARGET_NEGATIVE_PIXELS == 4096
    assert protocol["postprocessing"]["probability_threshold"] == 0.30
    assert protocol["postprocessing"]["box_confidence_threshold"] == 0.60
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3


def test_renderer_is_deterministic_and_ignore_band_is_explicit() -> None:
    first = render_training_patch(19)
    repeated = render_training_patch(19)
    exclusion = render_training_patch(TRAIN_SAMPLE_COUNT - 1)
    assert first.kind == "text"
    assert np.array_equal(first.bgr, repeated.bgr)
    assert np.array_equal(first.target, repeated.target)
    assert np.array_equal(first.supervision_mask, repeated.supervision_mask)
    assert np.count_nonzero(first.target) > 0
    assert np.count_nonzero(first.supervision_mask == 0) > 0
    assert np.all(first.supervision_mask[first.target > 0] == 255)
    assert exclusion.kind == "exclusion"
    assert np.count_nonzero(exclusion.target) == 0
    assert np.all(exclusion.supervision_mask == 255)
    assert all(value.casefold() not in {"chandler", "generalization"} for value in GENERIC_TEXT)


def test_validation_split_is_frozen_and_disjoint_from_v2() -> None:
    selection = _json(REVISION_ROOT / "SELECTION_MANIFEST.json")
    validation = build_validation_split()
    assert len(validation) == VALIDATION_TEXT_COUNT + VALIDATION_EXCLUSION_COUNT == 112
    assert split_fingerprint(validation) == selection["validation_split_fingerprint"]
    assert training_split_fingerprint() == selection["train_split_fingerprint"]
    prior = _json(REPO_ROOT / "ml/ocr/graph_text_balanced_v2/SELECTION_MANIFEST.json")
    assert selection["train_split_fingerprint"] != prior["train_split_fingerprint"]
    assert selection["validation_split_fingerprint"] != prior["validation_split_fingerprint"]
    assert selection["prior_selection_fixture_reused"] is False
    assert selection["prior_sealed_fixture_reused"] is False
    assert selection["chandler_included"] is False
    assert selection["generalization_label_included"] is False


def test_masked_ohem_loss_ignores_unsupervised_glyph_halo() -> None:
    probabilities = torch.full((2, 1, 16, 16), 0.2, dtype=torch.float32)
    target = torch.zeros_like(probabilities)
    target[0, :, 6:10, 6:10] = 1.0
    supervision = torch.ones_like(probabilities)
    supervision[0, :, 4:12, 4:12] = 0.0
    supervision[0, :, 6:10, 6:10] = 1.0
    first = _loss(probabilities, target, supervision)
    changed = probabilities.clone()
    changed[0, :, 4:6, 4:12] = 0.99
    changed[0, :, 10:12, 4:12] = 0.99
    second = _loss(changed, target, supervision)
    assert all(torch.isfinite(value) for value in first)
    assert all(torch.isfinite(value) for value in second)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_model_preserves_probability_map_shape_and_input_contract() -> None:
    model = IgnoreBandTextRegionNet(seed=20260907).eval()
    values = torch.zeros((2, 3, 64, 128), dtype=torch.float32)
    with torch.inference_mode():
        output = model(values)
    assert output.shape == (2, 1, 64, 128)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    with pytest.raises(ValueError, match=r"\[batch,3,H,W\]"):
        model(torch.zeros((1, 1, 64, 128)))
    with pytest.raises(ValueError, match="divisible by eight"):
        model(torch.zeros((1, 3, 63, 128)))


def test_frozen_hashes_and_ledger_authorize_only_unused_p1() -> None:
    expected_hashes = {
        "PROTOCOL.json": "257f791a8e125aa81ea6d6096644f27582aeea7a7153300075a5853dc3a4c421",
        "SELECTION_MANIFEST.json": "2c18959408cf7dc797287aa836c37d7020919c9a1a481800a125c26e160ae505",
        "SEALED_PUBLIC_TEST_SEAL.json": "9acd17d3c5cc2f4a631a0a2687c6ccfbbcca35c0b03eb635f5165d924142d966",
        "P1_PREREGISTRATION.json": "493239e1323c45ce6b9e075722d465ab612618a3b036f65a2d877e975d5370a9",
        "training/p1.json": "29f58aca0d700b93bc7979c3c5f28ba8e7b50decafafdb54c6c2128121437fe1",
    }
    for name, expected in expected_hashes.items():
        assert _sha256(REVISION_ROOT / name) == expected
    config = _json(REVISION_ROOT / "training/p1.json")
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]
    seal = _json(REVISION_ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    archive = REPO_ROOT / str(seal["fixture_archive_path"])
    assert _sha256(archive) == seal["fixture_archive_sha256"] == "1b23558516d72b2241501cd24ac0019ace01ba117e725aa9ad030c5e78e59a95"
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
    assert entry["public_gate_authorized"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_freeze_refuses_to_overwrite_any_existing_evidence(tmp_path: Path) -> None:
    occupied = tmp_path / "protocol.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refuses to overwrite evidence"):
        freeze_split(
            private_root=tmp_path.relative_to(REPO_ROOT) if tmp_path.is_relative_to(REPO_ROOT) else Path("ml/ocr/graph_text_ignore_band_v3/artifacts/split-freeze"),
            protocol_path=occupied.relative_to(REPO_ROOT) if occupied.is_relative_to(REPO_ROOT) else Path("ml/ocr/graph_text_ignore_band_v3/PROTOCOL.json"),
            selection_path=Path("ml/ocr/graph_text_ignore_band_v3/SELECTION_MANIFEST.json"),
            seal_path=Path("ml/ocr/graph_text_ignore_band_v3/SEALED_PUBLIC_TEST_SEAL.json"),
            preregistration_path=Path("ml/ocr/graph_text_ignore_band_v3/P1_PREREGISTRATION.json"),
            training_path=Path("ml/ocr/graph_text_ignore_band_v3/training/p1.json"),
            trigger_path=Path("ml/ocr/graph_text_balanced_v2/P3_RESULT.json"),
        )
