# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.component_geometric_v4.dataset import (
    build_split,
    isolate_glyphs,
    load_sealed_public_archive,
    save_sealed_public_archive,
    split_fingerprint,
)
from ml.ocr.component_geometric_v4.model import ComponentGeometricGlyphNet
from ml.ocr.component_geometric_v4.p2_dataset import isolate_glyphs_absolute_scale
from ml.ocr.component_geometric_v4.p2_pipeline import glyph_training_examples as p2_glyph_training_examples
from ml.ocr.component_geometric_v4.pipeline import glyph_training_examples
from ml.ocr.component_geometric_v4.prepare_split import freeze_split
from ml.ocr.component_geometric_v4.protocol import (
    ALPHABET,
    CLASS_COUNT,
    SPLITS,
    protocol_configuration,
)
from ml.ocr.component_geometric_v4.sealed_gate import EVALUATOR_SOURCE_PATHS
from ml.ocr.component_geometric_v4.train_p1 import RUNNER_SOURCE_PATHS
from ml.ocr.component_geometric_v4.train_p2 import RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS


REPOSITORY = Path(__file__).resolve().parents[4]


def test_registered_fonts_and_procedural_splits_are_checksum_bound_and_deterministic() -> None:
    for registration in SPLITS:
        assert sha256_file(REPOSITORY / registration.font_path) == registration.font_sha256
        first = build_split(registration.split)  # type: ignore[arg-type]
        second = build_split(registration.split)  # type: ignore[arg-type]
        assert split_fingerprint(first) == split_fingerprint(second)
        positives = [sample for sample in first if sample.target_text]
        assert all(len(isolate_glyphs(sample.raster)) == len(sample.target_text) for sample in positives)
        assert all(set(sample.target_text) <= set(ALPHABET) for sample in positives)


def test_model_is_non_convolutional_and_accepts_dynamic_glyph_counts() -> None:
    model = ComponentGeometricGlyphNet().eval()
    assert not any(isinstance(module, nn.Conv2d) for module in model.modules())
    for count in (1, 7, 19):
        output = model(torch.zeros(count, 1, 24, 20))
        assert output.shape == (count, CLASS_COUNT)
        assert torch.isfinite(output).all()


def test_exclusion_shapes_are_labeled_only_as_reject_class() -> None:
    samples = tuple(sample for sample in build_split("validation") if sample.exclusion_kind is not None)
    _, labels = glyph_training_examples(samples)
    assert len(labels) > 0
    assert np.unique(labels).tolist() == [CLASS_COUNT - 1]


def test_p2_preserves_absolute_vertical_scale_without_changing_isolation_counts() -> None:
    for split in ("train", "validation"):
        positives = [sample for sample in build_split(split) if sample.target_text]
        assert all(
            len(isolate_glyphs_absolute_scale(sample.raster)) == len(sample.target_text)
            for sample in positives
        )
    validation = build_split("validation")
    decimal = next(sample for sample in validation if "." in sample.target_text)
    divider = next(sample for sample in validation if sample.exclusion_kind == "divider")
    decimal_glyphs = isolate_glyphs_absolute_scale(decimal.raster)
    divider_glyphs = isolate_glyphs_absolute_scale(divider.raster)
    dot_index = decimal.target_text.index(".")
    assert np.count_nonzero(decimal_glyphs[dot_index] > 0.25) < np.count_nonzero(divider_glyphs[0] > 0.25)
    _, exclusion_labels = p2_glyph_training_examples(
        sample for sample in validation if sample.exclusion_kind is not None
    )
    assert np.unique(exclusion_labels).tolist() == [CLASS_COUNT - 1]


def test_sealed_archive_is_byte_deterministic_and_round_trips(tmp_path: Path) -> None:
    samples = build_split("sealed_public")
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    left_manifest = save_sealed_public_archive(samples, left)
    right_manifest = save_sealed_public_archive(samples, right)
    assert left.read_bytes() == right.read_bytes()
    assert left_manifest == right_manifest
    loaded = load_sealed_public_archive(left)
    assert split_fingerprint(loaded) == split_fingerprint(samples)


def test_temp_freeze_binds_sources_and_keeps_public_gate_closed(tmp_path: Path) -> None:
    result = freeze_split(
        private_root=tmp_path / "private",
        protocol_path=tmp_path / "PROTOCOL.json",
        selection_path=tmp_path / "SELECTION_MANIFEST.json",
        seal_path=tmp_path / "SEALED_PUBLIC_TEST_SEAL.json",
        gate_path=tmp_path / "gates/sealed-public-v1.json",
        training_path=tmp_path / "training/p1.json",
    )
    protocol = json.loads((tmp_path / "PROTOCOL.json").read_text(encoding="utf-8"))
    selection = json.loads((tmp_path / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    seal = json.loads((tmp_path / "SEALED_PUBLIC_TEST_SEAL.json").read_text(encoding="utf-8"))
    gate = json.loads((tmp_path / "gates/sealed-public-v1.json").read_text(encoding="utf-8"))
    training = json.loads((tmp_path / "training/p1.json").read_text(encoding="utf-8"))
    assert protocol["production_approval"] is False
    assert selection["chandler_included"] is False
    assert seal["truth_hidden_from_training_runner"] is True
    assert gate["evaluation_limit"] == 1 and gate["production_approval"] is False
    assert training["public_gate_evaluations"] == 0 and training["release_eligible"] is False
    assert training["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPOSITORY, RUNNER_SOURCE_PATHS
    )
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPOSITORY, EVALUATOR_SOURCE_PATHS
    )
    assert result["fixture_archive_sha256"] == seal["fixture_archive_sha256"]
    assert protocol_configuration()["state"] == "preregistered_before_training"


def test_canonical_preregistration_records_p1_failure_and_keeps_p2_and_public_gate_closed() -> None:
    p1_result = json.loads(
        (REPOSITORY / "ml/ocr/component_geometric_v4/P1_RESULT.json").read_text(encoding="utf-8")
    )
    assert p1_result["status"] == "failed_selection"
    assert p1_result["sealed_public_archive_opened"] is False
    assert p1_result["public_gate_evaluations"] == 0
    assert not (REPOSITORY / "ml/ocr/component_geometric_v4/artifacts/P2-run").exists()
    tracked = {
        path.relative_to(REPOSITORY).as_posix()
        for path in REPOSITORY.glob("ml/ocr/component_geometric_v4/**/*.json")
        if ".pytest" not in path.as_posix()
    }
    assert "ml/ocr/component_geometric_v4/PROTOCOL.json" in tracked
    assert "ml/ocr/component_geometric_v4/training/p1.json" in tracked
    assert "ml/ocr/component_geometric_v4/training/p2.json" in tracked
    assert not any("PUBLIC_GATE_REPORT" in path for path in tracked)


def test_p2_runner_source_bundle_matches_preregistration() -> None:
    training = json.loads(
        (REPOSITORY / "ml/ocr/component_geometric_v4/training/p2.json").read_text(encoding="utf-8")
    )
    assert training["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPOSITORY, P2_RUNNER_SOURCE_PATHS
    )
    assert training["candidate_id"] == "P2"
    assert training["public_gate_evaluations"] == 0
