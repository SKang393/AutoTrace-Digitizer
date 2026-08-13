# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic generator and evaluator tests for OCR V11."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.ocr.composite_proposal_role_v11.dataset import (
    encode_proposal,
    load_sealed_public_archive,
    proposal_summary,
    proposal_targets,
    proposals,
    render_scene,
    save_sealed_public_archive,
    split_fingerprint,
    training_examples,
)
from ml.ocr.composite_proposal_role_v11.pipeline import evaluate_thresholds
from ml.ocr.composite_proposal_role_v11.protocol import ROLE_ORDER


def _scenes(split: str, count: int = 3):
    return tuple(render_scene(split, index) for index in range(count))


def test_rendering_and_position_encoding_are_deterministic() -> None:
    first = _scenes("train")
    second = _scenes("train")
    assert split_fingerprint(first) == split_fingerprint(second)
    proposal = proposals(first[0].raster)[0]
    encoded = encode_proposal(first[0].raster, proposal)
    assert encoded.shape == (2, 32, 144)
    assert np.isfinite(encoded).all()
    expected = np.asarray(
        (
            (proposal.left + proposal.right + 1.0) / (2.0 * first[0].raster.shape[1]),
            (proposal.top + proposal.bottom + 1.0) / (2.0 * first[0].raster.shape[0]),
            proposal.left / first[0].raster.shape[1],
            proposal.top / first[0].raster.shape[0],
        ),
        dtype=np.float32,
    )
    assert np.array_equal(encoded[0, 0, -4:], expected)


def test_every_truth_has_one_proposal_and_roles_are_balanced() -> None:
    scenes = _scenes("validation", 16)
    summary = proposal_summary(scenes)
    assert summary["truth_region_count"] == 16 * len(ROLE_ORDER)
    assert summary["positive_proposal_count"] == summary["truth_region_count"]
    assert summary["negative_proposal_count"] > summary["positive_proposal_count"]
    assert summary["role_truth_counts"] == {role: 16 for role in ROLE_ORDER}


def test_training_examples_are_capped_and_never_use_gate_pixels() -> None:
    scenes = _scenes("train", 4)
    values, proposals_labels, role_labels, evidence = training_examples(scenes, negative_cap_per_scene=7)
    assert values.shape[1:] == (2, 32, 144)
    assert len(values) == len(proposals_labels) == len(role_labels)
    assert evidence["positive_proposal_count"] == 4 * len(ROLE_ORDER)
    assert evidence["negative_proposal_count"] <= 4 * 7
    assert evidence["validation_or_public_pixels_used"] is False
    assert evidence["v2_bytes_used"] is False
    assert np.all(role_labels[proposals_labels == 0] == -1)


def test_perfect_multitask_outputs_pass_exact_evaluator() -> None:
    scenes = _scenes("validation", 5)
    expected = []
    for scene in scenes:
        candidates = proposals(scene.raster)
        accepted, roles = proposal_targets(scene, candidates)
        output = np.full((len(candidates), 2 + len(ROLE_ORDER)), -9.0, dtype=np.float32)
        output[:, 0] = np.where(accepted == 0, 9.0, -9.0)
        output[:, 1] = np.where(accepted == 1, 9.0, -9.0)
        for index, role in enumerate(roles):
            if role >= 0:
                output[index, 2 + role] = 9.0
        expected.append(output)
    iterator = iter(expected)
    metrics = evaluate_thresholds(scenes, lambda _values: next(iterator), (0.95,))[0]["metrics"]
    assert metrics["exact_scene_count"] == metrics["scene_count"] == 5
    assert metrics["false_positives"] == metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
    assert metrics["role_accuracy"] == 1.0
    assert set(metrics["per_role_accuracy"].values()) == {1.0}


def test_sealed_archive_binds_png_and_truth_bytes(tmp_path: Path) -> None:
    scenes = _scenes("sealed_public", 3)
    archive = tmp_path / "fixtures.zip"
    private = save_sealed_public_archive(scenes, archive)
    loaded = load_sealed_public_archive(archive)
    assert split_fingerprint(loaded) == split_fingerprint(scenes)
    assert private["fixture_archive_sha256"]
    assert private["v2_bytes_used"] is False

