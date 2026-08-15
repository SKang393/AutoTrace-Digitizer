# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate-only fixture-feasibility diagnosis for the exhausted V5 selection split."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ml.markers.center.dense_contract_v5.dataset import read_archive
from ml.markers.center.dense_contract_v5.train_p1 import (
    HARD_NEGATIVE_TOLERANCE,
    MATCH_TOLERANCE,
    REPO_ROOT,
    REVISION,
)
from ml.markers.gate_seal import sha256_file


ROOT = REPO_ROOT / "ml/markers/center/dense_contract_v5"
SELECTION_MANIFEST_PATH = ROOT / "SELECTION_MANIFEST.json"


def diagnose() -> dict[str, object]:
    selection = json.loads(SELECTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / selection["validation"]["archive_path"]
    if sha256_file(archive_path) != selection["validation"]["archive_sha256"]:
        raise RuntimeError("Dense-contract V5 visible validation archive changed")
    archive = read_archive(archive_path)
    overlap_tolerance = MATCH_TOLERANCE + HARD_NEGATIVE_TOLERANCE
    truth_count = 0
    hard_negative_count = 0
    truth_hard_pair_count = 0
    exact_center_conflicts = 0
    exact_center_conflict_scenes = 0
    acceptance_radius_overlaps = 0
    acceptance_radius_overlap_scenes = 0
    artifact_truth_cleared_hard_points = 0
    artifact_truth_cleared_scenes = 0
    minimum_distance = math.inf
    for scene_index in range(int(archive["center_counts"].shape[0])):
        center_count = int(archive["center_counts"][scene_index])
        hard_count = int(archive["hard_counts"][scene_index])
        centers = archive["centers"][scene_index, :center_count, :2]
        hard_points = archive["hard_points"][scene_index, :hard_count]
        truth_count += center_count
        hard_negative_count += hard_count
        truth_hard_pair_count += center_count * hard_count
        distances = np.sqrt(((centers[:, None, :] - hard_points[None, :, :]) ** 2).sum(axis=2))
        minimum_distance = min(minimum_distance, float(distances.min()))
        exact_count = int((distances <= HARD_NEGATIVE_TOLERANCE).sum())
        overlap_count = int((distances <= overlap_tolerance).sum())
        exact_center_conflicts += exact_count
        exact_center_conflict_scenes += int(exact_count > 0)
        acceptance_radius_overlaps += overlap_count
        acceptance_radius_overlap_scenes += int(overlap_count > 0)
        cleared_count = 0
        for point in hard_points:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            cleared_count += int(float(archive["artifact_targets"][scene_index, 0, y, x]) < 0.5)
        artifact_truth_cleared_hard_points += cleared_count
        artifact_truth_cleared_scenes += int(cleared_count > 0)
    return {
        "schema": "graphreader.marker-center-fixture-feasibility-diagnosis.v1",
        "task": "marker-center",
        "revision": REVISION,
        "scope": "visible_selection_aggregate_only",
        "selection_manifest_path": SELECTION_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
        "selection_manifest_sha256": sha256_file(SELECTION_MANIFEST_PATH),
        "validation_archive_path": archive_path.relative_to(REPO_ROOT).as_posix(),
        "validation_archive_sha256": sha256_file(archive_path),
        "scene_count": int(archive["center_counts"].shape[0]),
        "truth_center_count": truth_count,
        "hard_negative_count": hard_negative_count,
        "truth_hard_pair_count": truth_hard_pair_count,
        "matching_tolerance_px": MATCH_TOLERANCE,
        "prohibited_hit_tolerance_px": HARD_NEGATIVE_TOLERANCE,
        "required_disjoint_clearance_px": overlap_tolerance,
        "minimum_truth_to_hard_negative_distance_px": minimum_distance,
        "exact_center_conflict_pair_count": exact_center_conflicts,
        "exact_center_conflict_scene_count": exact_center_conflict_scenes,
        "acceptance_radius_overlap_pair_count": acceptance_radius_overlaps,
        "acceptance_radius_overlap_scene_count": acceptance_radius_overlap_scenes,
        "artifact_truth_cleared_hard_point_count": artifact_truth_cleared_hard_points,
        "artifact_truth_cleared_scene_count": artifact_truth_cleared_scenes,
        "case_level_details_emitted": False,
        "fixture_pixels_emitted": False,
        "private_data": False,
        "chandler_used": False,
        "public_archive_opened": False,
        "public_gate_evaluations": 0,
        "v5_rerun_authorized": False,
        "new_defect_class_required": True,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = ["diagnose"]
