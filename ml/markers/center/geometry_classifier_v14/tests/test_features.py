# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from ml.markers.center.line_aware_v1.pipeline import extract_proposals
from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals
from ml.markers.center.geometry_classifier_v14.features import score_proposals
import json
from pathlib import Path


def test_scores_are_finite_and_bounded() -> None:
    scene = build_selection_scenes("dev")[0]
    proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
    features = score_proposals(scene.tensor, proposals)
    assert features
    assert all(0.0 <= feature.score <= 1.0 for feature in features)
    assert all(0.0 <= feature.mask_clear <= 1.0 for feature in features)


def test_v13_stream_covers_dev_truths() -> None:
    for scene in build_selection_scenes("dev"):
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        assert all(any((float(point[0]) - x) ** 2 + (float(point[1]) - y) ** 2 <= 25.0 for point in proposals.coordinates.tolist()) for x, y in scene.centers)


def test_tracked_dev_diagnostic_is_aggregate_only() -> None:
    report = json.loads((Path(__file__).parents[1] / "DEV_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed_dev"
    assert report["synthetic_only"] is True
    assert report["private_data"] is False
    assert report["sealed_runs"] == 0
    assert report["public_gate_evaluations"] == 0
    assert len(report["threshold_rows"]["train"]) == 5
    assert len(report["threshold_rows"]["dev"]) == 5
