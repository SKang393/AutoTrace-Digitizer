# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from ml.markers.center.line_aware_v1.pipeline import extract_proposals
from ml.markers.center.proposal_geometry_v13.dataset import FAMILIES, build_selection_scenes, selection_manifest
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals


def test_splits_are_disjoint_and_private_free() -> None:
    assert set(FAMILIES["train"]).isdisjoint(FAMILIES["dev"])
    manifest = selection_manifest()
    assert manifest["synthetic_only"] is True
    assert manifest["private_or_article_images"] is False
    assert manifest["public_gate_archive_opened"] is False


def test_geometry_filter_preserves_all_synthetic_dev_truths() -> None:
    for scene in build_selection_scenes("dev"):
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        assert all(any((float(point[0]) - x) ** 2 + (float(point[1]) - y) ** 2 <= 25 for point in proposals.coordinates.tolist()) for x, y in scene.centers)
