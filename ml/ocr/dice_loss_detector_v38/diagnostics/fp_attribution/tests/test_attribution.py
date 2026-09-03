# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Small aggregate-only checks for V38 attribution helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.dice_loss_detector_v38.diagnostics.fp_attribution.attribute import (
    CATEGORY_NAMES,
    _category_masks,
    _matched_pairs,
    _size_bucket,
)


def test_category_masks_are_disjoint_and_use_marker_mask() -> None:
    annotation = {
        "panels": [{
            "axes": [{"line": [[1, 1], [8, 1]]}],
            "ticks": [{"line": [[4, 1], [4, 5]]}],
            "markers": [{"center": [8, 8], "radius": 1}],
            "edges": [{"line": [[8, 8], [9, 9]]}],
            "dividers": [{"line": [[13, 0], [13, 12]]}],
            "texts": [{"rendered_pixel_box": [1, 8, 3, 2]}],
        }],
    }
    marker = np.zeros((16, 16), dtype=np.uint8)
    marker[8, 8] = 255
    category_map, _ = _category_masks(annotation, marker)
    assert set(np.unique(category_map)).issubset(set(range(len(CATEGORY_NAMES))))
    assert category_map[8, 8] == 1
    assert category_map[10, 13] == 2
    assert category_map[9, 1] == 3


def test_matching_and_size_bucket_are_deterministic() -> None:
    predicted = (Box(0, 0, 10, 10), Box(20, 20, 25, 25))
    truths = (Box(0, 0, 10, 10), Box(40, 40, 50, 50))
    assert _matched_pairs(predicted, truths) == {(0, 0)}
    assert _size_bucket(Box(0, 0, 10, 10)) == "small_area_le_200"
    assert _size_bucket(Box(0, 0, 20, 20)) == "medium_area_201_to_1000"
    assert _size_bucket(Box(0, 0, 40, 40)) == "large_area_gt_1000"


def test_report_is_aggregate_and_utf8_lf() -> None:
    path = Path(__file__).parents[1] / "ATTRIBUTION.json"
    if not path.is_file():
        return
    raw = path.read_bytes()
    assert b"\r" not in raw
    report = json.loads(raw.decode("utf-8"))
    encoded = json.dumps(report, sort_keys=True)
    assert report["schema"].endswith(".v1")
    assert "scene_id" not in encoded
    assert "prediction" not in encoded
    assert report["false_positive_components"]["false_positives"] >= 0
    assert set(report["false_positive_pixels"]["by_category"]) == set(CATEGORY_NAMES)
