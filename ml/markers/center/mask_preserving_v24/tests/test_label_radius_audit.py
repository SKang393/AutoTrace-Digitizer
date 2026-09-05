# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import numpy as np
import pytest
import hashlib
import json
from pathlib import Path

from ml.markers.center.mask_preserving_v24.diagnostics.label_radius_audit import radius_counts


def test_radius_comparison_preserves_stream_and_threshold_boundaries():
    counts = radius_counts(np.array([0, 3, 3.01, 5, 5.01]),
                           np.array([0.9, 0.25, 0.25, 0.1, 0.8]))
    assert counts["3"] == dict(positive=2, negative_above_threshold=2, negative_below_threshold=1)
    assert counts["5"] == dict(positive=4, negative_above_threshold=1, negative_below_threshold=0)
    assert all(sum(record.values()) == 5 for record in counts.values())


@pytest.mark.parametrize("distances,scores", [([0], [0, 1]), ([np.nan], [1]), ([-1], [1]), ([1], [np.inf])])
def test_invalid_stream_is_rejected(distances, scores):
    with pytest.raises(ValueError):
        radius_counts(np.array(distances), np.array(scores))


def test_recorded_sparse_confirmation_is_bound_and_not_production_approval():
    root = Path(__file__).resolve().parents[5]
    report = json.loads((root / "docs/GOAL-22-PHASE-4R-V24-SPARSE-DEV-CONFIRMATION.json").read_text())
    audit_path = root / "ml/markers/center/real_range_generator_v1/SPARSE_FRAGMENT_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    bars = json.loads((root / report["acceptance_bars_path"]).read_text())["tier1_reviewable_error"]
    assert report["input_audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert report["split_sha256"] == audit["splits"]["dev"]["aggregate_sha256"]
    assert report["selected"]["precision"] >= bars["marker_center_precision_minimum"]
    assert report["selected"]["recall"] >= bars["marker_center_recall_minimum"]
    assert report["scope"]["optimizer_steps"] == report["scope"]["private_reads"] == report["scope"]["sealed_reads"] == 0
    assert report["production_approval"] is False
