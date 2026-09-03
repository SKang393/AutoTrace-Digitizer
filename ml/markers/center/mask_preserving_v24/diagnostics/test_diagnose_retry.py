# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import json
from pathlib import Path

import torch
from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import STRATA, MORPHOLOGY_KEYS, _quantiles, _strata, _patch_morphology


def test_quantiles_are_aggregate_and_deterministic():
    assert _quantiles([0.1, 0.2, 0.3])["count"] == 3
    assert _quantiles([]) == {"count": 0}


def test_report_shape_has_required_strata_without_case_fields(tmp_path):
    report = {"proposals": {"negative_strata": {name: {} for name in STRATA}}}
    path = tmp_path / "report.json"; path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(loaded["proposals"]["negative_strata"]) == STRATA
    assert not any(key in loaded for key in ("scene_ids", "truth_rows", "pixels"))

def test_morphology_feature_contract():
    features = _patch_morphology(torch.zeros((3, 33, 33)))
    assert tuple(features) == MORPHOLOGY_KEYS
    assert features["dark_fraction_ge_012"] == 0.0
    assert features["max_ring_support_3_12"] == 0.0

def test_covariance_ratio_is_major_over_minor():
    patch = torch.zeros((3, 33, 33))
    patch[0, 16, 10:23] = 1.0
    features = _patch_morphology(patch)
    assert features["covariance_eigen_ratio"] > 100.0

def test_ring_uses_rounded_euclidean_points():
    patch = torch.zeros((3, 33, 33))
    for i in range(8):
        x = int(round(16 + 5 * __import__('math').cos(i*__import__('math').pi/4)))
        y = int(round(16 + 5 * __import__('math').sin(i*__import__('math').pi/4)))
        patch[0, y, x] = 1.0
    assert _patch_morphology(patch)["max_ring_support_3_12"] == 8.0
