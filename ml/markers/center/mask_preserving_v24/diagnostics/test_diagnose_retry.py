# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import json
from pathlib import Path

from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import STRATA, _quantiles, _strata


def test_quantiles_are_aggregate_and_deterministic():
    assert _quantiles([0.1, 0.2, 0.3])["count"] == 3
    assert _quantiles([]) == {"count": 0}


def test_report_shape_has_required_strata_without_case_fields(tmp_path):
    report = {"proposals": {"negative_strata": {name: {} for name in STRATA}}}
    path = tmp_path / "report.json"; path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(loaded["proposals"]["negative_strata"]) == STRATA
    assert not any(key in loaded for key in ("scene_ids", "truth_rows", "pixels"))
