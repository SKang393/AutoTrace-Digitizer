# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

from ml.markers.center.multiradius_geometry_v23.diagnose_v23 import DEFAULT_ONNX, summarize


ROOT = Path(__file__).parents[5]
def test_v23_is_aggregate_only_and_reports_fixed_radii() -> None:
    result = summarize(DEFAULT_ONNX)
    assert result["scope"]["fixed_dev_manifest_sha256"] == "771141bd5d22d2ffdb56e63f193c16c0775a0f47c02313ef67b06bbce9603126"
    assert result["scope"]["synthetic_only"] is True
    assert result["scope"]["training_performed"] is False
    assert result["scope"]["optimizer_steps"] == 0
    assert result["scope"]["scene_ids_emitted"] is False
    assert result["scope"]["case_level_details_emitted"] is False
    assert result["inference"]["ring_radii_px"] == list(range(3, 13))


def test_v23_tracked_diagnostic_is_utf8_lf_and_case_free() -> None:
    path = Path(__file__).parents[1] / "V23_FEASIBILITY_DIAGNOSTIC.json"
    data = path.read_bytes()
    assert b"\r" not in data
    result = json.loads(data.decode("utf-8"))
    assert result["schema"] == "graphreader.marker-center-multiradius-geometry-v23-feasibility.v1"
    assert result["scope"]["scene_ids_emitted"] is False
    assert result["scope"]["case_level_details_emitted"] is False
    assert "truth_rows" not in json.dumps(result, sort_keys=True)
