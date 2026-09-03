# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry4_generic_fp import (
    RETRY4_ONNX_SHA256,
    summarize,
)


def test_retry4_generic_fp_report_is_aggregate_and_exhaustive() -> None:
    report_path = Path(__file__).with_name("V24_RETRY4_GENERIC_FP_DIAGNOSIS.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["fixed_threshold_metrics"]["generic_accepted_false_positive_count"] == 136
    assert report["fixed_threshold_metrics"]["accepted_false_positive_count"] == 138
    assert sum(report["root_cause_counts"].values()) == 136
    assert report["fixed_threshold_metrics"]["root_cause_partition_exhaustive"] is True
    assert report["binding"]["model_sha256"] == RETRY4_ONNX_SHA256
    assert report["scope"]["real_dev_reads"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["case_ids_or_pixels_emitted"] is False
    assert "scene_id" not in report and "truth_rows" not in report and "pixels" not in report


def test_retry4_generic_fp_model_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry4.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry4 ONNX hash mismatch"):
        summarize(model)
