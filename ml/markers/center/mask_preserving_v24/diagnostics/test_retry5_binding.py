# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    RETRY5_DEV_SPLIT_SHA256,
    RETRY5_GENERATOR_AUDIT_SHA256,
    RETRY5_ONNX_SHA256,
    summarize,
)
from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry4_generic_fp import summarize as summarize_generic


def test_retry5_hash_and_split_bindings_are_explicit() -> None:
    assert RETRY5_ONNX_SHA256 == "d3445f0b1bf0e97a98942133d45341cae75548887be853743e887832cacad7bd"
    assert RETRY5_GENERATOR_AUDIT_SHA256 == "9562044526bce45b48254472346ccc1640c6e254915b9e744525154144121748"
    assert RETRY5_DEV_SPLIT_SHA256 == "f453e50e228f54e15bafba83b1a5dda422c435a555e9083dfea18457ed38204d"


def test_retry5_standard_and_generic_hash_mismatches_fail_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry5.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry5 ONNX hash mismatch"):
        summarize(model, retry5=True)
    with pytest.raises(ValueError, match="retry5 ONNX hash mismatch"):
        summarize_generic(model, retry5=True)


def test_retry5_reports_match_fixed_metrics_and_are_aggregate_only() -> None:
    standard = json.loads(Path(__file__).with_name("V24_RETRY5_DIAGNOSIS.json").read_text(encoding="utf-8"))
    generic = json.loads(Path(__file__).with_name("V24_RETRY5_GENERIC_FP_DIAGNOSIS.json").read_text(encoding="utf-8"))
    metrics = standard["fixed_threshold_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (1986, 184, 18)
    assert metrics["precision"] == pytest.approx(0.9152073732718894)
    assert metrics["recall"] == pytest.approx(0.9910179640718563)
    assert standard["retry5_attribution"]["accepted_false_positives"] == {"connector_anchor": 0, "generic": 183, "topology_fragment": 1, "topology_junction": 0}
    assert generic["fixed_threshold_metrics"]["generic_accepted_false_positive_count"] == 183
    assert generic["comparison_to_retry4"]["root_cause_counts"] == {"near_connecting_line": 102, "masked_context": 13, "marker_field_generic": 21}
    for report in (standard, generic):
        assert report["scope"]["real_dev_reads"] == 0
        assert report["scope"]["real_sealed_reads"] == 0
        assert report["scope"]["optimizer_steps"] == 0
        assert report["scope"]["case_ids_or_pixels_emitted"] is False
        assert "scene_id" not in report and "truth_rows" not in report and "pixels" not in report
    assert standard["binding"]["generator_dev_split_sha256"] == RETRY5_DEV_SPLIT_SHA256
    assert generic["binding"]["generator_dev_split_sha256"] == RETRY5_DEV_SPLIT_SHA256
