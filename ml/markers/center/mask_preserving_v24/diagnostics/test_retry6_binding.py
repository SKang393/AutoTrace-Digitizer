# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    RETRY6_DEV_SPLIT_SHA256,
    RETRY6_GENERATOR_AUDIT_SHA256,
    RETRY6_ONNX_SHA256,
    summarize,
)


def test_retry6_hash_bindings_are_explicit() -> None:
    assert RETRY6_ONNX_SHA256 == "31d473d6c24bf21edc1cbfb25f7da35eabfed7cbf8afc13bf52bef23d06bfeb9"
    assert RETRY6_GENERATOR_AUDIT_SHA256 == "9562044526bce45b48254472346ccc1640c6e254915b9e744525154144121748"
    assert RETRY6_DEV_SPLIT_SHA256 == "f453e50e228f54e15bafba83b1a5dda422c435a555e9083dfea18457ed38204d"


def test_retry6_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry6.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry6 ONNX hash mismatch"):
        summarize(model, retry6=True)


def test_retry6_report_matches_fixed_metrics_and_prohibited_attribution() -> None:
    report = json.loads(Path(__file__).with_name("V24_RETRY6_DIAGNOSIS.json").read_text(encoding="utf-8"))
    metrics = report["fixed_threshold_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (1988, 86, 16)
    assert metrics["precision"] == pytest.approx(0.9585342333654774)
    assert metrics["recall"] == pytest.approx(0.9920159680638723)
    assert metrics["prohibited_structure_hits"] == 1
    assert report["retry6_attribution"]["topology_sampler_radius_px"] == 12.0
    assert report["retry6_attribution"]["connector_endpoint_offset_px"] == 8.0
    assert report["retry6_attribution"]["connector_anchor_max_distance_px"] == 4.0
    assert report["prohibited_hit_attribution"]["total"] == 1
    assert report["scope"]["real_dev_reads"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["case_ids_or_pixels_emitted"] is False
    assert report["binding"]["generator_dev_split_sha256"] == RETRY6_DEV_SPLIT_SHA256
    assert "scene_id" not in report and "truth_rows" not in report and "pixels" not in report
