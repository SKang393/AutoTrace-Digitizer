# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from ml.markers.center.generator_input_gap.audit import audit


def test_audit_is_aggregate_only_and_fixed_shape() -> None:
    result = audit()
    assert result["scope"]["synthetic_only"] is True
    assert result["scope"]["model_loaded"] is False
    assert result["scope"]["training_performed"] is False
    assert result["scope"]["scene_ids_emitted"] is False
    assert result["scope"]["truth_rows_emitted"] is False
    assert result["aggregate"]["fixed_224x168_tensor"] is True
    assert result["aggregate"]["resize_degradation_roundtrip_restored"] is True
    assert result["aggregate"]["production_marker_frame_resize_observed"] is False


def test_audit_reports_both_splits_and_patch_ratios() -> None:
    result = audit()
    assert result["splits"]["train"]["scene_count"] == 21
    assert result["splits"]["dev"]["scene_count"] == 12
    for split in result["splits"].values():
        assert split["tensor_shapes"] == [[3, 168, 224]]
        assert split["marker_diameter_px"]["minimum"] >= 6.0
        assert split["marker_diameter_px"]["maximum"] <= 24.0
        assert split["marker_diameter_px"]["p05"] <= split["marker_diameter_px"]["p95"]
        assert split["diameter_to_33px_patch_ratio"]["maximum"] > 0
