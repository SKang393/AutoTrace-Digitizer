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


def test_audit_reports_zero_padded_truth_center_patch_aggregates() -> None:
    result = audit()
    for split in result["truth_center_patches"].values():
        assert split["patch_shape"] == [3, 33, 33]
        assert split["marker_count"] > 0
        assert set(split["channel_quantiles"]) == {
            "ink_mean", "ink_center_5x5_mean", "ink_max",
            "ocr_mask_mean", "ocr_mask_max",
            "artifact_mask_mean", "artifact_mask_max",
        }
        for quantiles in split["channel_quantiles"].values():
            assert set(quantiles) == {"minimum", "p05", "p10", "median", "p90", "p95", "maximum"}
        hits = split["truth_centers_mask_window_threshold_hits"]
        assert hits["threshold"] == 0.35
        assert hits["window_size"] == [5, 5]
