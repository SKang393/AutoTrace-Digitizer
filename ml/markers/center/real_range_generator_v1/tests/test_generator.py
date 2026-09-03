# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from ml.markers.center.real_range_generator_v1.generator import audit, build_split


def test_determinism_and_disjoint_seeds() -> None:
    a, b = build_split("train"), build_split("train")
    assert [s.seed for s in a] == [s.seed for s in b]
    assert [s.tensor.numpy().tobytes() for s in a] == [s.tensor.numpy().tobytes() for s in b]
    assert set(s.seed for s in a).isdisjoint(s.seed for s in build_split("dev"))


def test_range_and_masks_match_required_aggregate() -> None:
    result = audit()
    for split in result["splits"].values():
        assert split["marker_count"] == 2004
        assert split["tensor_shapes"] == [[3, 168, 224]]
        assert split["diameter_px"]["minimum"] == 1.0
        assert split["diameter_px"]["maximum"] == 48.0
        assert split["diameter_px"]["p05"] == 6.0
        assert split["diameter_px"]["median"] == 12.0
        assert split["diameter_px"]["p90"] == 24.0
        assert split["diameter_px"]["p95"] == 27.0
        assert split["rendered_diameter_px"]["minimum"] == 1.0
        assert split["rendered_diameter_px"]["maximum"] >= 48.0
        assert split["mask_center_hits"] == {"ocr": 75, "artifact": 332}
        assert split["mask_center_hit_rates"]["ocr"] == 75 / 2004
        assert split["mask_center_hit_rates"]["artifact"] == 332 / 2004
    assert all(scene.tensor.dtype.is_floating_point for name in ("train", "dev") for scene in build_split(name))
    assert {str(scene.tensor.dtype) for name in ("train", "dev") for scene in build_split(name)} == {"torch.float32"}
    masks = result["mask_overlap_scenarios"]
    assert masks["markers_per_split"] == 2004
    assert (masks["ocr_hard_hits"], masks["artifact_hard_hits"]) == (75, 332)
    assert result["truth_center_patch_distribution"]["shape"] == [3, 33, 33]
    quantiles = result["truth_center_patch_distribution"]["channel_mean_quantiles"]
    assert "ink_center_5x5" in quantiles
    assert .07 <= quantiles["ink"]["p05"] <= .11
    assert .15 <= quantiles["ink"]["median"] <= .23
    assert quantiles["artifact_mask"]["p90"] >= .12121
    assert quantiles["artifact_mask"]["maximum"] >= .23967
    assert all(result["distribution_gates"].values())


def test_mask_counts_are_measured_from_tensor_windows() -> None:
    for split in ("train", "dev"):
        ocr = artifact = 0
        for scene in build_split(split):
            for x_float, y_float in scene.centers:
                x, y = int(round(x_float)), int(round(y_float))
                ocr += int(float(scene.tensor[1, y - 2:y + 3, x - 2:x + 3].max()) >= .35)
                artifact += int(float(scene.tensor[2, y - 2:y + 3, x - 2:x + 3].max()) >= .35)
        assert (ocr, artifact) == (75, 332)


def test_scope_is_aggregate_only_and_negatives_present() -> None:
    result = audit()
    scope = result["scope"]
    assert all(scope[key] is False for key in
               ("model_loaded", "training_performed", "private_or_article_images",
                "candidate_revision_created", "scene_ids_emitted", "truth_rows_emitted",
                "pixels_emitted"))
    assert result["hard_negative_kinds"] == ["text", "line_intersection", "axis"]
