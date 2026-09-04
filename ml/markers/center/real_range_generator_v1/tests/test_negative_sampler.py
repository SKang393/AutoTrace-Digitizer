# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import pytest
import torch

from ml.markers.center.real_range_generator_v1.negative_sampler import (
    LOW_FAINT, P05_FAINT, OCR_P95, ARTIFACT_P95, CONNECTOR_ANCHOR_MAX_DISTANCE_PX, CONNECTOR_ENDPOINT_OFFSET_PX, GENERIC_CONNECTOR_BAND_RADIUS_PX, TOPOLOGY_SAMPLER_RADIUS_PX, _features, _stable_key,
    sample_negatives,
)


def test_threshold_boundaries_are_fixed_and_repeatable():
    patches = torch.zeros((4, 3, 33, 33))
    patches[0, 0] = LOW_FAINT
    patches[1, 0] = P05_FAINT
    patches[2, 0] = 0.5
    patches[3, 0] = 0.5
    patches[2, 1] = OCR_P95
    patches[3, 2] = ARTIFACT_P95
    first = _features(patches)
    second = _features(patches)
    assert torch.equal(first["faint_low"], torch.tensor([True, False, False, False]))
    assert torch.equal(first["faint_p05"], torch.tensor([False, True, False, False]))
    assert torch.equal(first["ocr_heavy"], torch.tensor([False, False, True, False]))
    assert torch.equal(first["artifact"], torch.tensor([False, False, False, True]))
    for key in first:
        assert torch.equal(first[key], second[key])


def test_stable_rank_key_is_repeatable_and_stratum_bound():
    key = _stable_key(20260904, "train", 4100, 7, "generic")
    assert key == _stable_key(20260904, "train", 4100, 7, "generic")
    assert key != _stable_key(20260904, "train", 4100, 7, "artifact")


def test_sampling_is_train_only_and_fails_closed_on_under_capacity():
    with pytest.raises(ValueError, match="train-only"):
        sample_negatives([], split="dev")
    with pytest.raises(ValueError, match="under-capacity"):
        sample_negatives([], split="train")


def test_full_train_sampler_matches_preregistered_contract():
    from ml.markers.center.mask_preserving_v24.train_p1 import _examples_with_report
    from ml.markers.center.real_range_generator_v1.generator import build_split

    patches, labels, _, _, hard, sampled = _examples_with_report(
        build_split("train"), 10, torch.Generator().manual_seed(20260904)
    )
    assert len(labels) == 35838
    assert int((labels > 0.5).sum()) == 3258
    assert int(hard.sum()) == 6856
    assert sampled.total == 32580
    assert sampled.counts == {
        "hard_existing": 6012,
        "faint_low": 326,
        "faint_p05": 1303,
        "ocr_heavy": 1629,
        "artifact": 1629,
        "generic_connector_band": 6720,
        "generic": 14961,
    }
    assert sampled.selected_index_sha256 == "d7460b95bbdbb89d79a12cafe7632604f02b8087e9986fb7a9d3ea940287567f"
    assert sampled.capacities["hard_existing"] == 6012
    assert sampled.topology_capacity == {"topology_junction": 4505, "topology_fragment": 4574}
    assert sampled.topology_selected == sampled.topology_capacity
    assert sampled.topology_selected_index_sha256 == "671e6e7c7affbbb79171cc31d76863fe8b541904b3727cfd633da2bed7fab95c"
    assert sampled.topology_sampler_radius_px == TOPOLOGY_SAMPLER_RADIUS_PX == 12.0
    assert sampled.connector_anchor_target_count == 3674
    assert sampled.connector_anchor_capacity == 3671
    assert sampled.connector_anchor_selected == 3671
    assert sampled.connector_endpoint_offset_px == CONNECTOR_ENDPOINT_OFFSET_PX == 8.0
    assert sampled.connector_anchor_max_distance_px == CONNECTOR_ANCHOR_MAX_DISTANCE_PX == 4.0
    assert sampled.connector_anchor_selected_index_sha256 == "fd20045f034c9d5c4882e81b28bfa3f357befe6150183817bdd772f3a04ceef2"
    assert sampled.capacities["generic_connector_band"] == 50373
    assert sampled.counts["generic_connector_band"] == 6720
    assert sampled.generic_connector_band_selected_index_sha256 == "4e58e9e353a0ff912bccb28845e7e1d619d4903929f9cf49c6244dc5017fc96a"
    assert GENERIC_CONNECTOR_BAND_RADIUS_PX == 4.0
    assert sampled.generic_remainder_selected == 2689
    assert sampled.topology_hard_capacity == {"topology_junction": 417, "topology_fragment": 484}
    assert sampled.topology_hard_selected == sampled.topology_hard_capacity
    assert sampled.hard_training_total == 6856
    assert patches.shape == (35838, 3, 33, 33)


def test_selected_connector_band_excludes_legacy_feature_strata():
    import hashlib

    from ml.markers.center.real_range_generator_v1.negative_sampler import (
        _connecting_segment_distances, _connector_anchor_indices,
        _generic_connector_band_indices, _hard_indices, _topology_indices,
        sample_negatives,
    )
    from ml.markers.center.real_range_generator_v1.negative_proposal_audit import _train_sampler_records

    records = _train_sampler_records()
    sampled = sample_negatives(records, split="train", seed=20260904)
    band_entries = []
    for scene_number, (scene, proposals, labels, _) in enumerate(records):
        features = _features(proposals.patches)
        legacy_hard = set(_hard_indices(scene, proposals.coordinates, labels).tolist())
        topology = _topology_indices(scene, proposals.coordinates, labels, radius_px=TOPOLOGY_SAMPLER_RADIUS_PX)
        topology_by_index = {index: kind for kind in topology for index in topology[kind]}
        connector = _connector_anchor_indices(scene, proposals.coordinates, labels)
        segment_distances = _connecting_segment_distances(scene, proposals.coordinates)
        band = _generic_connector_band_indices(
            scene, proposals.coordinates, labels, features, legacy_hard,
            topology_by_index, connector)
        band_entries.extend(
            (_stable_key(20260904, "train", int(scene.seed), index, "generic_connector_band"), scene_number, index)
            for index in band
        )
        for index in band:
            assert float(labels[index]) <= 0.5
            assert index not in legacy_hard
            assert index not in topology_by_index
            assert index not in connector
            assert float(segment_distances[index]) <= 4.0
            assert not any(bool(features[name][index]) for name in ("faint_low", "faint_p05", "ocr_heavy", "artifact"))
    selected_band = {
        (scene_number, index)
        for _, scene_number, index in band_entries
        if index in set(sampled.selections[scene_number])
    }
    digest = hashlib.sha256()
    for scene_number, index in sorted(selected_band):
        digest.update(f"{scene_number}:{index}\n".encode("ascii"))
    assert selected_band
    assert digest.hexdigest() == sampled.generic_connector_band_selected_index_sha256
