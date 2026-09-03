# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import pytest
import torch

from ml.markers.center.real_range_generator_v1.negative_sampler import (
    LOW_FAINT, P05_FAINT, OCR_P95, ARTIFACT_P95, CONNECTOR_ANCHOR_MAX_DISTANCE_PX, CONNECTOR_ENDPOINT_OFFSET_PX, TOPOLOGY_SAMPLER_RADIUS_PX, _features, _stable_key,
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
    assert int(hard.sum()) == 6012
    assert sampled.total == 32580
    assert sampled.counts == {
        "hard_existing": 6012,
        "faint_low": 326,
        "faint_p05": 1303,
        "ocr_heavy": 1629,
        "artifact": 1629,
        "generic": 21681,
    }
    assert sampled.selected_index_sha256 == "d2e1c5f22ba8657b04031242c1528a165730f56b50ccc56fdd89eb2e0c01bf1c"
    assert sampled.topology_capacity == {"topology_junction": 4505, "topology_fragment": 4574}
    assert sampled.topology_selected == sampled.topology_capacity
    assert sampled.topology_selected_index_sha256 == "92d8075523ac528eebf8777b9fbe77d7eb254aebbc4521e0b45967f6e3b78ea8"
    assert sampled.topology_sampler_radius_px == TOPOLOGY_SAMPLER_RADIUS_PX == 12.0
    assert sampled.connector_anchor_target_count == 3674
    assert sampled.connector_anchor_capacity == 3671
    assert sampled.connector_anchor_selected == 3671
    assert sampled.connector_endpoint_offset_px == CONNECTOR_ENDPOINT_OFFSET_PX == 8.0
    assert sampled.connector_anchor_max_distance_px == CONNECTOR_ANCHOR_MAX_DISTANCE_PX == 4.0
    assert sampled.connector_anchor_selected_index_sha256 == "2d9b5b6ffa7e70c390a7aa38ffe851871e8a5cebac3831aac616f60a0e141c84"
    assert sampled.generic_remainder_selected == 9409
    assert patches.shape == (35838, 3, 33, 33)
