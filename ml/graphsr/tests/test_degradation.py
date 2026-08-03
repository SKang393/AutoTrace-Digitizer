# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.graphsr.degradation import (
    REQUIRED_DEGRADATIONS,
    DegradationConfig,
    build_paired_crop,
)


EXPECTED_DEGRADATIONS = {
    "resize",
    "blur",
    "noise",
    "jpeg",
    "ringing",
    "paper",
    "halftone",
    "fade",
    "erosion",
    "dilation",
    "bleed",
    "skew",
    "perspective",
    "clipping",
    "jitter",
}


def _operation_names(metadata: dict[str, object]) -> set[str]:
    stages = metadata.get("stages")
    assert isinstance(stages, list)
    names: set[str] = set()
    geometry = metadata.get("geometry_preparation")
    assert isinstance(geometry, dict)
    assert geometry.get("applied_before") == "stage_1"
    geometry_operations = geometry.get("operations")
    assert isinstance(geometry_operations, list)
    for operation in geometry_operations:
        assert isinstance(operation, dict)
        assert operation.get("execution_phase") == "paired_target_preparation"
        if operation.get("applied") is True:
            names.add(str(operation["operation"]))
    for stage in stages:
        assert isinstance(stage, dict)
        operations = stage.get("operations")
        assert isinstance(operations, list) and operations
        for operation in operations:
            assert isinstance(operation, dict)
            name = operation.get("operation")
            parameters = operation.get("parameters")
            assert isinstance(name, str) and name
            assert isinstance(parameters, dict)
            if operation.get("applied") is True and name in EXPECTED_DEGRADATIONS:
                names.add(name)
    return names


def test_required_degradation_contract_is_complete() -> None:
    assert set(REQUIRED_DEGRADATIONS) == EXPECTED_DEGRADATIONS
    assert len(REQUIRED_DEGRADATIONS) == len(set(REQUIRED_DEGRADATIONS))


def test_two_stage_pair_and_metadata_are_seed_reproducible(
    chart_fixture: tuple[np.ndarray, tuple[tuple[float, float], ...]],
) -> None:
    image, centers = chart_fixture
    input_before = image.copy()
    config = DegradationConfig(
        stage_count=2,
        force_operations=tuple(REQUIRED_DEGRADATIONS),
    )

    first = build_paired_crop(image, centers, seed=20260803, config=config)
    second = build_paired_crop(image, centers, seed=20260803, config=config)
    changed = build_paired_crop(image, centers, seed=20260804, config=config)

    np.testing.assert_array_equal(image, input_before)
    np.testing.assert_array_equal(first.hr, second.hr)
    np.testing.assert_array_equal(first.lr, second.lr)
    assert json.dumps(first.metadata, sort_keys=True, allow_nan=False) == json.dumps(
        second.metadata, sort_keys=True, allow_nan=False
    )
    assert first.metadata["seed"] == 20260803
    assert len(first.metadata["stages"]) == 2
    assert _operation_names(first.metadata) == EXPECTED_DEGRADATIONS
    assert not np.array_equal(first.lr, changed.lr) or first.metadata != changed.metadata
    assert first.hr.dtype == np.uint8 and first.lr.dtype == np.uint8
    assert first.hr.shape == image.shape
    assert first.lr.shape == (image.shape[0] // 2, image.shape[1] // 2, 3)


@pytest.mark.parametrize("operation", sorted(EXPECTED_DEGRADATIONS))
def test_each_required_operation_records_concrete_parameters(
    chart_fixture: tuple[np.ndarray, tuple[tuple[float, float], ...]],
    operation: str,
) -> None:
    image, centers = chart_fixture
    pair = build_paired_crop(
        image,
        centers,
        seed=37,
        config=DegradationConfig(stage_count=2, force_operations=(operation,)),
    )

    assert operation in _operation_names(pair.metadata)
    json.dumps(pair.metadata, sort_keys=True, allow_nan=False)
    assert np.isfinite(pair.lr.astype(np.float32)).all()


@pytest.mark.parametrize("operation", ("skew", "perspective", "jitter"))
def test_geometry_operations_execute_before_both_raster_stages(
    chart_fixture: tuple[np.ndarray, tuple[tuple[float, float], ...]],
    operation: str,
) -> None:
    image, centers = chart_fixture
    pair = build_paired_crop(
        image,
        centers,
        seed=37,
        config=DegradationConfig(stage_count=2, force_operations=(operation,)),
    )

    assert not np.array_equal(pair.hr, image)
    geometry = pair.metadata["geometry_preparation"]
    applied = [item for item in geometry["operations"] if item["applied"]]
    assert [item["operation"] for item in applied] == [operation]
    stage_names = {
        item["operation"]
        for stage in pair.metadata["stages"]
        for item in stage["operations"]
    }
    assert operation not in stage_names


def test_paired_transforms_preserve_marker_center_correspondence(
    chart_fixture: tuple[np.ndarray, tuple[tuple[float, float], ...]],
) -> None:
    image, centers = chart_fixture
    pair = build_paired_crop(
        image,
        centers,
        seed=91,
        config=DegradationConfig(
            stage_count=2,
            force_operations=("resize", "skew", "perspective", "clipping"),
        ),
    )

    assert len(pair.marker_centers_hr) == len(centers)
    assert len(pair.marker_centers_lr) == len(centers)
    for source, hr, transformed in zip(
        centers,
        pair.marker_centers_hr,
        pair.marker_centers_lr,
        strict=True,
    ):
        assert pair.source_to_hr.apply(source) == pytest.approx(hr, abs=1e-6)
        assert pair.hr_to_source.apply(hr) == pytest.approx(source, abs=1e-5)
        mapped = pair.hr_to_lr.apply(hr)
        recovered = pair.lr_to_hr.apply(transformed)
        assert mapped == pytest.approx(transformed, abs=1e-6)
        assert recovered == pytest.approx(hr, abs=1e-5)
        assert 0 <= transformed[0] < pair.lr.shape[1]
        assert 0 <= transformed[1] < pair.lr.shape[0]
