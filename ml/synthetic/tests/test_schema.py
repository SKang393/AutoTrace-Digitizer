# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Declarative scene-schema and template coverage tests."""

from __future__ import annotations

import copy

import pytest

from ml.synthetic.schema import SceneValidationError, load_schema, validate_scene
from ml.synthetic.templates import (
    FAMILY_TO_SPLIT,
    FILL_STATES,
    LINE_STYLES,
    MARKER_SHAPES,
    RENDERER_FAMILIES,
    SUPPORTED_DESIGNS,
    build_scene,
)


EXPECTED_DESIGNS = {
    "ab",
    "aba",
    "abab",
    "multiple_baseline",
    "multiple_probe",
    "alternating_treatments",
    "changing_criterion",
    "maintenance",
    "generalization",
    "staggered_starts",
    "shared_baseline",
}


def test_schema_is_draft_2020_12_and_rejects_unknown_fields() -> None:
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    scene = build_scene("ab", 393, "vector_clean")
    invalid = copy.deepcopy(scene)
    invalid["copied_article_data"] = True
    with pytest.raises(SceneValidationError):
        validate_scene(invalid)


def test_all_required_designs_produce_valid_scenes() -> None:
    assert set(SUPPORTED_DESIGNS) == EXPECTED_DESIGNS
    for index, design in enumerate(SUPPORTED_DESIGNS):
        family = RENDERER_FAMILIES[index % len(RENDERER_FAMILIES)]
        scene = build_scene(design, 39300 + index, family)
        assert validate_scene(scene) is scene


def test_bounds_and_marker_catalog_are_complete() -> None:
    two_session = build_scene("ab", 1, "vector_clean", 1, session_count=2)
    hundred_session = build_scene(
        "shared_baseline",
        2,
        "hand_drawn",
        6,
        session_count=100,
    )
    validate_scene(two_session)
    validate_scene(hundred_session)
    assert len(hundred_session["panels"]) == 6
    assert len(MARKER_SHAPES) == 9
    assert set(FILL_STATES) == {"open", "filled", "degraded"}
    assert {"solid", "dashed", "missing", "partially_occluded"} <= set(LINE_STYLES)


def test_family_split_tables_are_internally_consistent() -> None:
    for renderer_family in RENDERER_FAMILIES:
        scene = build_scene("ab", 12, renderer_family)
        splits = {
            family["split"] for family in scene["families"].values()
        }
        assert len(splits) == 1
        for category, family in scene["families"].items():
            assert FAMILY_TO_SPLIT[category][family["key"]] == family["split"]
