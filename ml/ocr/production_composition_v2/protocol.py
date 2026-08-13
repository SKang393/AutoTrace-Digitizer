# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the OCR production-composition V2 gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-production-composition"
REVISION = "graphreader-v9-official-spacing-p2-numeric-v5-composition-v2"
PUBLIC_REVISION = f"{REVISION}-public-v1"
VALIDATION_REVISION = f"{REVISION}-validation-v1"
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
PLOT_BOUNDS = (104, 48, 510, 256)
DETECTOR_THRESHOLD = 0.925
NUMERIC_THRESHOLD = 0.65
TRUTH_MATCH_IOU_MINIMUM = 0.5
EXACT_MATCH_MINIMUM = 0.90
CHARACTER_ERROR_RATE_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
NUMERIC_EXACT_MINIMUM = 0.90
WORD_EXACT_MINIMUM = 0.90
AMBIGUITY_EXACT_MINIMUM = 0.90

DETECTOR_ONNX_SHA256 = "2d35ce2f55cee8317dfe1faf0281a6e87693cca485dbdaf39e4039ced5b97d9c"
OFFICIAL_RECOGNIZER_ONNX_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
NUMERIC_RECOGNIZER_ONNX_SHA256 = "9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84"
OFFICIAL_INFERENCE_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
SPACING_SOURCE_SHA256 = "42377b7ff0c650518861940a04f31382c8bed40013089354586e34bde12d4c40"


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    scene_count: int
    seed_offset: int
    renderer_family: str
    degradation_family: str
    font_paths: tuple[str, ...]
    font_sha256: tuple[str, ...]


_REGULAR = "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"
_MEDIUM = "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"
_SEMIBOLD = "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"
_REGULAR_SHA = "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823"
_MEDIUM_SHA = "635d93d1131d791f2576de90b3bb0f7cdf61929906e8420a61b5f7f8e76420bb"
_SEMIBOLD_SHA = "a4e91fd530ac2b4ef5367240144ff37d7d65d66cf76f2e9a2187b93c676f92d0"

SPLITS = (
    SplitRegistration(
        "validation",
        80,
        503_000,
        "five-role-staggered-spacing-scenes-composition-v2-validation",
        "box-resample-column-fade-composition-v2-validation",
        (_MEDIUM, _REGULAR, _SEMIBOLD),
        (_MEDIUM_SHA, _REGULAR_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "sealed_public",
        112,
        541_000,
        "five-role-reflected-spacing-scenes-composition-v2-public",
        "gamma-quantization-speckle-composition-v2-public",
        (_SEMIBOLD, _MEDIUM, _REGULAR),
        (_SEMIBOLD_SHA, _MEDIUM_SHA, _REGULAR_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR production-composition V2 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-production-composition-protocol.v2",
        "task": TASK,
        "revision": REVISION,
        "state": "fresh_splits_frozen_before_any_composed_model_execution",
        "composition_id": REVISION,
        "predecessor": {
            "revision": "graphreader-v8-official-english-v5-numeric-composition-v1",
            "validation_report_sha256": "a3d851c043993bdc5546c68dfa26be837c88495600f7cbd214c55ee107ef330f",
            "status": "failed",
            "failure": "113 detector false negatives across 320 truths",
            "public_archive_opened": False,
            "fixture_bytes_reused": False,
        },
        "models": {
            "detector": {
                "revision": "graph-text-component-recall-v9-p3",
                "onnx_sha256": DETECTOR_ONNX_SHA256,
                "threshold": DETECTOR_THRESHOLD,
                "public_component_report_sha256": "232eea4b01917bcbc786e289752b023f0d1ec9106df0349d2606dce06e76eeb9",
            },
            "general_recognizer": {
                "revision": "en_PP-OCRv5_mobile_rec",
                "onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
                "inference_yaml_sha256": OFFICIAL_INFERENCE_YAML_SHA256,
                "postprocess_revision": "official-ppocrv5-image-spacing-v2-p2",
                "postprocess_source_sha256": SPACING_SOURCE_SHA256,
                "public_component_report_sha256": "83afdcfd4fd3dd8b4fa8c0881b5f12d652c0d8c5f9b889af380b4fdf7b83091c",
            },
            "numeric_specialist": {
                "revision": "graph-numeric-component-ensemble-v5",
                "onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
                "threshold": NUMERIC_THRESHOLD,
            },
        },
        "routing": {
            "algorithm": "geometry-gated-numeric-specialist-v1",
            "general_recognizer_default": True,
            "numeric_specialist_requires_numeric_grammar": True,
            "numeric_specialist_requires_x_or_y_tick_geometry": True,
            "participant_phase_legend_annotation_numeric_substitution_forbidden": True,
        },
        "splits": [asdict(item) for item in SPLITS],
        "gates": {
            "exact_region_count_every_fixture": True,
            "false_region_count": 0,
            "missed_region_count": 0,
            "duplicate_region_count": 0,
            "prohibited_structure_hits": 0,
            "exact_match_minimum": EXACT_MATCH_MINIMUM,
            "character_error_rate_maximum": CHARACTER_ERROR_RATE_MAXIMUM,
            "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
            "numeric_exact_match_minimum": NUMERIC_EXACT_MINIMUM,
            "word_exact_match_minimum": WORD_EXACT_MINIMUM,
            "ambiguity_exact_match_minimum": AMBIGUITY_EXACT_MINIMUM,
            "spacing_changed_nonspace_truth_count": 0,
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
            "three_exact_onnx_payloads_required": True,
        },
        "data_scope": (
            "fresh procedural graph scenes with new seed, renderer, degradation, layout, and label-family "
            "registrations; generic numeric, phase, annotation, ambiguity, and legend labels only; no Chandler, "
            "Generalization, private/article images, external datasets, predecessor fixture bytes, or tuning"
        ),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "marker_creation_evaluated": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "AMBIGUITY_EXACT_MINIMUM", "CHARACTER_ERROR_RATE_MAXIMUM", "DETECTOR_ONNX_SHA256",
    "DETECTOR_THRESHOLD", "EXACT_MATCH_MINIMUM", "NUMERIC_EXACT_MINIMUM",
    "NUMERIC_RECOGNIZER_ONNX_SHA256", "NUMERIC_THRESHOLD", "OFFICIAL_INFERENCE_YAML_SHA256",
    "OFFICIAL_RECOGNIZER_ONNX_SHA256", "PLOT_BOUNDS", "PUBLIC_REVISION", "REVISION",
    "ROLE_ACCURACY_MINIMUM", "SCENE_HEIGHT", "SCENE_WIDTH", "SPACING_SOURCE_SHA256", "SPLITS",
    "TASK", "TRUTH_MATCH_IOU_MINIMUM", "VALIDATION_REVISION", "WORD_EXACT_MINIMUM",
    "protocol_configuration", "split_registration",
]
