# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the OCR production-composition V1 gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-production-composition"
REVISION = "graphreader-v8-official-english-v5-numeric-composition-v1"
PUBLIC_REVISION = f"{REVISION}-public-v1"
VALIDATION_REVISION = f"{REVISION}-validation-v1"
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
PLOT_BOUNDS = (104, 48, 510, 256)
DETECTOR_THRESHOLD = 0.95
NUMERIC_THRESHOLD = 0.65
TRUTH_MATCH_IOU_MINIMUM = 0.5
EXACT_MATCH_MINIMUM = 0.90
CHARACTER_ERROR_RATE_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90

DETECTOR_ONNX_SHA256 = "e0254920b26784a87369aa25cc4ec387c6544db30bda4f9542b7ce9a8712e431"
OFFICIAL_RECOGNIZER_ONNX_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
NUMERIC_RECOGNIZER_ONNX_SHA256 = "9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84"
OFFICIAL_INFERENCE_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"


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
        64,
        319_000,
        "four-role-offset-graph-scenes-composition-v1-validation",
        "bicubic-resample-row-fade-composition-v1-validation",
        (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "sealed_public",
        96,
        337_000,
        "four-role-reflected-graph-scenes-composition-v1-public",
        "gamma-quantization-pixel-loss-composition-v1-public",
        (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR production-composition split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-production-composition-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "split_frozen_before_any_composed_model_execution",
        "composition_id": REVISION,
        "models": {
            "detector": {
                "revision": "graph-text-component-fusion-v8",
                "onnx_sha256": DETECTOR_ONNX_SHA256,
                "threshold": DETECTOR_THRESHOLD,
            },
            "general_recognizer": {
                "revision": "en_PP-OCRv5_mobile_rec",
                "onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
                "inference_yaml_sha256": OFFICIAL_INFERENCE_YAML_SHA256,
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
        "recognition_crops": {
            "general": {
                "width": 320,
                "height": 48,
                "horizontal_padding_pixels": 8,
                "vertical_padding_pixels": 2,
                "right_padding_source_value": 0.5,
                "resampling": "half_pixel_bilinear_v1",
            },
            "numeric": {
                "width": 128,
                "height": 32,
                "horizontal_padding_pixels": 12,
                "vertical_padding_pixels": 1,
                "vertical_content_padding_ratio": 0.25,
                "right_padding_source_value": 1.0,
                "resampling": "half_pixel_bilinear_v1",
            },
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
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
            "three_exact_onnx_payloads_required": True,
        },
        "data_scope": (
            "fresh procedural graph scenes with independently registered renderer, degradation, seed, and layout "
            "families; generic numeric, phase, annotation, and legend labels only; no Chandler, Generalization, "
            "private or article images, external datasets, prior selection/public fixture bytes, or model tuning"
        ),
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "font_notice_sha256": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
        "official_notice_paths": [
            "LICENSES/PaddlePaddle-PP-OCRv5-Models-Apache-2.0.txt",
            "LICENSES/PaddlePaddle-PP-OCRv5-Models-Notice.txt",
        ],
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "CHARACTER_ERROR_RATE_MAXIMUM",
    "DETECTOR_ONNX_SHA256",
    "DETECTOR_THRESHOLD",
    "EXACT_MATCH_MINIMUM",
    "NUMERIC_RECOGNIZER_ONNX_SHA256",
    "NUMERIC_THRESHOLD",
    "OFFICIAL_INFERENCE_YAML_SHA256",
    "OFFICIAL_RECOGNIZER_ONNX_SHA256",
    "PLOT_BOUNDS",
    "PUBLIC_REVISION",
    "REVISION",
    "ROLE_ACCURACY_MINIMUM",
    "SCENE_HEIGHT",
    "SCENE_WIDTH",
    "SPLITS",
    "TASK",
    "TRUTH_MATCH_IOU_MINIMUM",
    "VALIDATION_REVISION",
    "protocol_configuration",
    "split_registration",
]
