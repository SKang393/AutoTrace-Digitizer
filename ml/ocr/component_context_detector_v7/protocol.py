# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol constants for the context-aware OCR detector V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-component-context-v7"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20261112
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
CROP_WIDTH = 128
CROP_HEIGHT = 32
INPUT_CHANNELS = 2
GEOMETRY_FEATURE_COUNT = 12
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
PROPOSAL_THRESHOLD_MINIMUM = 32
PROPOSAL_THRESHOLD_MAXIMUM = 224
PROPOSAL_THRESHOLD_MEAN_RATIO = 0.8
MINIMUM_COMPONENT_AREA = 2
MAXIMUM_COMPONENT_WIDTH_RATIO = 0.15
MAXIMUM_COMPONENT_HEIGHT_RATIO = 0.20
MINIMUM_VERTICAL_OVERLAP_RATIO = 0.35
MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO = 2.5
MAXIMUM_COMPONENT_HEIGHT_RATIO_WITHIN_LINE = 2.0
MAXIMUM_MERGED_HEIGHT_GROWTH_RATIO = 1.6
TIGHT_HORIZONTAL_PADDING_PIXELS = 1.0
TIGHT_VERTICAL_PADDING_RATIO = 0.25
CONTEXT_HORIZONTAL_PADDING_HEIGHT_RATIO = 2.0
CONTEXT_VERTICAL_PADDING_HEIGHT_RATIO = 1.5
CONTEXT_MINIMUM_PADDING_PIXELS = 8.0
TRUTH_MATCH_IOU_MINIMUM = 0.5
THRESHOLDS = (0.45, 0.55, 0.65, 0.75)
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    scene_count: int
    renderer_family: str
    degradation_family: str
    seed_offset: int
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
        "train",
        256,
        "randomized-plot-context-label-scenes-v7-train",
        "gamma-boxblur-column-noise-v7-train",
        121_000,
        (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "validation",
        64,
        "rotated-slot-context-label-scenes-v7-validation",
        "anisotropic-resample-row-fade-v7-validation",
        132_000,
        (_MEDIUM, _SEMIBOLD, _REGULAR),
        (_MEDIUM_SHA, _SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "sealed_public",
        80,
        "mirrored-plot-context-label-scenes-v7-public",
        "tonecurve-quantization-sparse-dropout-v7-public",
        143_000,
        (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR component-context V7 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-component-context-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "split_frozen_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "architecture": "dual-context-component-proposal-cnn-v1",
        "distinct_from": [
            "component-line-proposal-binary-cnn-v1",
            "tiny-strided-encoder-decoder-probability-map-v1",
            "skip-connected-balanced-probability-map-v1",
            "stride-four-high-resolution-db-map-v1",
            "db-two-head-differentiable-binarization-v1",
        ],
        "trigger_evidence": {
            "prior_revision": "graph-text-component-region-v6",
            "public_scene_count": 64,
            "public_false_regions": 7,
            "public_missed_regions": 0,
            "sample_or_pixel_inspection_used": False,
        },
        "proposal_algorithm": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "foreground_threshold": "clamp(round(mean_gray*0.8),32,224)",
            "connectivity": 4,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "maximum_component_width_ratio": MAXIMUM_COMPONENT_WIDTH_RATIO,
            "maximum_component_height_ratio": MAXIMUM_COMPONENT_HEIGHT_RATIO,
            "minimum_vertical_overlap_ratio": MINIMUM_VERTICAL_OVERLAP_RATIO,
            "maximum_horizontal_gap_height_ratio": MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO,
            "maximum_component_height_ratio_within_line": MAXIMUM_COMPONENT_HEIGHT_RATIO_WITHIN_LINE,
            "maximum_merged_height_growth_ratio": MAXIMUM_MERGED_HEIGHT_GROWTH_RATIO,
            "ordering": "top,left,bottom,right",
        },
        "input": ["proposal_count", INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH],
        "output": ["proposal_count", 2],
        "preprocessing": {
            "source": "immutable original-pixel Gray8 raster",
            "channels": ["tight_proposal_ink", "expanded_scene_context_ink"],
            "crop_resize_mode": "preserve_aspect_ratio_pad",
            "crop_width": CROP_WIDTH,
            "crop_height": CROP_HEIGHT,
            "tight_horizontal_padding_pixels": TIGHT_HORIZONTAL_PADDING_PIXELS,
            "tight_vertical_padding_ratio": TIGHT_VERTICAL_PADDING_RATIO,
            "context_horizontal_padding_height_ratio": CONTEXT_HORIZONTAL_PADDING_HEIGHT_RATIO,
            "context_vertical_padding_height_ratio": CONTEXT_VERTICAL_PADDING_HEIGHT_RATIO,
            "context_minimum_padding_pixels": CONTEXT_MINIMUM_PADDING_PIXELS,
            "padding_value": 255,
            "image_encoding": "1-gray/255",
            "geometry_feature_count": GEOMETRY_FEATURE_COUNT,
        },
        "selection_thresholds": list(THRESHOLDS),
        "selection_gates": {
            "exact_region_count_every_fixture": True,
            "false_region_count": 0,
            "missed_region_count": 0,
            "duplicate_region_count": 0,
            "prohibited_structure_hits": 0,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
            "provider": "CPUExecutionProvider",
        },
        "splits": [asdict(item) for item in SPLITS],
        "data_scope": (
            "new procedural numeric and generic-word graph scenes only; no Chandler, Generalization, "
            "private or article images, external datasets, pretrained weights, downloaded training data, "
            "or predecessor public fixture bytes"
        ),
        "license": "Apache-2.0",
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "font_notice_sha256": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
        "production_approval": False,
        "release_eligible": False,
    }
