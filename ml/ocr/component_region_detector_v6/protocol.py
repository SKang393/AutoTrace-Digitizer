# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol constants for the component-proposal detector V6."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-component-region-v6"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20261111
SCENE_WIDTH = 512
SCENE_HEIGHT = 256
CROP_WIDTH = 128
CROP_HEIGHT = 32
GEOMETRY_FEATURE_COUNT = 8
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
PROPOSAL_THRESHOLD_MINIMUM = 32
PROPOSAL_THRESHOLD_MAXIMUM = 224
PROPOSAL_THRESHOLD_MEAN_RATIO = 0.8
MINIMUM_COMPONENT_AREA = 2
MAXIMUM_COMPONENT_WIDTH_RATIO = 0.15
MAXIMUM_COMPONENT_HEIGHT_RATIO = 0.20
MINIMUM_VERTICAL_OVERLAP_RATIO = 0.35
MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO = 2.5
CROP_HORIZONTAL_PADDING_PIXELS = 1.0
CROP_VERTICAL_PADDING_RATIO = 0.25
TRUTH_MATCH_IOU_MINIMUM = 0.5
THRESHOLDS = (0.45, 0.55, 0.65)
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
        192,
        "offset-margin-mixed-label-scenes-v6-train",
        "train-gradient-blur-speckle-v6",
        81_000,
        (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "validation",
        48,
        "corner-swapped-mixed-label-scenes-v6-validation",
        "validation-resample-scanline-fade-v6",
        92_000,
        (_MEDIUM, _SEMIBOLD, _REGULAR),
        (_MEDIUM_SHA, _SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "sealed_public",
        64,
        "asymmetric-margin-mixed-label-scenes-v6-public",
        "sealed-ramp-dropout-quantization-v6",
        103_000,
        (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR component-region V6 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-component-region-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "split_frozen_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": None,
        "architecture": "component-line-proposal-binary-cnn-v1",
        "distinct_from": [
            "tiny-strided-encoder-decoder-probability-map-v1",
            "skip-connected-balanced-probability-map-v1",
            "stride-four-high-resolution-db-map-v1",
            "db-two-head-differentiable-binarization-v1",
        ],
        "proposal_algorithm": {
            "algorithm": "adaptive-gray-four-connected-line-grouping-v1",
            "foreground_threshold": "clamp(round(mean_gray*0.8),32,224)",
            "connectivity": 4,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "maximum_component_width_ratio": MAXIMUM_COMPONENT_WIDTH_RATIO,
            "maximum_component_height_ratio": MAXIMUM_COMPONENT_HEIGHT_RATIO,
            "minimum_vertical_overlap_ratio": MINIMUM_VERTICAL_OVERLAP_RATIO,
            "maximum_horizontal_gap_height_ratio": MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO,
            "ordering": "top,left,bottom,right",
        },
        "input": ["proposal_count", 1, CROP_HEIGHT, ENCODED_WIDTH],
        "output": ["proposal_count", 2],
        "preprocessing": {
            "source": "immutable original-pixel Gray8 raster",
            "crop_resize_mode": "preserve_aspect_ratio_pad",
            "crop_width": CROP_WIDTH,
            "crop_height": CROP_HEIGHT,
            "horizontal_padding_pixels": CROP_HORIZONTAL_PADDING_PIXELS,
            "vertical_padding_ratio": CROP_VERTICAL_PADDING_RATIO,
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
            "or exposed predecessor fixture bytes"
        ),
        "license": "Apache-2.0",
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "font_notice_sha256": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
        "production_approval": False,
        "release_eligible": False,
    }

