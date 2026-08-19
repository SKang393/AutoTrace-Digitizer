# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol constants for the OCR component-fusion detector V8."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-component-fusion-v8"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20261208
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
CROP_WIDTH = 128
CROP_HEIGHT = 32
INPUT_CHANNELS = 2
GEOMETRY_FEATURE_COUNT = 12
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE = 24
TRUTH_MATCH_IOU_MINIMUM = 0.5
THRESHOLDS = (0.55, 0.65, 0.75, 0.85, 0.90, 0.925, 0.95)
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
        "hard-negative-shape-context-scenes-v8-train",
        "median-blur-tone-ramp-v8-train",
        181_000,
        (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
    ),
    SplitRegistration(
        "validation",
        72,
        "offset-structure-context-scenes-v8-validation",
        "lanczos-resample-local-fade-v8-validation",
        193_000,
        (_REGULAR, _SEMIBOLD, _MEDIUM),
        (_REGULAR_SHA, _SEMIBOLD_SHA, _MEDIUM_SHA),
    ),
    SplitRegistration(
        "sealed_public",
        88,
        "reflected-structure-context-scenes-v8-public",
        "gamma-quantization-pixel-loss-v8-public",
        207_000,
        (_MEDIUM, _REGULAR, _SEMIBOLD),
        (_MEDIUM_SHA, _REGULAR_SHA, _SEMIBOLD_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR component-fusion V8 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-component-fusion-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "split_frozen_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "architecture": "separated-visual-geometry-component-fusion-cnn-v1",
        "distinct_from": [
            "component-line-proposal-binary-cnn-v1",
            "dual-context-component-proposal-cnn-v1",
            "tiny-strided-encoder-decoder-probability-map-v1",
            "skip-connected-balanced-probability-map-v1",
            "stride-four-high-resolution-db-map-v1",
            "db-two-head-differentiable-binarization-v1",
        ],
        "trigger_evidence": {
            "prior_revision": "graph-text-component-context-v7",
            "selection_scene_count": 64,
            "best_false_regions": 1,
            "best_missed_regions": 0,
            "best_training_loss": 2.69520371547386e-7,
            "failure": "visual and repeated geometry columns were mixed in one flattening CNN and overfit a compact structure while threshold-only calibration could not separate it",
            "public_archive_opened_by_training_or_gate": False,
            "prior_public_sample_or_pixel_inspection_used": False,
        },
        "proposal_algorithm": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "source": "ml/ocr/component_context_detector_v7/dataset.py",
            "ordering": "top,left,bottom,right",
        },
        "input": ["proposal_count", INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH],
        "output": ["proposal_count", 2],
        "preprocessing": {
            "source": "immutable original-pixel Gray8 raster",
            "source_revision": "graph-text-component-context-v7-encoding-v1",
            "channels": ["tight_proposal_ink", "expanded_scene_context_ink"],
            "crop_width": CROP_WIDTH,
            "crop_height": CROP_HEIGHT,
            "geometry_feature_count": GEOMETRY_FEATURE_COUNT,
            "geometry_extraction": "first channel, first row, final twelve columns",
        },
        "model_branches": {
            "visual": "two 32x128 image channels through three convolutional blocks and adaptive pooling",
            "geometry": "twelve scalar features through a separate MLP",
            "fusion": "concatenate visual and geometry embeddings before binary logits",
        },
        "training": {
            "loss": "unweighted cross entropy",
            "negative_proposal_cap_per_training_scene": TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE,
            "negative_selection": "first twenty-four deterministic proposal-order negatives per training scene",
            "sampler": "deterministic class-balanced batches with every retained example eligible each epoch",
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
            "new procedural numeric and generic-word graph scenes with independently varied compact polygons, "
            "markers, brackets, arrows, legends, axes, ticks, dividers, connectors, and intersections; no Chandler, "
            "Generalization, private or article images, external datasets, pretrained weights, downloaded training "
            "data, or predecessor selection/public fixture bytes"
        ),
        "license": "Apache-2.0",
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "font_notice_sha256": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "CROP_HEIGHT",
    "CROP_WIDTH",
    "ENCODED_WIDTH",
    "EXPERIMENT_BUDGET",
    "GEOMETRY_FEATURE_COUNT",
    "INPUT_CHANNELS",
    "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR",
    "PUBLIC_REVISION",
    "REVISION",
    "SCENE_HEIGHT",
    "SCENE_WIDTH",
    "SEED",
    "SPLITS",
    "TASK",
    "THRESHOLDS",
    "TRUTH_MATCH_IOU_MINIMUM",
    "TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE",
    "protocol_configuration",
    "split_registration",
]
