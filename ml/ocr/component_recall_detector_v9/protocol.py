# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for OCR component-recall detector V9."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-component-recall-v9"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20261309
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
CROP_WIDTH = 128
CROP_HEIGHT = 32
INPUT_CHANNELS = 2
GEOMETRY_FEATURE_COUNT = 12
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE = 32
TRUTH_MATCH_IOU_MINIMUM = 0.5
THRESHOLDS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.925, 0.95)
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
        320,
        "five-role-shifted-legend-scenes-v9-train",
        "median-tone-local-fade-v9-train",
        409_000,
        (_MEDIUM, _SEMIBOLD, _REGULAR),
        (_MEDIUM_SHA, _SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "validation",
        80,
        "five-role-reflected-tick-scenes-v9-validation",
        "area-resample-row-fade-v9-validation",
        431_000,
        (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "sealed_public",
        112,
        "five-role-offset-annotation-scenes-v9-public",
        "gamma-quantization-speckle-v9-public",
        457_000,
        (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR component-recall V9 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-component-recall-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "split_frozen_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "architecture": "separated-visual-geometry-component-fusion-cnn-v1",
        "isolated_change": "new training and selection families explicitly balance all five production graph-text roles while retaining the exact V8 proposal, encoding, architecture, loss, and inference contract",
        "trigger_evidence": {
            "prior_revision": "graph-text-component-fusion-v8",
            "composition_report_sha256": "a3d851c043993bdc5546c68dfa26be837c88495600f7cbd214c55ee107ef330f",
            "validation_scene_count": 64,
            "truth_region_count": 320,
            "true_positives": 207,
            "false_negatives": 113,
            "false_positives": 0,
            "misses_by_role": {"legend_text": 56, "x_tick": 25, "y_tick": 21, "annotation": 11},
            "detected_word_exact_match": 1.0,
            "detected_numeric_exact_match": 1.0,
            "prior_public_composition_archive_opened": False,
            "prior_validation_pixels_used": False,
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
        },
        "training": {
            "loss": "unweighted cross entropy",
            "negative_proposal_cap_per_training_scene": TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE,
            "sampler": "deterministic class-balanced batches",
            "positive_role_balance": "exactly one y tick, x tick, phase heading, annotation, and legend label per scene",
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
            "fresh procedural five-role graph scenes with new seed, renderer, degradation, layout, and label-family "
            "registrations; no Chandler, Generalization, private/article images, external datasets, pretrained "
            "weights, downloaded training data, predecessor fixture bytes, or consumed V8-composition pixels"
        ),
        "license": "Apache-2.0",
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "font_notice_sha256": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "CROP_HEIGHT", "CROP_WIDTH", "ENCODED_WIDTH", "EXPERIMENT_BUDGET",
    "GEOMETRY_FEATURE_COUNT", "INPUT_CHANNELS", "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR",
    "PUBLIC_REVISION", "REVISION", "SCENE_HEIGHT", "SCENE_WIDTH", "SEED", "SPLITS",
    "TASK", "THRESHOLDS", "TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE",
    "TRUTH_MATCH_IOU_MINIMUM", "protocol_configuration", "split_registration",
]
