# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the ignore-band graph text detector V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-ignore-band-v3"
SEED = 20260907
EXPERIMENT_BUDGET = 3
CANDIDATE_ID = "P1"
FRAME_WIDTH = 384
FRAME_HEIGHT = 192
PATCH_WIDTH = 512
PATCH_HEIGHT = 192
TRAIN_SAMPLE_COUNT = 640
VALIDATION_TEXT_COUNT = 80
VALIDATION_EXCLUSION_COUNT = 32
SEALED_TEXT_COUNT = 112
SEALED_EXCLUSION_COUNT = 40
EPOCHS = 36
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DICE_LOSS_WEIGHT = 1.0
NEGATIVE_TO_POSITIVE_RATIO = 3
MINIMUM_NEGATIVE_PIXELS = 1024
EMPTY_TARGET_NEGATIVE_PIXELS = 4096
IGNORE_BAND_EXPANSION_PIXELS = 1
DB_SHRINK_RATIO = 0.40
PROBABILITY_THRESHOLD = 0.30
BOX_CONFIDENCE_THRESHOLD = 0.60
UNCLIP_RATIO = 1.5
MINIMUM_SIDE_LENGTH = 3
MAXIMUM_REGIONS = 1000
MATCH_IOU_MINIMUM = 0.50
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-4
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_ignore_band_v3/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    renderer_family: str
    degradation_family: str
    seed_offset: int
    text_count: int
    exclusion_count: int
    selection_visible: bool


SPLITS = (
    SplitRegistration(
        "train",
        "offset-key-hook-frame-v3",
        "train-gamma-jpeg-anisotropic-v3",
        71_000,
        TRAIN_SAMPLE_COUNT * 3 // 4,
        TRAIN_SAMPLE_COUNT // 4,
        True,
    ),
    SplitRegistration(
        "validation",
        "notched-margin-crosshatch-frame-v3",
        "validation-channel-shift-soft-raster-v3",
        83_000,
        VALIDATION_TEXT_COUNT,
        VALIDATION_EXCLUSION_COUNT,
        True,
    ),
    SplitRegistration(
        "sealed_public",
        "stepped-capsule-diamond-frame-v3",
        "sealed-public-posterize-dropout-median-v3",
        97_000,
        SEALED_TEXT_COUNT,
        SEALED_EXCLUSION_COUNT,
        False,
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown ignore-band detector split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-graph-text-ignore-band-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": CANDIDATE_ID,
        "architecture": "skip-connected-probability-map-v2",
        "trigger": {
            "prior_revision": "graph-text-balanced-recall-v2",
            "prior_candidate": "P3",
            "prior_exact_fixture_count": 30,
            "prior_fixture_count": 96,
            "prior_false_region_count": 82,
            "prior_exclusion_false_region_count": 20,
            "prior_text_missed_fixture_count": 13,
            "defect_class": (
                "DB core supervision labels visible glyph and antialias pixels outside the shrunken "
                "positive core as background, coupling recall failure to false-region suppression"
            ),
        },
        "input": ["batch", 3, "H", "W"],
        "output": ["batch", 1, "H", "W"],
        "preprocessing": {
            "channel_order": "BGR",
            "channel_means": [0.485, 0.456, 0.406],
            "channel_scales": [1 / 0.229, 1 / 0.224, 1 / 0.225],
            "maximum_side_length": 960,
            "dimension_multiple": 128,
        },
        "postprocessing": {
            "algorithm": "db_postprocess_v1",
            "score_mode": "fast",
            "probability_threshold": PROBABILITY_THRESHOLD,
            "box_confidence_threshold": BOX_CONFIDENCE_THRESHOLD,
            "unclip_ratio": UNCLIP_RATIO,
            "minimum_side_length": MINIMUM_SIDE_LENGTH,
            "maximum_regions": MAXIMUM_REGIONS,
        },
        "training": {
            "patch_width": PATCH_WIDTH,
            "patch_height": PATCH_HEIGHT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dice_loss_weight": DICE_LOSS_WEIGHT,
            "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
            "minimum_negative_pixels": MINIMUM_NEGATIVE_PIXELS,
            "empty_target_negative_pixels": EMPTY_TARGET_NEGATIVE_PIXELS,
            "ignore_band_expansion_pixels": IGNORE_BAND_EXPANSION_PIXELS,
            "db_shrink_ratio": DB_SHRINK_RATIO,
            "seed": SEED,
        },
        "splits": [asdict(item) for item in SPLITS],
        "selection_gates": {
            "exact_region_count_every_fixture": True,
            "duplicate_region_count": 0,
            "false_region_count": 0,
            "exclusion_false_region_count": 0,
            "truth_match_iou_minimum": MATCH_IOU_MINIMUM,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
            "provider": "CPUExecutionProvider",
        },
        "fixed_experiment_policy": {
            "threshold_sweeps": 0,
            "validation_regeneration_after_training": 0,
            "sealed_public_evaluations_before_selection": 0,
            "candidate_changes": "one preregistered isolated change per subsequent candidate",
        },
        "data_scope": (
            "new procedurally rendered generic graph labels and exclusion structures only; no V1 or V2 "
            "selection or sealed fixtures, Chandler, Generalization, private images, article images, "
            "external datasets, pretrained weights, or downloaded training data"
        ),
        "license": "Apache-2.0",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "BATCH_SIZE",
    "BOX_CONFIDENCE_THRESHOLD",
    "CANDIDATE_ID",
    "CANONICAL_OUTPUT",
    "DB_SHRINK_RATIO",
    "DICE_LOSS_WEIGHT",
    "EMPTY_TARGET_NEGATIVE_PIXELS",
    "EPOCHS",
    "EXPERIMENT_BUDGET",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "IGNORE_BAND_EXPANSION_PIXELS",
    "LEARNING_RATE",
    "MATCH_IOU_MINIMUM",
    "MAXIMUM_REGIONS",
    "MINIMUM_NEGATIVE_PIXELS",
    "MINIMUM_SIDE_LENGTH",
    "NEGATIVE_TO_POSITIVE_RATIO",
    "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR",
    "PATCH_HEIGHT",
    "PATCH_WIDTH",
    "PROBABILITY_THRESHOLD",
    "REVISION",
    "SEALED_EXCLUSION_COUNT",
    "SEALED_TEXT_COUNT",
    "SEED",
    "TASK",
    "TRAIN_SAMPLE_COUNT",
    "UNCLIP_RATIO",
    "VALIDATION_EXCLUSION_COUNT",
    "VALIDATION_TEXT_COUNT",
    "WEIGHT_DECAY",
    "protocol_configuration",
    "split_registration",
]
