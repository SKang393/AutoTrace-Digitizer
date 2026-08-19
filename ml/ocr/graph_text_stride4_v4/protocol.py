# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the stride-4 graph text detector V4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-stride4-v4"
SEED = 20260911
EXPERIMENT_BUDGET = 3
CANDIDATE_ID = "P1"
FRAME_WIDTH = 384
FRAME_HEIGHT = 192
PATCH_WIDTH = 512
PATCH_HEIGHT = 192
TRAIN_SOURCE_COUNT = 640
TILES_PER_SOURCE = 3
TRAIN_SAMPLE_COUNT = TRAIN_SOURCE_COUNT * TILES_PER_SOURCE
VALIDATION_TEXT_COUNT = 96
VALIDATION_EXCLUSION_COUNT = 40
SEALED_TEXT_COUNT = 128
SEALED_EXCLUSION_COUNT = 48
EPOCHS = 12
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BOUNDARY_PROBABILITY_CEILING = 0.25
BOUNDARY_MARGIN_LOSS_WEIGHT = 1.0
DB_SHRINK_RATIO = 0.40
IGNORE_BAND_EXPANSION_PIXELS = 1
PROBABILITY_THRESHOLD = 0.30
BOX_CONFIDENCE_THRESHOLD = 0.60
UNCLIP_RATIO = 1.5
MINIMUM_SIDE_LENGTH = 3
MAXIMUM_REGIONS = 1000
MATCH_IOU_MINIMUM = 0.50
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-4
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_stride4_v4/artifacts/P1-run")


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
        "staggered-rail-callout-v4",
        "train-contrast-blockdrop-resample-v4",
        111_000,
        TRAIN_SOURCE_COUNT * 3 // 4,
        TRAIN_SOURCE_COUNT // 4,
        True,
    ),
    SplitRegistration(
        "validation",
        "split-legend-chevron-grid-v4",
        "validation-defocus-ringing-channel-v4",
        127_000,
        VALIDATION_TEXT_COUNT,
        VALIDATION_EXCLUSION_COUNT,
        True,
    ),
    SplitRegistration(
        "sealed_public",
        "offset-capsule-fan-grid-v4",
        "sealed-speckle-median-poster-v4",
        149_000,
        SEALED_TEXT_COUNT,
        SEALED_EXCLUSION_COUNT,
        False,
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown stride-4 detector split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-graph-text-stride4-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": CANDIDATE_ID,
        "architecture": "fine-skip-stride4-probability-map-v1",
        "trigger": {
            "prior_revision": "graph-text-ignore-band-v3",
            "prior_candidate": "P3",
            "prior_exact_fixture_count": 92,
            "prior_fixture_count": 112,
            "prior_false_region_count": 2,
            "prior_exclusion_false_region_count": 0,
            "prior_text_missed_fixture_count": 19,
            "defect_class": (
                "Eightfold spatial downsampling suppresses thin or low-contrast text cores before "
                "the high-precision ignored-boundary objective can retain them"
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
            "source_count": TRAIN_SOURCE_COUNT,
            "tiles_per_source": TILES_PER_SOURCE,
            "sample_count": TRAIN_SAMPLE_COUNT,
            "patch_width": PATCH_WIDTH,
            "patch_height": PATCH_HEIGHT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "boundary_probability_ceiling": BOUNDARY_PROBABILITY_CEILING,
            "boundary_margin_loss_weight": BOUNDARY_MARGIN_LOSS_WEIGHT,
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
            "new procedural generic graph labels and exclusion structures only; no V1, V2, or V3 "
            "selection or sealed fixtures, Chandler, Generalization, private images, article images, "
            "external datasets, pretrained weights, or downloaded training data"
        ),
        "license": "Apache-2.0",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [name for name in globals() if name.isupper()] + ["protocol_configuration", "split_registration"]
