# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the DB-objective graph text detector V5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-db-objective-v5"
SEED = 20261017
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
WEIGHT_DECAY = 0.00005
DB_SHRINK_RATIO = 0.40
DB_BINARIZATION_K = 50.0
DB_SHRINK_LOSS_WEIGHT = 5.0
DB_THRESHOLD_LOSS_WEIGHT = 10.0
DB_BINARY_LOSS_WEIGHT = 1.0
DB_THRESHOLD_MINIMUM = 0.30
DB_THRESHOLD_MAXIMUM = 0.70
IGNORE_BAND_EXPANSION_PIXELS = 1
PROBABILITY_THRESHOLD = 0.30
BOX_CONFIDENCE_THRESHOLD = 0.60
UNCLIP_RATIO = 1.5
MINIMUM_SIDE_LENGTH = 3
MAXIMUM_REGIONS = 1000
MATCH_IOU_MINIMUM = 0.50
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-4
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_db_objective_v5/artifacts/P1-run")


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
        "cantilever-label-stair-v5",
        "train-halftone-bloom-resample-v5",
        171_000,
        TRAIN_SOURCE_COUNT * 3 // 4,
        TRAIN_SOURCE_COUNT // 4,
        True,
    ),
    SplitRegistration(
        "validation",
        "nested-capsule-crossbar-v5",
        "validation-ringing-defocus-channel-v5",
        193_000,
        VALIDATION_TEXT_COUNT,
        VALIDATION_EXCLUSION_COUNT,
        True,
    ),
    SplitRegistration(
        "sealed_public",
        "offset-rail-kite-grid-v5",
        "sealed-quantize-speckle-pulse-v5",
        227_000,
        SEALED_TEXT_COUNT,
        SEALED_EXCLUSION_COUNT,
        False,
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown DB-objective detector split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-graph-text-db-objective-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": CANDIDATE_ID,
        "architecture": "dual-head-db-stride4-v1",
        "trigger": {
            "prior_revision": "graph-text-stride4-v4",
            "prior_candidate": "P3",
            "best_prior_candidate": "P2",
            "best_prior_exact_fixture_count": 108,
            "best_prior_fixture_count": 136,
            "best_prior_text_missed_fixture_count": 24,
            "best_prior_false_region_count": 15,
            "best_prior_multi_region_fixture_count": 3,
            "defect_class": (
                "A single shrink-probability head trained without a learned threshold map cannot "
                "jointly stabilize DB contour continuity and tight region geometry at the frozen "
                "production thresholds"
            ),
        },
        "input": ["batch", 3, "H", "W"],
        "inference_output": ["batch", 1, "H", "W"],
        "training_outputs": ["shrink_map", "threshold_map", "differentiable_binary_map"],
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
            "db_shrink_ratio": DB_SHRINK_RATIO,
            "db_binarization_k": DB_BINARIZATION_K,
            "shrink_loss_weight": DB_SHRINK_LOSS_WEIGHT,
            "threshold_loss_weight": DB_THRESHOLD_LOSS_WEIGHT,
            "binary_loss_weight": DB_BINARY_LOSS_WEIGHT,
            "threshold_map_range": [DB_THRESHOLD_MINIMUM, DB_THRESHOLD_MAXIMUM],
            "ignore_band_expansion_pixels": IGNORE_BAND_EXPANSION_PIXELS,
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
            "new procedural generic graph labels and exclusion structures only; no V1, V2, V3, "
            "or V4 selection or sealed fixtures, Chandler, Generalization, private images, article "
            "images, external datasets, pretrained weights, or downloaded training data"
        ),
        "license": "Apache-2.0",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [name for name in globals() if name.isupper()] + ["protocol_configuration", "split_registration"]
