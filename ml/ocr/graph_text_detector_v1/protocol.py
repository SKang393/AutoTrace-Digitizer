# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the first project graph text-region detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-region-detector-v1"
SEED = 20260901
EXPERIMENT_BUDGET = 3
CANDIDATE_ID = "P1"
PATCH_WIDTH = 256
PATCH_HEIGHT = 128
TRAIN_SAMPLE_COUNT = 512
VALIDATION_TEXT_COUNT = 72
VALIDATION_EXCLUSION_COUNT = 24
SEALED_TEXT_COUNT = 96
SEALED_EXCLUSION_COUNT = 32
EPOCHS = 24
BATCH_SIZE = 8
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
PROBABILITY_THRESHOLD = 0.30
BOX_CONFIDENCE_THRESHOLD = 0.60
UNCLIP_RATIO = 1.5
MINIMUM_SIDE_LENGTH = 3
MAXIMUM_REGIONS = 1000
MATCH_IOU_MINIMUM = 0.50
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-4
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_detector_v1/artifacts/P1-run")


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
        "tensor-space-mixed-label-patches-v1",
        "train-ink-bleed-speckle-resample-v1",
        11_000,
        TRAIN_SAMPLE_COUNT * 3 // 4,
        TRAIN_SAMPLE_COUNT // 4,
        True,
    ),
    SplitRegistration(
        "validation",
        "asymmetric-panel-frame-v1",
        "validation-fax-gamma-banding-v1",
        23_000,
        VALIDATION_TEXT_COUNT,
        VALIDATION_EXCLUSION_COUNT,
        True,
    ),
    SplitRegistration(
        "sealed_public",
        "border-offset-multipanel-frame-v1",
        "sealed-public-perspective-dropout-v1",
        37_000,
        SEALED_TEXT_COUNT,
        SEALED_EXCLUSION_COUNT,
        False,
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown graph text detector split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-graph-text-detector-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": CANDIDATE_ID,
        "architecture": "tiny-strided-encoder-decoder-probability-map-v1",
        "trigger": {
            "diagnostic_report_sha256": "frozen_by_prepare_split",
            "official_detector_text_detection_exact_rate": 0.22916666666666666,
            "official_detector_composition_exact_rate": 0.4861111111111111,
            "official_detector_false_region_count": 9,
            "strict_probability_violation_count": 0,
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
            "candidate_architecture_changes": "require_next_candidate_id",
        },
        "data_scope": (
            "procedurally rendered generic graph labels and exclusion structures only; "
            "no Chandler, Generalization, private images, article images, external datasets, "
            "pretrained weights, or exposed diagnostic fixtures"
        ),
        "license": "Apache-2.0",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "BATCH_SIZE",
    "CANONICAL_OUTPUT",
    "CANDIDATE_ID",
    "EPOCHS",
    "EXPERIMENT_BUDGET",
    "LEARNING_RATE",
    "MATCH_IOU_MINIMUM",
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
    "VALIDATION_EXCLUSION_COUNT",
    "VALIDATION_TEXT_COUNT",
    "WEIGHT_DECAY",
    "protocol_configuration",
    "split_registration",
]
