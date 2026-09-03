# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen preparation protocol for the V36 shrink-region detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-shrink-region-detector-v36"
CANDIDATE_ID = "P1"
EXPERIMENT_BUDGET = 1
SEED = 20260935
DATA_REVISION = "graph-text-real-range-classifier-finetune-v32"
DATA_MODULE = Path("ml/ocr/real_range_classifier_finetune_v32/dataset.py")
V35_RESULT_PATH = Path("ml/ocr/real_range_detector_v35/P1_RESULT.json")
V35_RESULT_SHA256 = "519b003c2155153e0ed152e26f19471511e0923cd20045f0e233ccecb63a8ed8"
V35_DIAGNOSTIC_PATH = Path("ml/ocr/real_range_detector_v35/diagnostics/DIAGNOSTIC.json")
V35_DIAGNOSTIC_SHA256 = "c6b035f8bac27f27d2b157a2af015f12f600d1287b1135ce4f0cddfbd7d21526"
MODEL_LICENSE = "Apache-2.0"
TILE_SIZE = 256
TILE_OVERLAP = 64
INPUT_CHANNELS = 1
TRUTH_MATCH_IOU_MINIMUM = 0.50
PIXEL_THRESHOLD = 0.40
MINIMUM_COMPONENT_AREA = 1
PRECISION_MINIMUM = 0.95
RECALL_MINIMUM = 0.95
EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
POSITIVE_WEIGHT = 4.0
ONNX_PARITY_TOLERANCE = 0.00001
ONNX_PROVIDER = "CPUExecutionProvider"
PARITY_BATCH_SIZES = (1, 7, 64)
DB_SHRINK_RATIO = 0.40
CANONICAL_OUTPUT = Path("ml/ocr/shrink_region_detector_v36/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    data_revision: str
    data_module: str
    family_disjoint: bool
    public_or_sealed_reads: int


SPLITS = (
    SplitRegistration("train", DATA_REVISION, DATA_MODULE.as_posix(), True, 0),
    SplitRegistration("dev", DATA_REVISION, DATA_MODULE.as_posix(), True, 0),
)


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-shrink-region-detector-v36-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "experiment_budget": EXPERIMENT_BUDGET,
        "state": "preregistered_before_training",
        "evidence_policy": "ml/policy/evidence-policy.json",
        "acceptance_bars": "ml/policy/acceptance-bars.json",
        "trigger_evidence": {
            "v35_result_path": V35_RESULT_PATH.as_posix(),
            "v35_result_sha256": V35_RESULT_SHA256,
            "v35_diagnostic_path": V35_DIAGNOSTIC_PATH.as_posix(),
            "v35_diagnostic_sha256": V35_DIAGNOSTIC_SHA256,
            "v35_raw_proposal_precision": 0.20512820512820512,
            "v35_raw_proposal_recall": 0.46511627906976744,
            "v35_interpretation": "source-scale full-box pixel segmentation and its connected-component recovery missed regions; threshold or morphology sweeps did not recover the ceiling",
        },
        "isolated_change": "replace full-box mask targets and full-mask component recovery with per-truth DB-style shrunken rectangular cores and deterministic paired expansion to source boxes",
        "architecture": {
            "input": ["tile_count", INPUT_CHANNELS, TILE_SIZE, TILE_SIZE],
            "output": ["tile_count", 1, TILE_SIZE, TILE_SIZE],
            "model": "detail-skip-source-scale-segmentation-v1",
            "training_from_scratch": True,
            "model_license": MODEL_LICENSE,
        },
        "tile_contract": {
            "tile_size": TILE_SIZE,
            "overlap": TILE_OVERLAP,
            "coordinate_space": "original_pixels",
            "edge_tiles_padded": True,
            "merge": "overlap_average_then_threshold",
        },
        "db_geometry": {
            "shrink_ratio": DB_SHRINK_RATIO,
            "representation": "independent_axis_aligned_rectangular_core_per_truth_box",
            "distance": "centered_axis_ratio_per_width_and_height",
            "rounding": "round_core_extent_then_split_integer_insets",
            "expansion": "fixed_ratio_center_expansion_with_half_up_extent_rounding_and_clipping",
            "minimum_core_extent": 1,
            "overlap_resolution": "no_morphology; connected cores remain separate before expansion",
            "reversible_transform": "ShrinkGeometry retains integer side insets for exact audit reversal; predicted cores use the same fixed axis ratio for deterministic approximate source-box recovery",
        },
        "postprocessing": {
            "pixel_threshold": PIXEL_THRESHOLD,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "connectivity": 8,
            "box_coordinates": "thresholded core components expanded by half-up full-extent recovery and clipped to original pixels",
        },
        "splits": [asdict(item) for item in SPLITS],
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "positive_weight": POSITIVE_WEIGHT,
            "selection": "lowest aggregate train loss only",
        },
        "selection_gates": {
            "raw_proposal_precision_minimum": PRECISION_MINIMUM,
            "raw_proposal_recall_minimum": RECALL_MINIMUM,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
            "onnx_provider": ONNX_PROVIDER,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_TOLERANCE,
            "parity_batch_sizes": list(PARITY_BATCH_SIZES),
            "public_or_sealed_reads": 0,
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "BATCH_SIZE", "CANONICAL_OUTPUT", "CANDIDATE_ID", "DB_SHRINK_RATIO", "EPOCHS",
    "EXPERIMENT_BUDGET", "INPUT_CHANNELS", "LEARNING_RATE", "MINIMUM_COMPONENT_AREA",
    "MODEL_LICENSE", "ONNX_PARITY_TOLERANCE", "ONNX_PROVIDER", "PARITY_BATCH_SIZES",
    "PIXEL_THRESHOLD", "POSITIVE_WEIGHT", "PRECISION_MINIMUM", "RECALL_MINIMUM",
    "REVISION", "SEED", "TASK", "TILE_OVERLAP", "TILE_SIZE", "TRUTH_MATCH_IOU_MINIMUM",
    "V35_DIAGNOSTIC_PATH", "V35_DIAGNOSTIC_SHA256", "V35_RESULT_PATH", "V35_RESULT_SHA256",
    "WEIGHT_DECAY", "protocol_configuration",
]
