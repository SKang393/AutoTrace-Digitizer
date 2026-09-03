# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen, authorization-free protocol for OCR detector V38."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-dice-loss-detector-v38"
CANDIDATE_ID = "P1"
EXPERIMENT_BUDGET = 1
STATE = "prepared_not_authorized"
MODEL_LICENSE = "Apache-2.0"

V37_RESULT_PATH = Path("ml/ocr/degradation_coverage_detector_v37/P1_RESULT.json")
V37_RESULT_SHA256 = "dbc6978b3bcd7ca722c73442d440d56ee047f162d4e903ea5f2992e009a4cb5e"
V37_DIAGNOSTIC_PATH = Path("ml/ocr/degradation_coverage_detector_v37/diagnostics/DIAGNOSTIC.json")
V37_DIAGNOSTIC_SHA256 = "73f2a248a2b86d60cb5ddb04f75b10c5a09244b905135289d64773208e6129ba"

DATA_REVISION = "graph-text-real-range-classifier-finetune-v32"
V32_TRAIN_SEED = 32031
V32_DEV_SEED = 32032
V32_TRAIN_SCENE_COUNT = 5
V32_DEV_SCENE_COUNT = 5
EXPECTED_V32_TRAIN_FINGERPRINT = "6e33a247078508e5c1c38801eb6a769212d76a4b092e54b14e8cd4e745b7b70a"
EXPECTED_V32_DEV_FINGERPRINT = "67952b4575972542087281b2c14958e86518ae0e12e88d43f5c47c16252a3687"
EXPECTED_V37_TRAIN_FINGERPRINT = "1773172bcaac6d636ee1b9482e47ec13592c3986190160b04674b3488966809c"

ARCHITECTURE = "detail-skip-source-scale-segmentation-v1"
INPUT_CHANNELS = 1
TILE_SIZE = 256
TILE_OVERLAP = 64
PIXEL_THRESHOLD = 0.40
MINIMUM_COMPONENT_AREA = 8
TRUTH_MATCH_IOU_MINIMUM = 0.50
PRECISION_MINIMUM = 0.95
RECALL_MINIMUM = 0.95
SEED = 20260935
EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
POSITIVE_WEIGHT = 4.0
BCE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0
DICE_EPSILON = 1e-6
ONNX_PROVIDER = "CPUExecutionProvider"
ONNX_PARITY_TOLERANCE = 0.00001
PARITY_BATCH_SIZES = (1, 7, 64)

TRAIN_VARIANTS_PER_BASE = 1
TRAIN_SCENE_COUNT = V32_TRAIN_SCENE_COUNT * (TRAIN_VARIANTS_PER_BASE + 1)
CANONICAL_OUTPUT = Path("ml/ocr/dice_loss_detector_v38/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    data_revision: str
    data_module: str
    family_disjoint: bool
    public_or_sealed_reads: int
    dev_augmentation: bool


SPLITS = (
    SplitRegistration("train", DATA_REVISION, "ml/ocr/degradation_coverage_detector_v37/dataset.py", True, 0, False),
    SplitRegistration("dev", DATA_REVISION, "ml/ocr/real_range_classifier_finetune_v32/dataset.py", True, 0, False),
)


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-dice-loss-detector-v38-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "state": STATE,
        "experiment_budget": EXPERIMENT_BUDGET,
        "evidence_policy": "ml/policy/evidence-policy.json",
        "acceptance_bars": "ml/policy/acceptance-bars.json",
        "trigger_evidence": {
            "v37_result_path": V37_RESULT_PATH.as_posix(),
            "v37_result_sha256": V37_RESULT_SHA256,
            "v37_diagnostic_path": V37_DIAGNOSTIC_PATH.as_posix(),
            "v37_diagnostic_sha256": V37_DIAGNOSTIC_SHA256,
            "v37_full_box_pixel_failure": True,
            "v37_tiling_coverage_passed": True,
        },
        "isolated_change": "replace V37 positive-weighted BCE pixel objective with a fixed equal-weight sum of the same BCE and batch soft-Dice loss",
        "retained_v37_contract": {
            "architecture": ARCHITECTURE,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "pixel_threshold": PIXEL_THRESHOLD,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "positive_weight": POSITIVE_WEIGHT,
            "onnx_provider": ONNX_PROVIDER,
            "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
            "parity_batch_sizes": list(PARITY_BATCH_SIZES),
        },
        "pixel_objective": {
            "bce_weight": BCE_LOSS_WEIGHT,
            "dice_weight": DICE_LOSS_WEIGHT,
            "dice_reduction": "batch",
            "dice_epsilon": DICE_EPSILON,
            "dice_formula": "1 - (2 * sum(sigmoid(logits) * target) + epsilon) / (sum(sigmoid(logits)) + sum(target) + epsilon)",
        },
        "data": {
            "base_revision": DATA_REVISION,
            "base_train_seed": V32_TRAIN_SEED,
            "base_dev_seed": V32_DEV_SEED,
            "base_train_scene_count": V32_TRAIN_SCENE_COUNT,
            "train_scene_count": TRAIN_SCENE_COUNT,
            "train_variants_per_base": TRAIN_VARIANTS_PER_BASE,
            "v37_train_split_fingerprint": EXPECTED_V37_TRAIN_FINGERPRINT,
            "expected_base_train_fingerprint": EXPECTED_V32_TRAIN_FINGERPRINT,
            "expected_dev_fingerprint": EXPECTED_V32_DEV_FINGERPRINT,
            "dev_passthrough": True,
        },
        "splits": [asdict(item) for item in SPLITS],
        "selection_gates": {
            "raw_proposal_precision_minimum": PRECISION_MINIMUM,
            "raw_proposal_recall_minimum": RECALL_MINIMUM,
            "onnx_provider": ONNX_PROVIDER,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_TOLERANCE,
            "public_or_sealed_reads": 0,
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "public_or_sealed_reads": 0,
        "real_reads": 0,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "ARCHITECTURE", "BATCH_SIZE", "BCE_LOSS_WEIGHT", "CANONICAL_OUTPUT", "CANDIDATE_ID",
    "DICE_EPSILON", "DICE_LOSS_WEIGHT", "EPOCHS", "EXPECTED_V32_DEV_FINGERPRINT",
    "EXPECTED_V32_TRAIN_FINGERPRINT", "EXPECTED_V37_TRAIN_FINGERPRINT", "EXPERIMENT_BUDGET",
    "INPUT_CHANNELS", "LEARNING_RATE", "MINIMUM_COMPONENT_AREA", "MODEL_LICENSE",
    "ONNX_PARITY_TOLERANCE", "ONNX_PROVIDER", "PARITY_BATCH_SIZES", "PIXEL_THRESHOLD",
    "POSITIVE_WEIGHT", "PRECISION_MINIMUM", "RECALL_MINIMUM", "REVISION", "SEED", "TASK",
    "TILE_OVERLAP", "TILE_SIZE", "TRAIN_SCENE_COUNT", "TRUTH_MATCH_IOU_MINIMUM",
    "V37_DIAGNOSTIC_PATH", "V37_DIAGNOSTIC_SHA256", "V37_RESULT_PATH", "V37_RESULT_SHA256",
    "WEIGHT_DECAY", "protocol_configuration",
]
