# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen, authorization-free protocol for OCR proposal detector V37."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-degradation-coverage-detector-v37"
CANDIDATE_ID = "P1"
EXPERIMENT_BUDGET = 1
STATE = "prepared_not_authorized"
MODEL_LICENSE = "Apache-2.0"

V35_RESULT_PATH = Path("ml/ocr/real_range_detector_v35/P1_RESULT.json")
V35_RESULT_SHA256 = "519b003c2155153e0ed152e26f19471511e0923cd20045f0e233ccecb63a8ed8"
V35_DIAGNOSTIC_PATH = Path("ml/ocr/real_range_detector_v35/diagnostics/DIAGNOSTIC.json")
V35_DIAGNOSTIC_SHA256 = "c6b035f8bac27f27d2b157a2af015f12f600d1287b1135ce4f0cddfbd7d21526"
V36_RESULT_PATH = Path("ml/ocr/shrink_region_detector_v36/P1_RESULT.json")
V36_RESULT_SHA256 = "95f30817aa12908eca46e9871c0517e3905b047c4bde0fbf80c98da0e15e47d9"
V36_DIAGNOSTIC_PATH = Path("ml/ocr/shrink_region_detector_v36/diagnostics/DIAGNOSTIC.json")
V36_DIAGNOSTIC_SHA256 = "f4b20a000a70221e60187bad48dee5ebbb4d37180eebf25a87af9bd8e423e2e2"

DATA_REVISION = "graph-text-real-range-classifier-finetune-v32"
DATA_MODULE = Path("ml/ocr/real_range_classifier_finetune_v32/dataset.py")
V32_TRAIN_SEED = 32031
V32_DEV_SEED = 32032
V32_TRAIN_SCENE_COUNT = 5
V32_DEV_SCENE_COUNT = 5
EXPECTED_V32_TRAIN_FINGERPRINT = "6e33a247078508e5c1c38801eb6a769212d76a4b092e54b14e8cd4e745b7b70a"
EXPECTED_V32_DEV_FINGERPRINT = "67952b4575972542087281b2c14958e86518ae0e12e88d43f5c47c16252a3687"

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
ONNX_PROVIDER = "CPUExecutionProvider"
ONNX_PARITY_TOLERANCE = 0.00001
PARITY_BATCH_SIZES = (1, 7, 64)

# JPEG quality is the measured aggregate envelope in the corrected generator.
# The remaining values are bounded train-only perturbation controls; they do
# not read or estimate any private image property.
DEGRADATION_GRIDS: dict[str, tuple[float, ...]] = {
    "blur": (0.50, 0.875, 1.25, 1.625, 2.00),
    "contrast": (0.60, 0.80, 1.00, 1.20, 1.40),
    "gaussian_noise": (2.0, 5.5, 9.0, 12.5, 16.0),
    "jpeg": (55.0, 62.5, 70.0, 77.5, 85.0),
}
DEGRADATION_KINDS = tuple(DEGRADATION_GRIDS)
TRAIN_VARIANT_SEEDS = tuple(2_026_100_000 + index for index in range(V32_TRAIN_SCENE_COUNT))
TRAIN_VARIANTS_PER_BASE = 1
TRAIN_SCENE_COUNT = V32_TRAIN_SCENE_COUNT * (TRAIN_VARIANTS_PER_BASE + 1)
CANONICAL_OUTPUT = Path("ml/ocr/degradation_coverage_detector_v37/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    data_revision: str
    data_module: str
    family_disjoint: bool
    public_or_sealed_reads: int
    dev_augmentation: bool


SPLITS = (
    SplitRegistration("train", DATA_REVISION, DATA_MODULE.as_posix(), True, 0, False),
    SplitRegistration("dev", DATA_REVISION, DATA_MODULE.as_posix(), True, 0, False),
)


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-degradation-coverage-detector-v37-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "state": STATE,
        "experiment_budget": EXPERIMENT_BUDGET,
        "evidence_policy": "ml/policy/evidence-policy.json",
        "acceptance_bars": "ml/policy/acceptance-bars.json",
        "trigger_evidence": {
            "v35_result_path": V35_RESULT_PATH.as_posix(),
            "v35_result_sha256": V35_RESULT_SHA256,
            "v35_diagnostic_path": V35_DIAGNOSTIC_PATH.as_posix(),
            "v35_diagnostic_sha256": V35_DIAGNOSTIC_SHA256,
            "v36_result_path": V36_RESULT_PATH.as_posix(),
            "v36_result_sha256": V36_RESULT_SHA256,
            "v36_diagnostic_path": V36_DIAGNOSTIC_PATH.as_posix(),
            "v36_diagnostic_sha256": V36_DIAGNOSTIC_SHA256,
            "v35_full_box_failure": True,
            "v36_shrink_core_failure": True,
            "v36_expansion_oracle_passed": True,
            "v36_tiling_coverage_passed": True,
        },
        "isolated_change": "retain V35 full-box supervision and add deterministic train-only blur, contrast, Gaussian-noise, and JPEG variants derived from V32 train scenes",
        "retained_v35_contract": {
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
        "data": {
            "base_revision": DATA_REVISION,
            "base_train_seed": V32_TRAIN_SEED,
            "base_dev_seed": V32_DEV_SEED,
            "base_train_scene_count": V32_TRAIN_SCENE_COUNT,
            "train_scene_count": TRAIN_SCENE_COUNT,
            "train_variants_per_base": TRAIN_VARIANTS_PER_BASE,
            "train_variant_seeds": list(TRAIN_VARIANT_SEEDS),
            "train_variant_recipe": "sequential_blur_contrast_gaussian_noise_jpeg",
            "degradation_grids": {key: list(values) for key, values in DEGRADATION_GRIDS.items()},
            "dev_passthrough": True,
            "expected_base_train_fingerprint": EXPECTED_V32_TRAIN_FINGERPRINT,
            "expected_dev_fingerprint": EXPECTED_V32_DEV_FINGERPRINT,
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
        "real_reads": 0,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "ARCHITECTURE", "BATCH_SIZE", "CANONICAL_OUTPUT", "DEGRADATION_GRIDS",
    "DEGRADATION_KINDS", "EPOCHS", "EXPERIMENT_BUDGET", "EXPECTED_V32_DEV_FINGERPRINT",
    "EXPECTED_V32_TRAIN_FINGERPRINT", "INPUT_CHANNELS", "LEARNING_RATE",
    "MINIMUM_COMPONENT_AREA", "MODEL_LICENSE", "ONNX_PARITY_TOLERANCE", "ONNX_PROVIDER",
    "PARITY_BATCH_SIZES", "PIXEL_THRESHOLD", "POSITIVE_WEIGHT", "PRECISION_MINIMUM",
    "RECALL_MINIMUM", "REVISION", "SEED", "TASK", "TILE_OVERLAP", "TILE_SIZE",
    "TRAIN_SCENE_COUNT", "TRAIN_VARIANT_SEEDS", "TRAIN_VARIANTS_PER_BASE",
    "TRUTH_MATCH_IOU_MINIMUM", "V35_DIAGNOSTIC_PATH", "V35_DIAGNOSTIC_SHA256",
    "V35_RESULT_PATH", "V35_RESULT_SHA256", "V36_DIAGNOSTIC_PATH", "V36_DIAGNOSTIC_SHA256",
    "V36_RESULT_PATH", "V36_RESULT_SHA256", "WEIGHT_DECAY", "protocol_configuration",
]
