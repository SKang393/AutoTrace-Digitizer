# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for the project-owned real-range OCR classifier V33."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-real-range-classifier-v33"
CANDIDATE_ID = "P1"
SEED = 20260933
TRAINING_FROM_SCRATCH = True
MODEL_LICENSE = "Apache-2.0"
SOURCE_FINE_TUNE_REVISION = "graph-text-real-range-classifier-finetune-v32"
SOURCE_MODEL_REVISION = "graph-text-spaced-component-recall-v10-p2"
OFFICIAL_DIAGNOSTIC_PATH = "docs/GOAL-22-PHASE-4-CORRECTED-OCR-DIAGNOSTIC.json"
OFFICIAL_DIAGNOSTIC_SHA256 = "53e6c458f0a9688aca51595b37038811639b798eb2435e8ce653eb8501c74569"
FINE_TUNE_RESULT_PATH = "ml/ocr/real_range_classifier_finetune_v32/P1_RESULT.json"
FINE_TUNE_RESULT_SHA256 = "76795363e8d77034fa6f0f055ad59a1de7b362bbdce93ed440347dbb17a1a91d"
TRAIN_DATA_REVISION = "graph-text-real-range-classifier-finetune-v32"
CROP_HEIGHT = 32
CROP_WIDTH = 128
GEOMETRY_FEATURE_COUNT = 12
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
INPUT_CHANNELS = 2
TRUTH_MATCH_IOU_MINIMUM = 0.50
PROPOSAL_SCORE_THRESHOLD = 0.82
PRECISION_MINIMUM = 0.95
RECALL_MINIMUM = 0.95
CLASS_WEIGHTS = (1.0, 8.0)
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
ONNX_PARITY_TOLERANCE = 0.00001
CANONICAL_OUTPUT = Path("ml/ocr/real_range_classifier_v33/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    source_revision: str
    source_module: str
    family_disjoint_from: str
    sealed_public_read_count: int


SPLITS = (
    SplitRegistration("train", TRAIN_DATA_REVISION, "ml/ocr/real_range_classifier_finetune_v32/dataset.py", "dev", 0),
    SplitRegistration("dev", TRAIN_DATA_REVISION, "ml/ocr/real_range_classifier_finetune_v32/dataset.py", "train", 0),
)


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-real-range-classifier-v33-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "state": "preregistered_before_training",
        "evidence_policy": "ml/policy/evidence-policy.json",
        "defect_class": "V32 compatible V10 adaptation failed corrected real-range dev twice; proposal classifier representation remains insufficient for source-scale graph-component separation",
        "model_sourcing_decision": {
            "approved_pretrained_attempted": True,
            "approved_pretrained_result": "official DB detector failed corrected real-range synthetic evaluation",
            "approved_pretrained_result_path": OFFICIAL_DIAGNOSTIC_PATH,
            "approved_pretrained_result_sha256": OFFICIAL_DIAGNOSTIC_SHA256,
            "approved_fine_tune_attempted": True,
            "approved_fine_tune_revision": SOURCE_FINE_TUNE_REVISION,
            "approved_fine_tune_result": "two V32 dev attempts failed fixed 0.95 precision and recall bars",
            "approved_fine_tune_result_path": FINE_TUNE_RESULT_PATH,
            "approved_fine_tune_result_sha256": FINE_TUNE_RESULT_SHA256,
            "project_owned_architecture_permitted": True,
            "reason": "model sourcing order requires attempting an approved pretrained option and a compatible fine-tune before project-owned training; both prior routes failed the corrected synthetic dev gate, so V33 may use a richer project-owned architecture",
        },
        "isolated_change": "replace the failed V32 V10-compatible classifier with a richer multiscale visual and geometry fusion classifier; retain V32 family-disjoint corrected synthetic proposals, proposal generator, 0.82 operating threshold, and maximum-cardinality IoU 0.50 metric",
        "architecture": {
            "input": ["proposal_count", INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH],
            "output": ["proposal_count", 2],
            "visual_branch": "multiscale convolution with high-resolution detail and adaptive pooled context",
            "geometry_branch": "12-feature MLP",
            "fusion": "visual-detail-context-geometry concatenation",
            "training_from_scratch": TRAINING_FROM_SCRATCH,
            "model_license": MODEL_LICENSE,
        },
        "data": {
            "source_revision": TRAIN_DATA_REVISION,
            "train_module": "ml/ocr/real_range_classifier_finetune_v32/training_data.py",
            "dev_module": "ml/ocr/real_range_classifier_finetune_v32/dataset.py",
            "family_disjoint": True,
            "private_or_article_images": False,
            "public_or_sealed_reads": 0,
        },
        "proposal_contract": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "source": "ml/ocr/component_context_detector_v7/dataset.py",
            "proposal_score_threshold": PROPOSAL_SCORE_THRESHOLD,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
        },
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "class_weights": list(CLASS_WEIGHTS),
        },
        "selection_gates": {
            "dev_only_until_margin": True,
            "precision_minimum": PRECISION_MINIMUM,
            "recall_minimum": RECALL_MINIMUM,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_TOLERANCE,
            "parity_batch_sizes": [1, 7, 64, 257],
            "sealed_public_evaluations": 0,
        },
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "BATCH_SIZE", "CANDIDATE_ID", "CANONICAL_OUTPUT", "CLASS_WEIGHTS", "CROP_HEIGHT",
    "CROP_WIDTH", "EPOCHS", "ENCODED_WIDTH", "GEOMETRY_FEATURE_COUNT",
    "INPUT_CHANNELS", "LEARNING_RATE", "ONNX_PARITY_TOLERANCE", "PRECISION_MINIMUM",
    "PROPOSAL_SCORE_THRESHOLD", "RECALL_MINIMUM", "REVISION", "SEED", "TASK",
    "TRAINING_FROM_SCRATCH", "TRUTH_MATCH_IOU_MINIMUM", "WEIGHT_DECAY",
    "protocol_configuration",
]
