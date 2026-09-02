# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for corrected real-range classifier adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-real-range-classifier-finetune-v32"
CANDIDATE_ID = "P1"
SEED = 20260932
SOURCE_REVISION = "graph-text-spaced-component-recall-v10-p2"
SOURCE_MODEL_SEED = 20261422
SOURCE_CHECKPOINT_SHA256 = "c452e66013610a5ea79de78f6bbbac53d809c6c80f82d64798c4c694a27e7301"
SOURCE_ONNX_SHA256 = "474b8468dbd91416f4e4978dafc46cb2317775d59d821c0470e0cd3e0f6203db"
SOURCE_CHECKPOINT_PATH = Path("ml/ocr/component_spaced_recall_detector_v10/artifacts/P2-run/graph-text-spaced-component-recall-v10-p2.pt")
SOURCE_ONNX_PATH = Path("ml/ocr/component_spaced_recall_detector_v10/artifacts/P2-run/graph-text-spaced-component-recall-v10-p2.onnx")
MODEL_LICENSE = "Apache-2.0"
TRUTH_MATCH_IOU_MINIMUM = 0.50
PROPOSAL_SCORE_THRESHOLD = 0.82
PRECISION_MINIMUM = 0.95
RECALL_MINIMUM = 0.95
NEGATIVE_CAP_PER_SCENE = 64
EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00001
POSITIVE_CLASS_WEIGHT = 4.0
TRAIN_SEED = 32031
DEV_SEED = 32032
CANONICAL_OUTPUT = Path("ml/ocr/real_range_classifier_finetune_v32/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    seed: int
    scene_count: int
    renderer_families: tuple[str, ...]
    font_families: tuple[str, ...]
    degradation_families: tuple[str, ...]
    template_families: tuple[str, ...]
    marker_families: tuple[str, ...]


SPLITS = (
    SplitRegistration("train", TRAIN_SEED, 5, ("vector_clean", "print_monochrome"), ("system_sans", "system_serif"), ("none", "print_light"), ("classic_single", "stacked_shared_axes"), ("geometric_basic", "mixed_print")),
    SplitRegistration("dev", DEV_SEED, 5, ("scan_rough", "hand_drawn"), ("system_mono", "system_handwritten"), ("scan_noise", "camera_skew"), ("compact_legend", "hand_drawn_grid"), ("symbolic", "irregular")),
)


def split_registration(split: str) -> SplitRegistration:
    for item in SPLITS:
        if item.split == split:
            return item
    raise ValueError(f"Unknown V32 split: {split}")


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-real-range-classifier-finetune-v32-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "state": "preregistered_before_training",
        "defect_class": "proposal-classifier adaptation to corrected real-range source text and graph-scale distribution",
        "isolated_change": "fine-tune the existing checksum-bound V10 P2 proposal classifier on fresh family-disjoint corrected real-range synthetic proposals; retain proposal generation, threshold, and IoU metric",
        "source_model": {
            "revision": SOURCE_REVISION,
            "model_seed": SOURCE_MODEL_SEED,
            "checkpoint_path": SOURCE_CHECKPOINT_PATH.as_posix(),
            "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "onnx_sha256": SOURCE_ONNX_SHA256,
            "model_license": MODEL_LICENSE,
            "weights_reused": True,
            "train_from_scratch": False,
        },
        "splits": [asdict(item) for item in SPLITS],
        "proposal_contract": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "source": "ml/ocr/component_context_detector_v7/dataset.py",
            "proposal_score_threshold": PROPOSAL_SCORE_THRESHOLD,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
            "negative_cap_per_scene": NEGATIVE_CAP_PER_SCENE,
        },
        "training": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "positive_class_weight": POSITIVE_CLASS_WEIGHT},
        "selection_gates": {
            "dev_only_until_margin": True,
            "precision_minimum": PRECISION_MINIMUM,
            "recall_minimum": RECALL_MINIMUM,
            "onnx_parity_maximum_absolute_error": 0.00001,
            "sealed_public_evaluations": 0,
        },
        "data_scope": "fresh project-owned corrected real_range synthetic scenes; no private/article/public/sealed images, prior fixture bytes, or prior validation pixels",
        "production_approval": False,
        "release_eligible": False,
    }
