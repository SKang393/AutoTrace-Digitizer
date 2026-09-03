# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for the learned real-range proposal detector V35."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-real-range-detector-v35"
CANDIDATE_ID = "P1"
SEED = 20260935
DATA_REVISION = "graph-text-real-range-classifier-finetune-v32"
DATA_MODULE = Path("ml/ocr/real_range_classifier_finetune_v32/dataset.py")
V34_DIAGNOSTIC = Path("ml/ocr/real_range_proposal_v34/DEV_DIAGNOSTIC.json")
V34_DIAGNOSTIC_SHA256 = "c6f533e42de1c81ca78f3493f7ac9b42ac38147c4aa69175d3c3483ef6026b16"
V32_RESULT_PATH = Path("ml/ocr/real_range_classifier_finetune_v32/P1_RESULT.json")
V32_RESULT_SHA256 = "76795363e8d77034fa6f0f055ad59a1de7b362bbdce93ed440347dbb17a1a91d"
V33_RESULT_PATH = Path("ml/ocr/real_range_classifier_v33/P1_RESULT.json")
V33_RESULT_SHA256 = "96526456b1bc191af2f6490dd929ba35b756bd76506ac1c23d1c79393470e51b"
MODEL_LICENSE = "Apache-2.0"
TILE_SIZE = 256
TILE_OVERLAP = 64
INPUT_CHANNELS = 1
TRUTH_MATCH_IOU_MINIMUM = 0.50
PIXEL_THRESHOLD = 0.40
MINIMUM_COMPONENT_AREA = 8
PRECISION_MINIMUM = 0.95
RECALL_MINIMUM = 0.95
EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
POSITIVE_WEIGHT = 4.0
ONNX_PARITY_TOLERANCE = 0.00001
CANONICAL_OUTPUT = Path("ml/ocr/real_range_detector_v35/artifacts/P1-run")


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
        "schema": "graphreader.ocr-real-range-detector-v35-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "state": "preregistered_before_training",
        "evidence_policy": "ml/policy/evidence-policy.json",
        "acceptance_bars": "ml/policy/acceptance-bars.json",
        "defect_class": "V34 deterministic proposal expansion failed; raw V32 proposals maximum-match only 61 of 86 dev truths",
        "trigger_evidence": {
            "v34_diagnostic_path": V34_DIAGNOSTIC.as_posix(),
            "v34_diagnostic_sha256": V34_DIAGNOSTIC_SHA256,
            "v34_raw_proposal_recall": 0.7093023256,
            "v34_deterministic_expansion_failed": True,
            "v32_classifier_failure": "compatible V10 fine-tune failed two dev attempts",
            "v32_result_path": V32_RESULT_PATH.as_posix(),
            "v32_result_sha256": V32_RESULT_SHA256,
            "v33_classifier_failure": "project-owned classifier failed fixed dev bars",
            "v33_result_path": V33_RESULT_PATH.as_posix(),
            "v33_result_sha256": V33_RESULT_SHA256,
            "classifier_only_insufficient": True,
        },
        "model_sourcing_decision": {
            "approved_pretrained_attempted": True,
            "compatible_fine_tune_attempted": True,
            "project_owned_classifier_attempted": True,
            "deterministic_proposal_repair_attempted": True,
            "learned_proposal_detector_permitted": True,
            "reason": "Official, compatible fine-tune, project-owned classifier, and deterministic proposal expansion routes failed the corrected synthetic dev requirements; V35 is the permitted learned proposal-generator step.",
        },
        "isolated_change": "learn a source-scale tiled text probability map from V32 truth masks; retain V32 train/dev scenes, fixed IoU metric, and no classifier or threshold adaptation",
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
        "postprocessing": {
            "pixel_threshold": PIXEL_THRESHOLD,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "connectivity": 8,
            "box_coordinates": "exact integer tile offset plus connected-component bounds",
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
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_TOLERANCE,
            "parity_batch_sizes": [1, 7, 64],
            "public_or_sealed_reads": 0,
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "BATCH_SIZE", "CANONICAL_OUTPUT", "EPOCHS", "INPUT_CHANNELS", "LEARNING_RATE",
    "MINIMUM_COMPONENT_AREA", "ONNX_PARITY_TOLERANCE", "PIXEL_THRESHOLD",
    "POSITIVE_WEIGHT", "PRECISION_MINIMUM", "RECALL_MINIMUM", "REVISION", "SEED",
    "TASK", "TILE_OVERLAP", "TILE_SIZE", "TRUTH_MATCH_IOU_MINIMUM", "WEIGHT_DECAY",
    "protocol_configuration",
]
