# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for deterministic real-range proposal expansion V34."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-detection"
REVISION = "graph-text-real-range-proposal-v34"
CANDIDATE_ID = "P1"
V32_REVISION = "graph-text-real-range-classifier-finetune-v32"
V33_REVISION = "graph-text-real-range-classifier-v33"
TRAIN_DATA_REVISION = "graph-text-real-range-classifier-finetune-v32"
DATASET_MODULE = Path("ml/ocr/real_range_classifier_finetune_v32/dataset.py")
PROPOSAL_SOURCE = Path("ml/ocr/component_context_detector_v7/dataset.py")
SEED = 20260934
TRUTH_MATCH_IOU_MINIMUM = 0.50
PRECISION_MINIMUM = 0.95
RECALL_MINIMUM = 0.95
EXPANSION_MARGIN_PIXELS = 1
CANONICAL_OUTPUT = Path("ml/ocr/real_range_proposal_v34/artifacts/P1-dev")
RAW_PROPOSAL_DIAGNOSTIC_PATH = "docs/GOAL-22-PHASE-4R-OCR-RAW-PROPOSAL-DIAGNOSTIC.json"
RAW_PROPOSAL_DIAGNOSTIC_SHA256 = "8b26d337b8ff29a1d9d937c09c38073635ac2cf9b24dabfc6e7a41e9ea22de8a"
V33_RESULT_PATH = "ml/ocr/real_range_classifier_v33/P1_RESULT.json"
V33_RESULT_SHA256 = "96526456b1bc191af2f6490dd929ba35b756bd76506ac1c23d1c79393470e51b"


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    data_revision: str
    data_module: str
    family_disjoint: bool
    public_or_sealed_reads: int


SPLITS = (
    SplitRegistration("train", TRAIN_DATA_REVISION, DATASET_MODULE.as_posix(), True, 0),
    SplitRegistration("dev", TRAIN_DATA_REVISION, DATASET_MODULE.as_posix(), True, 0),
)


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-real-range-proposal-v34-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "state": "preregistered_before_execution",
        "evidence_policy": "ml/policy/evidence-policy.json",
        "defect_class": "raw proposal-generation recall ceiling on corrected real-range graph text",
        "trigger_evidence": {
            "v32_revision": V32_REVISION,
            "v32_raw_proposal_maximum_match_true_positives": 61,
            "v32_raw_proposal_truth_regions": 86,
            "v32_raw_proposal_recall": 0.7093023256,
            "raw_proposal_diagnostic_path": RAW_PROPOSAL_DIAGNOSTIC_PATH,
            "raw_proposal_diagnostic_sha256": RAW_PROPOSAL_DIAGNOSTIC_SHA256,
            "v33_revision": V33_REVISION,
            "v33_project_owned_classifier_result": "failed fixed dev bars",
            "v33_result_path": V33_RESULT_PATH,
            "v33_result_sha256": V33_RESULT_SHA256,
            "classifier_only_cannot_recover_missing_raw_proposals": True,
        },
        "model_sourcing": {
            "approved_pretrained_attempted": True,
            "compatible_finetune_attempted": True,
            "project_owned_classifier_attempted": True,
            "learned_detector_needed": False,
            "reason": "V32 and V33 classifier routes failed their dev bars, while V32 raw proposals maximum-match only 61 of 86 truths; V34 repairs the proposal ceiling before any learned detector is considered.",
        },
        "isolated_change": "union the committed V32 proposal generator with deterministic percentile-contrast proposals, one-pixel component expansion, and existing line grouping; retain the V32 five-axis family-disjoint corrected synthetic data",
        "proposal_contract": {
            "base_source": PROPOSAL_SOURCE.as_posix(),
            "base_algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "repair_algorithm": "percentile-contrast-union-expand-group-v1",
            "expansion_margin_pixels": EXPANSION_MARGIN_PIXELS,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
        },
        "runtime_compatibility": "not_applicable_no_learned_model_or_onnx_payload",
        "splits": [asdict(item) for item in SPLITS],
        "selection_gates": {
            "raw_proposal_precision_minimum": PRECISION_MINIMUM,
            "raw_proposal_recall_minimum": RECALL_MINIMUM,
            "maximum_cardinality_matching": True,
            "public_or_sealed_reads": 0,
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "CANDIDATE_ID", "CANONICAL_OUTPUT", "DATASET_MODULE", "EXPANSION_MARGIN_PIXELS",
    "PRECISION_MINIMUM", "PROPOSAL_SOURCE", "RECALL_MINIMUM", "REVISION", "SEED",
    "TASK", "TRUTH_MATCH_IOU_MINIMUM", "protocol_configuration",
]
