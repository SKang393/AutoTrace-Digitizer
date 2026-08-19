# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen preregistration for composite OCR proposal and role repair V11."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-composite-proposal-role-v11"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20261831
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
CROP_WIDTH = 128
CROP_HEIGHT = 32
INPUT_CHANNELS = 2
GEOMETRY_FEATURE_COUNT = 16
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
TRUTH_MATCH_IOU_MINIMUM = 0.5
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5
ROLE_ACCURACY_MINIMUM = 0.90
ROLE_CLASS_ACCURACY_MINIMUM = 0.80
THRESHOLDS = (0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975)
ROLE_ORDER = (
    "YTick",
    "XTick",
    "AxisTitle",
    "PhaseHeading",
    "LegendText",
    "Participant",
    "Annotation",
    "Other",
)


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    scene_count: int
    seed_offset: int
    renderer_family: str
    degradation_family: str
    font_paths: tuple[str, ...]
    font_sha256: tuple[str, ...]


_REGULAR = "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"
_MEDIUM = "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"
_SEMIBOLD = "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"
_REGULAR_SHA = "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823"
_MEDIUM_SHA = "635d93d1131d791f2576de90b3bb0f7cdf61929906e8420a61b5f7f8e76420bb"
_SEMIBOLD_SHA = "a4e91fd530ac2b4ef5367240144ff37d7d65d66cf76f2e9a2187b93c676f92d0"

SPLITS = (
    SplitRegistration(
        "train", 320, 811_000, "dense-eight-role-composite-v11-train",
        "local-tone-median-speckle-v11-train", (_MEDIUM, _SEMIBOLD, _REGULAR),
        (_MEDIUM_SHA, _SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "validation", 96, 853_000, "offset-eight-role-composite-v11-validation",
        "anisotropic-box-row-fade-v11-validation", (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "sealed_public", 128, 907_000, "mirrored-eight-role-composite-v11-public",
        "gamma-quantized-column-fade-v11-public", (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V11 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-composite-proposal-role-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "design_preregistered_before_split_materialization",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "execution_authorized": False,
        "execution_blocker": "Split bytes, fingerprints, candidate configuration, and source checksums are not frozen yet.",
        "defect_class": (
            "aggregate V2 composition evidence showed 119 false text regions, 20 missed text regions, "
            "0 exact scenes, role accuracy 0.64, and 18 text-origin marker creations"
        ),
        "trigger_evidence": {
            "report_path": "artifacts/production-validation/ocr-marker-csharp-v2-report.json",
            "report_sha256": "9875d0e9f82fb2cae2fecde8a1b38653286ee5a7eff92c5c851b85763525d670",
            "scene_count": 40,
            "text_truth_count": 200,
            "true_positives": 180,
            "false_positives": 119,
            "false_negatives": 20,
            "duplicates": 0,
            "role_accuracy": 0.64,
            "text_marker_creations": 18,
            "scene_level_or_fixture_byte_access_for_v11": False,
            "consumed_gate_rerun_authorized": False,
        },
        "isolated_change": (
            "replace the binary-only proposal classifier with a fresh shared visual-geometry encoder and "
            "separate binary proposal and eight-role heads trained on new structure-dense composite scenes"
        ),
        "architecture": "dual-context-shared-encoder-proposal-role-multitask-cnn-v1",
        "distinct_from": [
            "separated-visual-geometry-component-fusion-cnn-fixed-pool-v1",
            "threshold-only V10 repair",
            "official PP-OCR DB detector",
        ],
        "proposal_algorithm": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "tensor_encoding": "graph-text-component-context-position-v11-encoding-v1",
            "ordering": "top,left,bottom,right",
            "component_grouping_unchanged_from_production": True,
            "encoding_change": "append normalized proposal center and origin coordinates to the V7 features",
        },
        "input": ["proposal_count", INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH],
        "output": ["proposal_count", 10],
        "output_contract": {
            "proposal_logits": [0, 2],
            "role_logits": [2, 10],
            "role_order": list(ROLE_ORDER),
            "role_logits_ignored_for_rejected_proposals": True,
        },
        "training": {
            "candidate": "P1",
            "seed": SEED,
            "epochs": 20,
            "batch_size": 256,
            "learning_rate": 0.0003,
            "weight_decay": 0.0002,
            "negative_cap_per_scene": 48,
            "proposal_loss": "balanced-cross-entropy",
            "role_loss": "positive-proposals-only-balanced-cross-entropy",
            "role_loss_weight": 0.6,
            "candidate_budget": 3,
        },
        "selection_thresholds": list(THRESHOLDS),
        "selection_gates": {
            "exact_region_count_every_scene": True,
            "false_regions": 0,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 0,
            "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
            "per_role_accuracy_minimum": ROLE_CLASS_ACCURACY_MINIMUM,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
        },
        "downstream_composition_gates": {
            "recognition_exact_match_minimum": 0.90,
            "character_error_rate_maximum": 0.05,
            "numeric_family_accuracy_minimum": 0.90,
            "word_family_accuracy_minimum": 0.90,
            "ambiguity_family_accuracy_minimum": 0.90,
            "marker_exact_count_every_scene": True,
            "marker_false_positives": 0,
            "marker_false_negatives": 0,
            "marker_duplicates": 0,
            "text_origin_marker_creations": 0,
            "structure_hits": 0,
        },
        "splits": [asdict(item) for item in SPLITS],
        "split_policy": {
            "train_validation_public_family_ids_disjoint": True,
            "sealed_public_truth_hidden_until_one_time_gate": True,
            "predecessor_fixture_bytes_reused": False,
            "v2_scene_bytes_or_truth_reused": False,
            "validation_or_public_pixels_used_for_training": False,
        },
        "data_scope": (
            "fresh procedural eight-role graph scenes with dense axes, ticks, dividers, brackets, arrows, "
            "legends, intersections, connecting lines, and open and filled markers; synthetic only; no Chandler, "
            "Generalization, private or article images, external datasets, downloaded training data, or V2 bytes"
        ),
        "model_license": "Apache-2.0",
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "CROP_HEIGHT", "CROP_WIDTH", "ENCODED_WIDTH", "EXPERIMENT_BUDGET", "GEOMETRY_FEATURE_COUNT",
    "INPUT_CHANNELS", "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR", "PUBLIC_REVISION", "REVISION",
    "ROLE_ACCURACY_MINIMUM", "ROLE_CLASS_ACCURACY_MINIMUM", "ROLE_ORDER", "SCENE_HEIGHT",
    "SCENE_WIDTH", "SEED", "SPLITS", "TASK", "THRESHOLDS", "TRUTH_MATCH_IOU_MINIMUM",
    "protocol_configuration", "split_registration",
]
