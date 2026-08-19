# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Preregistration for structural-graph OCR proposal and role repair V14."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-structural-graph-proposal-role-v14"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20262041
SCENE_WIDTH = 704
SCENE_HEIGHT = 352
CROP_WIDTH = 128
CROP_HEIGHT = 32
INPUT_CHANNELS = 2
GEOMETRY_FEATURE_COUNT = 16
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
TRUTH_MATCH_IOU_MINIMUM = 0.5
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5
ROLE_ACCURACY_MINIMUM = 0.90
ROLE_CLASS_ACCURACY_MINIMUM = 0.85
THRESHOLDS = (0.70, 0.78, 0.84, 0.88, 0.92, 0.94, 0.96)
ROLE_ORDER = (
    "YTick", "XTick", "AxisTitle", "PhaseHeading",
    "LegendText", "Participant", "Annotation", "Other",
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
        "train", 480, 1_817_000, "structural-crosshatch-v14-train",
        "anisotropic-smear-speckle-v14-train", (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "validation", 144, 1_963_000, "structural-offset-ruler-v14-validation",
        "phase-resample-shadow-v14-validation", (_MEDIUM, _SEMIBOLD),
        (_MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "sealed_public", 208, 2_117_000, "structural-skeleton-collision-v14-public",
        "threshold-erosion-banding-v14-public", (_SEMIBOLD, _REGULAR),
        (_SEMIBOLD_SHA, _REGULAR_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V14 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-structural-graph-proposal-role-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "design_preregistered_before_stored_split_materialization",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "execution_authorized": False,
        "execution_blocker": (
            "The new renderer, proposal-set contract, runner sources, frozen split bytes, fingerprints, "
            "candidate configuration, and single-use public evaluator must be committed before execution."
        ),
        "defect_class": (
            "aggregate consumed V13 hidden-public evidence showed 223 of 224 exact scenes, "
            "one false prohibited structural region, zero misses, zero duplicates, and perfect role classification"
        ),
        "trigger_evidence": {
            "report_sha256": "fe0a0d63d0c00c35c463e45c11a87f97aa81bb625d0744c8c2307537b6d6d2ff",
            "scene_count": 224,
            "exact_scene_count": 223,
            "true_positives": 1792,
            "truth_regions": 1792,
            "false_regions": 1,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "role_accuracy": 1.0,
            "prohibited_structure_hits": 1,
            "case_level_details_emitted": False,
            "evidence_scope_used_for_v14_design": "aggregate metrics only",
            "v13_public_fixture_bytes_scene_truth_or_case_identity_used": False,
            "consumed_gate_rerun_authorized": False,
        },
        "isolated_change": (
            "replace V13 anisotropic morphology mixture branches with a topology-spectrum residual encoder "
            "that learns from tight and contextual ink, fixed horizontal and vertical edge magnitudes, "
            "multi-cell occupancy, and unchanged production geometry; replace every renderer and degradation "
            "family with fresh structure-collision scenes while retaining [N,2,32,144] to [N,10]"
        ),
        "architecture": "dual-context-topology-spectrum-residual-proposal-role-cnn-v1",
        "distinct_from": [
            "dual-context-anisotropic-morphology-mixture-proposal-role-cnn-v1",
            "dual-context-geometry-gated-proposal-role-cnn-v2",
            "official PP-OCR DB detector",
        ],
        "proposal_algorithm": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "tensor_encoding": "graph-text-component-context-position-v11-encoding-v1",
            "ordering": "top,left,bottom,right",
            "component_grouping_unchanged_from_production": True,
            "input_and_output_dimensions_unchanged_from_v13": True,
            "one_production_proposal_per_truth_required_before_freeze": True,
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
            "epochs": 16,
            "batch_size": 512,
            "learning_rate": 0.00025,
            "weight_decay": 0.0003,
            "negative_cap_per_scene": 72,
            "negative_sampling": "deterministic-round-robin-by-structural-family-v2",
            "output_logit_scale": 0.05,
            "proposal_loss": "balanced-cross-entropy",
            "role_loss": "positive-proposals-only-balanced-cross-entropy",
            "role_loss_weight": 0.75,
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
            "role_accuracy_minimum": 0.90,
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
            "v13_public_fixture_bytes_scene_truth_or_case_identity_reused": False,
            "validation_or_public_pixels_used_for_training": False,
        },
        "data_scope": (
            "fresh procedural scientific graph scenes with structural collisions, crosshatches, rulers, "
            "ticks, grids, multiple marker series, dividers, brackets, arrows, legend frames, and intersections; "
            "synthetic only; no Chandler, Generalization, private or article images, external datasets, "
            "downloaded training data, or predecessor fixture bytes"
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
