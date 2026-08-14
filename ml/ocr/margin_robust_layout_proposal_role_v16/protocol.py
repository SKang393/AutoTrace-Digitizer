# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Preregistration for margin-robust OCR proposal and role repair V16."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-margin-robust-layout-proposal-role-v16"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20262179
SCENE_WIDTH = 704
SCENE_HEIGHT = 352
CROP_WIDTH = 128
CROP_HEIGHT = 32
INPUT_CHANNELS = 2
BASE_GEOMETRY_FEATURE_COUNT = 16
PLOT_GEOMETRY_FEATURE_COUNT = 8
GEOMETRY_FEATURE_COUNT = BASE_GEOMETRY_FEATURE_COUNT + PLOT_GEOMETRY_FEATURE_COUNT
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
TRUTH_MATCH_IOU_MINIMUM = 0.5
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5
ROLE_ACCURACY_MINIMUM = 0.90
ROLE_CLASS_ACCURACY_MINIMUM = 0.85
PROPOSAL_MARGIN = 1.2
PROPOSAL_MARGIN_LOSS_WEIGHT = 0.5
ROBUST_THRESHOLD_RUN_LENGTH = 3
THRESHOLDS = (0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80)
ROLE_ORDER = (
    "YTick", "XTick", "AxisTitle", "PhaseHeading",
    "LegendText", "Participant", "Annotation", "Other",
)
PLOT_GEOMETRY_ORDER = (
    "center_x_over_plot", "center_y_over_plot", "left_over_plot", "right_over_plot",
    "top_over_plot", "bottom_over_plot", "width_over_plot", "height_over_plot",
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
        "train", 640, 2_791_000, "margin-lane-pressure-v16-train",
        "fractional-stroke-band-v16-train", (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "validation", 192, 3_011_000, "boundary-margin-grid-v16-validation",
        "mixed-width-threshold-v16-validation", (_SEMIBOLD, _REGULAR),
        (_SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "sealed_public", 256, 3_239_000, "role-margin-interleave-v16-public",
        "edge-contrast-envelope-v16-public", (_MEDIUM, _REGULAR),
        (_MEDIUM_SHA, _REGULAR_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V16 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-margin-robust-layout-proposal-role-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "design_preregistered_before_stored_split_materialization",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": None,
        "execution_authorized": False,
        "execution_blocker": (
            "Fresh split bytes, fingerprints, candidate configuration, runner sources, and the "
            "single-use public evaluator must be committed before a candidate can execute."
        ),
        "defect_class": (
            "aggregate consumed V15 public evidence showed 217 of 224 exact scenes, one false "
            "prohibited region, one missed truth, zero duplicates, role accuracy "
            "0.9972082635399219, and minimum XTick accuracy 0.9820627802690582"
        ),
        "trigger_evidence": {
            "report_sha256": "8bd7170db115f6fccbfc9b998bd5f6fce0d8ae001469b692fa07e8392068553d",
            "scene_count": 224,
            "exact_scene_count": 217,
            "truth_regions": 1792,
            "true_positives": 1791,
            "false_regions": 1,
            "missed_regions": 1,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 1,
            "role_accuracy": 0.9972082635399219,
            "minimum_role_accuracy": 0.9820627802690582,
            "case_level_details_emitted": False,
            "evidence_scope_used_for_v16_design": "aggregate metrics only",
            "v15_public_fixture_bytes_scene_truth_or_case_identity_used": False,
            "consumed_v15_candidate_or_gate_rerun_authorized": False,
        },
        "isolated_change": (
            "add a separate eight-value plot-geometry proposal residual to the V15 layout role "
            "architecture, train with a signed proposal-margin objective on fresh V16 procedural "
            "families, and require a zero-error interior run of at least three adjacent thresholds"
        ),
        "architecture": "dual-context-topology-layout-margin-proposal-role-cnn-v1",
        "distinct_from": [
            "dual-context-topology-layout-conditioned-proposal-role-cnn-v1",
            "official PP-OCR DB detector",
        ],
        "renderer_implementation": {
            "procedural_implementation_source": "layout-conditioned-proposal-role-v15-renderer-code",
            "fresh_seed_offsets": True,
            "fresh_renderer_family_ids": True,
            "fresh_degradation_family_ids": True,
            "predecessor_fixture_bytes_reused": False,
        },
        "proposal_algorithm": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "base_tensor_encoding": "graph-text-component-context-position-v11-encoding-v1",
            "plot_geometry_source": "verified-axis-stage-plot-bounds-v1",
            "plot_geometry_order": list(PLOT_GEOMETRY_ORDER),
            "ordering": "top,left,bottom,right",
            "component_grouping_unchanged_from_production": True,
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
            "epochs": 18,
            "batch_size": 512,
            "learning_rate": 0.00020,
            "weight_decay": 0.0003,
            "negative_cap_per_scene": 72,
            "negative_sampling": "deterministic-round-robin-by-structural-family-v3",
            "proposal_loss": "balanced-cross-entropy",
            "proposal_margin_loss": "signed-hinge-margin",
            "proposal_margin": PROPOSAL_MARGIN,
            "proposal_margin_loss_weight": PROPOSAL_MARGIN_LOSS_WEIGHT,
            "role_loss": "positive-proposals-only-balanced-cross-entropy",
            "role_loss_weight": 1.0,
            "candidate_budget": 3,
        },
        "selection_thresholds": list(THRESHOLDS),
        "selection_gates": {
            "exact_region_and_role_every_scene": True,
            "false_regions": 0,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 0,
            "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
            "per_role_accuracy_minimum": ROLE_CLASS_ACCURACY_MINIMUM,
            "minimum_consecutive_passing_thresholds": ROBUST_THRESHOLD_RUN_LENGTH,
            "selected_threshold": "interior-midpoint-of-longest-passing-run",
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
        },
        "splits": [asdict(item) for item in SPLITS],
        "split_policy": {
            "train_validation_public_family_ids_disjoint": True,
            "sealed_public_truth_hidden_until_one_time_gate": True,
            "predecessor_fixture_bytes_reused": False,
            "v15_public_fixture_bytes_scene_truth_or_case_identity_reused": False,
            "validation_or_public_pixels_used_for_training": False,
            "public_case_level_failure_analysis_permitted": False,
        },
        "data_scope": (
            "fresh procedural scientific graph scenes only; no Chandler, Generalization, private "
            "or article images, external datasets, downloaded training data, predecessor fixture "
            "bytes, or V15 public case identities"
        ),
        "model_license": "Apache-2.0",
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "BASE_GEOMETRY_FEATURE_COUNT", "CROP_HEIGHT", "CROP_WIDTH", "ENCODED_WIDTH",
    "EXPERIMENT_BUDGET", "GEOMETRY_FEATURE_COUNT", "INPUT_CHANNELS",
    "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR", "PLOT_GEOMETRY_FEATURE_COUNT",
    "PLOT_GEOMETRY_ORDER", "PROPOSAL_MARGIN", "PROPOSAL_MARGIN_LOSS_WEIGHT",
    "PUBLIC_REVISION", "REVISION", "ROBUST_THRESHOLD_RUN_LENGTH", "ROLE_ACCURACY_MINIMUM",
    "ROLE_CLASS_ACCURACY_MINIMUM", "ROLE_ORDER", "SCENE_HEIGHT", "SCENE_WIDTH", "SEED",
    "SPLITS", "TASK", "THRESHOLDS", "TRUTH_MATCH_IOU_MINIMUM", "protocol_configuration",
    "split_registration",
]
