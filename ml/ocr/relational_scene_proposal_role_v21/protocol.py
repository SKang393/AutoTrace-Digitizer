# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Preregistration for the relational-scene OCR defect class V21."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-role-composition"
REVISION = "graph-text-relational-scene-proposal-role-v21"
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
INPUT_CHANNELS = 2
CROP_HEIGHT = 32
CROP_WIDTH = 128
GEOMETRY_FEATURE_COUNT = 24
ENCODED_WIDTH = CROP_WIDTH + GEOMETRY_FEATURE_COUNT
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
TRUTH_MATCH_IOU_MINIMUM = 0.50
SEED = 2_608_152_101
CANDIDATE_LIMIT = 3
THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


@dataclass(frozen=True)
class SplitRegistration:
    name: str
    scene_count: int
    seed_offset: int
    renderer_families: tuple[str, ...]
    degradation_families: tuple[str, ...]
    font_paths: tuple[str, ...]


_SPLITS = {
    "train": SplitRegistration(
        "train",
        384,
        2_101_000,
        (
            "relational-grid-ledger-v21-train-a",
            "relational-staggered-panels-v21-train-b",
            "relational-offset-margins-v21-train-c",
            "relational-compact-legend-v21-train-d",
        ),
        (
            "subpixel-column-shear-v21-train",
            "low-frequency-paper-bowl-v21-train",
            "alternating-ink-spread-v21-train",
            "weak-row-dropout-v21-train",
        ),
        (
            "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf",
            "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf",
        ),
    ),
    "validation": SplitRegistration(
        "validation",
        128,
        2_403_000,
        (
            "relational-wide-axis-v21-selection-e",
            "relational-inset-plot-v21-selection-f",
            "relational-split-caption-v21-selection-g",
            "relational-floating-note-v21-selection-h",
        ),
        (
            "anisotropic-resample-v21-selection",
            "corner-illumination-v21-selection",
            "paired-scanline-v21-selection",
            "mild-block-quantization-v21-selection",
        ),
        (
            "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf",
            "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf",
        ),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public",
        192,
        2_807_000,
        (
            "relational-tall-plot-v21-public-i",
            "relational-lateral-legend-v21-public-j",
            "relational-multi-divider-v21-public-k",
            "relational-asymmetric-caption-v21-public-l",
        ),
        (
            "off-axis-paper-wave-v21-public",
            "localized-contrast-shelf-v21-public",
            "cross-channel-resample-v21-public",
            "sparse-speckle-band-v21-public",
        ),
        (
            "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf",
            "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf",
        ),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V21 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-relational-scene-proposal-role-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "defect_class": "scene-relational proposal acceptance and eight-role generalization",
        "candidate_budget": {
            "candidate_limit": CANDIDATE_LIMIT,
            "candidate_number": 1,
            "optimizer_steps_maximum": 1536,
            "public_execution_limit": 1,
        },
        "predecessor_aggregate_only": {
            "p3_selection_result_sha256": "f7935a64de07f1187a4ca854c01a90b4fcd004c533f57c6f464d98c5e59105e2",
            "case_level_evidence_used": False,
            "fixture_bytes_truth_or_scene_ids_reused": False,
            "aggregate_failure": {
                "exact_scenes": 102,
                "scene_count": 192,
                "false_regions": 5,
                "missed_regions": 91,
                "role_accuracy": 0.494140625,
            },
        },
        "architecture": {
            "id": "dynamic-proposal-set-relational-message-passing-v1",
            "input": [1, "proposal_count", INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "per_proposal_visual_encoder": "dual-crop convolutional encoder",
            "geometry_features": GEOMETRY_FEATURE_COUNT,
            "scene_context": "two learned all-proposal message-passing blocks",
            "proposal_and_role_heads": "separate binary proposal and eight-role heads",
            "dynamic_proposal_count": True,
            "single_tensor_input": True,
            "ordinary_production_reachable": False,
        },
        "isolated_change": (
            "replace independent per-proposal classification with scene-level relational message passing; "
            "retain production component grouping, crop pixels, and original-coordinate geometry"
        ),
        "training": {
            "synthetic_only": True,
            "optimizer": "AdamW",
            "learning_rate": 0.00035,
            "weight_decay": 0.0001,
            "epochs": 4,
            "scene_batch_size": 1,
            "proposal_loss": "balanced-cross-entropy",
            "role_loss": "positive-only-balanced-cross-entropy",
            "role_loss_weight": 0.75,
            "seed": SEED,
            "deterministic_algorithms": True,
            "cpu_threads": 1,
        },
        "splits": {name: asdict(registration) for name, registration in _SPLITS.items()},
        "gates": {
            "thresholds": list(THRESHOLDS),
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
            "exact_region_count_every_fixture": True,
            "false_regions": 0,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 0,
            "role_accuracy_minimum": 0.90,
            "per_role_accuracy_minimum": 0.90,
            "every_required_role_observed": True,
            "onnx_parity_maximum": 0.00001,
            "provider": "CPUExecutionProvider",
            "public_execution_count": 1,
        },
        "required_roles": list(ROLE_ORDER),
        "source_bytes_immutable": True,
        "original_coordinate_output": True,
        "chandler_included": False,
        "generalization_label_included": False,
        "private_or_article_images": False,
        "external_training_data": False,
        "fixture_identity_frozen": False,
        "training_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "artifact_mask_production_approval": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
        "state": "preregistered_before_fixture_identity_or_optimizer_execution",
    }


__all__ = [
    "CANDIDATE_LIMIT",
    "CROP_HEIGHT",
    "CROP_WIDTH",
    "ENCODED_WIDTH",
    "GEOMETRY_FEATURE_COUNT",
    "INPUT_CHANNELS",
    "REVISION",
    "ROLE_ORDER",
    "SCENE_HEIGHT",
    "SCENE_WIDTH",
    "SEED",
    "TASK",
    "THRESHOLDS",
    "TRUTH_MATCH_IOU_MINIMUM",
    "protocol_configuration",
    "split_registration",
]
