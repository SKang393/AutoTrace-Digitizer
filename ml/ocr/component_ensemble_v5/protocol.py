# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the multi-renderer component-ensemble OCR V5 revision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-recognition"
REVISION = "graph-numeric-component-ensemble-v5"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20260820
ALPHABET = "0123456789.-%"
REJECT_CLASS_INDEX = len(ALPHABET)
CLASS_COUNT = len(ALPHABET) + 1
GLYPH_HEIGHT = 24
GLYPH_WIDTH = 20
GEOMETRY_FEATURE_COUNT = 6
ENCODED_GLYPH_WIDTH = GLYPH_WIDTH + GEOMETRY_FEATURE_COUNT
TRAIN_POSITIVE_COUNT = 6144
TRAIN_NEGATIVE_COUNT = 1024
VALIDATION_POSITIVE_COUNT = 512
VALIDATION_NEGATIVE_COUNT = 160
SEALED_POSITIVE_COUNT = 512
SEALED_NEGATIVE_COUNT = 160
EPOCHS = 32
BATCH_SIZE = 512
LEARNING_RATE = 0.0015
WEIGHT_DECAY = 0.0001
VALIDATION_EXACT_MATCH_MINIMUM = 0.90
SEALED_EXACT_MATCH_MINIMUM = 0.90
SEALED_CER_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
MARKER_EXCLUSION_ACCURACY_MINIMUM = 1.0
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-4
STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO = 0.75
THRESHOLDS = (0.35, 0.45, 0.55, 0.65)
CANONICAL_OUTPUT = Path("ml/ocr/component_ensemble_v5/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    positive_count: int
    negative_count: int
    font_paths: tuple[str, ...]
    font_sha256: tuple[str, ...]
    renderer_family: str
    degradation_family: str
    seed_offset: int
    supersample: int


_REGULAR = "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"
_MEDIUM = "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"
_SEMIBOLD = "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"
_REGULAR_SHA = "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823"
_MEDIUM_SHA = "635d93d1131d791f2576de90b3bb0f7cdf61929906e8420a61b5f7f8e76420bb"
_SEMIBOLD_SHA = "a4e91fd530ac2b4ef5367240144ff37d7d65d66cf76f2e9a2187b93c676f92d0"

SPLITS = (
    SplitRegistration(
        "train",
        TRAIN_POSITIVE_COUNT,
        TRAIN_NEGATIVE_COUNT,
        (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
        "noto-mixed-subpixel-cells-v5-train",
        "train-multiscale-resample-fade-and-speckle-v2",
        51_000,
        2,
    ),
    SplitRegistration(
        "validation",
        VALIDATION_POSITIVE_COUNT,
        VALIDATION_NEGATIVE_COUNT,
        (_MEDIUM, _SEMIBOLD, _REGULAR),
        (_MEDIUM_SHA, _SEMIBOLD_SHA, _REGULAR_SHA),
        "noto-mixed-threefold-baseline-jitter-v5-validation",
        "validation-downsample-scanline-and-contrast-v2",
        62_000,
        3,
    ),
    SplitRegistration(
        "sealed_public",
        SEALED_POSITIVE_COUNT,
        SEALED_NEGATIVE_COUNT,
        (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
        "noto-mixed-fourfold-sheared-cells-v5-public",
        "sealed-shear-fade-and-block-dropout-v2",
        73_000,
        4,
    ),
)

EXCLUSION_KINDS = (
    "filled_circle",
    "open_circle",
    "axis_or_tick",
    "divider",
    "bracket",
    "arrow",
    "legend_box",
    "line_intersection",
    "filled_square",
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V5 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-component-ensemble-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_training",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "architecture": "multi-renderer-fixed-feature-ensemble-mlp-v1",
        "distinct_from": [
            "compact-graph-numeric-ctc-v1",
            "spatial-alignment-supervised-sequence-v2",
            "canonical-slot-convolutional-v3",
            "component-geometric-projection-mlp-v4",
        ],
        "input": ["glyph_count", 1, GLYPH_HEIGHT, ENCODED_GLYPH_WIDTH],
        "output": ["glyph_count", CLASS_COUNT],
        "alphabet": ALPHABET,
        "reject_class_index": REJECT_CLASS_INDEX,
        "structural_reject_minimum_height_ratio": STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
        "seed": SEED,
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "thresholds": list(THRESHOLDS),
        "splits": [asdict(item) for item in SPLITS],
        "gates": {
            "validation_exact_match_minimum": VALIDATION_EXACT_MATCH_MINIMUM,
            "sealed_exact_match_minimum": SEALED_EXACT_MATCH_MINIMUM,
            "sealed_cer_maximum": SEALED_CER_MAXIMUM,
            "validation_role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
            "sealed_role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
            "marker_exclusion_accuracy_minimum": MARKER_EXCLUSION_ACCURACY_MINIMUM,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
            "provider": "CPUExecutionProvider",
        },
        "data_scope": (
            "procedurally rendered graph-numeric labels and exclusion shapes only; "
            "no Chandler, private article images, external datasets, or pretrained weights"
        ),
        "exposed_predecessor_cases_used_for_selection": False,
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "font_notice_sha256": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
        "production_approval": False,
        "release_eligible": False,
    }
