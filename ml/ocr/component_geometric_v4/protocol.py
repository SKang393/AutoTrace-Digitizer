# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for the component-geometric OCR V4 defect class."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TASK = "ocr-recognition"
REVISION = "graph-numeric-component-geometric-v4"
PUBLIC_REVISION = "graph-numeric-component-geometric-v4-public-v1"
PROTOCOL_STATE = "preregistered_before_training"
EXPERIMENT_BUDGET = 3
SEED = 20260810
ALPHABET = "0123456789.-%"
REJECT_CLASS_INDEX = len(ALPHABET)
CLASS_COUNT = len(ALPHABET) + 1
GLYPH_HEIGHT = 24
GLYPH_WIDTH = 20
TRAIN_POSITIVE_COUNT = 2048
TRAIN_NEGATIVE_COUNT = 512
VALIDATION_POSITIVE_COUNT = 256
VALIDATION_NEGATIVE_COUNT = 128
SEALED_POSITIVE_COUNT = 256
SEALED_NEGATIVE_COUNT = 128
EPOCHS = 24
BATCH_SIZE = 256
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
VALIDATION_EXACT_MATCH_MINIMUM = 0.90
SEALED_EXACT_MATCH_MINIMUM = 0.90
SEALED_CER_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
MARKER_EXCLUSION_ACCURACY_MINIMUM = 1.0
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-4
THRESHOLDS = (0.55, 0.65, 0.75, 0.85)
CANONICAL_OUTPUT = Path("ml/ocr/component_geometric_v4/artifacts/P1-run")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    positive_count: int
    negative_count: int
    font_path: str
    font_sha256: str
    renderer_family: str
    degradation_family: str
    seed_offset: int


SPLITS = (
    SplitRegistration(
        "train",
        TRAIN_POSITIVE_COUNT,
        TRAIN_NEGATIVE_COUNT,
        "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf",
        "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823",
        "noto-sans-regular-explicit-cells-v1",
        "train-contrast-blur-and-speckle-v1",
        11_000,
    ),
    SplitRegistration(
        "validation",
        VALIDATION_POSITIVE_COUNT,
        VALIDATION_NEGATIVE_COUNT,
        "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf",
        "635d93d1131d791f2576de90b3bb0f7cdf61929906e8420a61b5f7f8e76420bb",
        "noto-sans-medium-explicit-cells-v1",
        "validation-pixel-scale-and-scanline-v1",
        22_000,
    ),
    SplitRegistration(
        "sealed_public",
        SEALED_POSITIVE_COUNT,
        SEALED_NEGATIVE_COUNT,
        "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf",
        "a4e91fd530ac2b4ef5367240144ff37d7d65d66cf76f2e9a2187b93c676f92d0",
        "noto-sans-semibold-explicit-cells-v1",
        "sealed-offset-dropout-and-fade-v1",
        33_000,
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
        raise ValueError(f"Unknown component-geometric split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-component-geometric-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": PROTOCOL_STATE,
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "architecture": "component-geometric-projection-mlp-v1",
        "distinct_from": [
            "compact-graph-numeric-ctc-v1",
            "spatial-alignment-supervised-sequence-v2",
            "canonical-slot-convolutional-v3",
            "whole-crop-semantic-query-v1",
        ],
        "input": ["glyph_count", 1, GLYPH_HEIGHT, GLYPH_WIDTH],
        "output": ["glyph_count", CLASS_COUNT],
        "alphabet": ALPHABET,
        "reject_class_index": REJECT_CLASS_INDEX,
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
        "font_notice_path": "LICENSES/NotoSans-OFL-1.1.txt",
        "font_notice_sha256": "cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a",
        "production_approval": False,
        "release_eligible": False,
    }
