# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V10 spaced multi-glyph detector protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection"
REVISION = "graph-text-spaced-component-recall-v10"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20261421
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
TRUTH_MATCH_IOU_MINIMUM = 0.5
THRESHOLDS = (0.80, 0.85, 0.875, 0.90, 0.91, 0.925, 0.95)
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5
BASE_ONNX_SHA256 = "2d35ce2f55cee8317dfe1faf0281a6e87693cca485dbdaf39e4039ced5b97d9c"


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
        "train", 240, 601_000, "spaced-five-role-scenes-v10-train",
        "median-tone-speckle-v10-train", (_MEDIUM, _SEMIBOLD, _REGULAR),
        (_MEDIUM_SHA, _SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "validation", 80, 631_000, "spaced-five-role-scenes-v10-validation",
        "box-resample-column-fade-v10-validation", (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "sealed_public", 112, 673_000, "spaced-five-role-scenes-v10-public",
        "gamma-quantization-row-fade-v10-public", (_SEMIBOLD, _REGULAR, _MEDIUM),
        (_SEMIBOLD_SHA, _REGULAR_SHA, _MEDIUM_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V10 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-spaced-component-recall-protocol.v1",
        "task": TASK, "revision": REVISION, "state": "split_frozen_before_candidate_execution",
        "experiment_budget": EXPERIMENT_BUDGET, "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "defect_class": "generic spaced multi-glyph text-region recall in production composition",
        "trigger_evidence": {
            "composition_revision": "graphreader-v9-official-spacing-p2-numeric-v5-composition-v2",
            "validation_report_sha256": "7a20ae70e9c970f2d10dd80f03a41ab363424cf1a33d98327e835727b587bed1",
            "scene_count": 80, "truth_region_count": 400, "true_positives": 399,
            "false_negatives": 1, "false_positives": 0, "duplicates": 0,
            "missed_role": "annotation", "missed_family": "spaced_multi_glyph",
            "public_archive_opened": False,
        },
        "p1": {
            "optimizer_steps": 0, "weights_changed": False,
            "base_revision": "graph-text-component-recall-v9-p3",
            "base_onnx_sha256": BASE_ONNX_SHA256,
            "isolated_change": "select only the inference threshold on new V10 selection scenes",
        },
        "p2_p3_policy": "remain unregistered unless P1 direct selection evidence proves a distinct training defect",
        "proposal_algorithm": {
            "algorithm": "adaptive-gray-baseline-bounded-line-grouping-v2",
            "source": "ml/ocr/component_context_detector_v7/dataset.py", "ordering": "top,left,bottom,right",
        },
        "input": ["proposal_count", 2, 32, 140], "output": ["proposal_count", 2],
        "selection_thresholds": list(THRESHOLDS),
        "selection_gates": {
            "exact_region_count_every_fixture": True, "false_region_count": 0,
            "missed_region_count": 0, "duplicate_region_count": 0,
            "prohibited_structure_hits": 0, "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
        },
        "splits": [asdict(item) for item in SPLITS],
        "data_scope": (
            "fresh procedural five-role graph scenes emphasizing generic spaced multi-glyph annotations with new "
            "seed, renderer, degradation, and layout families; no Chandler, Generalization, private/article images, "
            "external data, pretrained additions, or V9/V2 fixture-byte reuse"
        ),
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "production_approval": False, "release_eligible": False,
    }


__all__ = [
    "BASE_ONNX_SHA256", "EXPERIMENT_BUDGET", "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR", "PUBLIC_REVISION",
    "REVISION", "SCENE_HEIGHT", "SCENE_WIDTH", "SEED", "SPLITS", "TASK", "THRESHOLDS",
    "TRUTH_MATCH_IOU_MINIMUM", "protocol_configuration", "split_registration",
]
