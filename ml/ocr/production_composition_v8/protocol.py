# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen identities and gates for production composition V8."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ml.ocr.production_composition_v7.protocol import (
    AMBIGUITY_EXACT_MINIMUM,
    AMBIGUITY_PUBLIC_REPORT_SHA256,
    AMBIGUITY_RECOGNIZER_ONNX_SHA256,
    CHARACTER_ERROR_RATE_MAXIMUM,
    CONSENSUS_RESCUE_SCORE_MINIMUM,
    DETECTOR_ONNX_SHA256,
    DETECTOR_THRESHOLD,
    EXACT_MATCH_MINIMUM,
    NUMERIC_EXACT_MINIMUM,
    NUMERIC_RECOGNIZER_ONNX_SHA256,
    NUMERIC_THRESHOLD,
    OFFICIAL_INFERENCE_YAML_SHA256,
    OFFICIAL_RECOGNIZER_ONNX_SHA256,
    OFFICIAL_RESCUE_SCORE_MINIMUM,
    ROLE_ACCURACY_MINIMUM,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    SPACING_SOURCE_SHA256,
    TRUTH_MATCH_IOU_MINIMUM,
    WORD_EXACT_MINIMUM,
)


TASK = "ocr-production-composition"
REVISION = "graphreader-v10-bounded-zero-consensus-ambiguity-alias-composition-v8"
VALIDATION_REVISION = f"{REVISION}-validation-v1"
PUBLIC_REVISION = f"{REVISION}-public-v1"
ZERO_CONSENSUS_RESCUE_SCORE_MINIMUM = 0.82
AMBIGUITY_INPUT_ALIASES = ("!", "i")


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    scene_count: int
    source_index_offset: int
    renderer_family: str
    degradation_family: str


SPLITS = (
    SplitRegistration(
        "validation", 128, 40_000,
        "fresh-v6-renderer-index-family-composition-v8-validation",
        "fresh-v6-degradation-index-family-composition-v8-validation",
    ),
    SplitRegistration(
        "sealed_public", 160, 50_000,
        "fresh-v6-renderer-index-family-composition-v8-public",
        "fresh-v6-degradation-index-family-composition-v8-public",
    ),
)


def split_registration(split: str) -> SplitRegistration:
    return next(item for item in SPLITS if item.split == split)


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-production-composition-protocol.v8",
        "task": TASK,
        "revision": REVISION,
        "state": "fresh_splits_frozen_before_any_composed_model_execution",
        "predecessor": {
            "revision": "graphreader-v10-two-band-consensus-source-group-invariant-repair-composition-v7",
            "validation_report_sha256": "7d1b2ace57af890fcb95476cc74e1b73f9caf1c22f4c4c9b5a178ab8b80e5dd8",
            "status": "failed_one_missed_axis_zero",
            "fixture_bytes_reused": False,
            "public_archive_opened": False,
            "missed_scene_id": "ocr-production-composition-v7-validation-00079",
            "missed_truth": "0",
            "missed_truth_role": "x_tick",
            "missed_detector_score": 0.8214316368103027,
            "official_prediction": "0",
            "numeric_prediction": "0",
            "numeric_confidence": 0.9999194145202637,
            "ambiguity_exact_match": 0.9,
            "observed_ambiguity_aliases": list(AMBIGUITY_INPUT_ALIASES),
        },
        "models": {
            "detector": {"onnx_sha256": DETECTOR_ONNX_SHA256, "threshold": DETECTOR_THRESHOLD},
            "official_recognizer": {
                "onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
                "spacing_revision": "official-ppocrv5-conservative-spacing-v3-p1",
            },
            "numeric_specialist": {"onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256, "threshold": NUMERIC_THRESHOLD},
            "ambiguity_specialist": {
                "onnx_sha256": AMBIGUITY_RECOGNIZER_ONNX_SHA256,
                "revision": "graph-ambiguity-source-group-v3",
                "public_report_sha256": AMBIGUITY_PUBLIC_REPORT_SHA256,
                "public_accuracy": 1.0,
            },
        },
        "isolated_changes": [
            "allow a lower score band from 0.82 inclusive to 0.85 exclusive only when official and numeric recognizers both return the exact digit zero with matching x_tick or y_tick geometry",
            "route official exclamation-mark and lowercase-i confusions through the exact public-passing source-group classifier only when every nonspace character belongs to O/o/l/I or those two exposed aliases",
        ],
        "splits": [asdict(item) for item in SPLITS],
        "gates": {
            "exact_region_count_every_fixture": True,
            "false_region_count": 0,
            "missed_region_count": 0,
            "duplicate_region_count": 0,
            "prohibited_structure_hits": 0,
            "exact_match_minimum": EXACT_MATCH_MINIMUM,
            "character_error_rate_maximum": CHARACTER_ERROR_RATE_MAXIMUM,
            "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
            "numeric_exact_match_minimum": NUMERIC_EXACT_MINIMUM,
            "word_exact_match_minimum": WORD_EXACT_MINIMUM,
            "ambiguity_exact_match_minimum": AMBIGUITY_EXACT_MINIMUM,
            "spacing_changed_nonspace_truth_count": 0,
            "forbidden_numeric_route_count": 0,
            "forbidden_official_rescue_route_count": 0,
            "forbidden_consensus_rescue_route_count": 0,
            "forbidden_zero_consensus_rescue_route_count": 0,
            "numeric_onnx_direct_execution_minimum": 1,
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "marker_creation_evaluated": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [name for name in globals() if name.isupper()] + ["protocol_configuration", "split_registration"]
