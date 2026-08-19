# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen identities and gates for production composition V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ml.ocr.production_composition_v6.protocol import (
    AMBIGUITY_PUBLIC_REPORT_SHA256,
    AMBIGUITY_RECOGNIZER_ONNX_SHA256,
    AMBIGUITY_EXACT_MINIMUM,
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
REVISION = "graphreader-v10-two-band-consensus-source-group-invariant-repair-composition-v7"
VALIDATION_REVISION = f"{REVISION}-validation-v1"
PUBLIC_REVISION = f"{REVISION}-public-v1"


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    scene_count: int
    source_index_offset: int
    renderer_family: str
    degradation_family: str


SPLITS = (
    SplitRegistration(
        "validation", 124, 20_000,
        "fresh-v6-renderer-index-family-composition-v7-validation",
        "fresh-v6-degradation-index-family-composition-v7-validation",
    ),
    SplitRegistration(
        "sealed_public", 156, 30_000,
        "fresh-v6-renderer-index-family-composition-v7-public",
        "fresh-v6-degradation-index-family-composition-v7-public",
    ),
)


def split_registration(split: str) -> SplitRegistration:
    return next(item for item in SPLITS if item.split == split)


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-production-composition-protocol.v7",
        "task": TASK,
        "revision": REVISION,
        "state": "fresh_splits_frozen_before_any_composed_model_execution",
        "predecessor": {
            "revision": "graphreader-v10-two-band-tick-consensus-source-group-segmentation-composition-v6",
            "validation_report_sha256": "b50b8fc1f20da8e589a7436e4d8b41143f85f12a45445fc68ee38483175aa12f",
            "status": "failed_instrumentation_invariant",
            "fixture_bytes_reused": False,
            "public_archive_opened": False,
            "all_scientific_metrics_passed": True,
            "numeric_onnx_calls": 512,
            "accepted_truth_regions": 600,
        },
        "models": {
            "detector": {"onnx_sha256": DETECTOR_ONNX_SHA256, "threshold": DETECTOR_THRESHOLD},
            "official_recognizer": {"onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256, "spacing_revision": "official-ppocrv5-conservative-spacing-v3-p1"},
            "numeric_specialist": {"onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256, "threshold": NUMERIC_THRESHOLD},
            "ambiguity_specialist": {
                "onnx_sha256": AMBIGUITY_RECOGNIZER_ONNX_SHA256,
                "revision": "graph-ambiguity-source-group-v3",
                "public_report_sha256": AMBIGUITY_PUBLIC_REPORT_SHA256,
                "public_accuracy": 1.0,
            },
        },
        "isolated_change": "replace the invalid one-numeric-ONNX-call-per-truth assertion with numeric ONNX direct execution greater than zero because preprocessing legitimately returns before inference for crops without encodable glyph components",
        "composition_source": {
            "pipeline": "ml/ocr/production_composition_v6/pipeline.py",
            "scientific_behavior_changed": False,
        },
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
