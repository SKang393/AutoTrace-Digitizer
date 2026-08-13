# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen identities and gates for production composition V5."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-production-composition"
REVISION = "graphreader-v10-official-rejected-tick-rescue-source-group-v3-composition-v5"
VALIDATION_REVISION = f"{REVISION}-validation-v1"
PUBLIC_REVISION = f"{REVISION}-public-v1"
SCENE_WIDTH, SCENE_HEIGHT = 640, 320
PLOT_BOUNDS = (104, 48, 510, 256)
DETECTOR_THRESHOLD = 0.95
OFFICIAL_RESCUE_SCORE_MINIMUM = 0.90
NUMERIC_THRESHOLD = 0.65
TRUTH_MATCH_IOU_MINIMUM = 0.5
EXACT_MATCH_MINIMUM = 0.90
CHARACTER_ERROR_RATE_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
NUMERIC_EXACT_MINIMUM = 0.90
WORD_EXACT_MINIMUM = 0.90
AMBIGUITY_EXACT_MINIMUM = 0.90

DETECTOR_ONNX_SHA256 = "474b8468dbd91416f4e4978dafc46cb2317775d59d821c0470e0cd3e0f6203db"
OFFICIAL_RECOGNIZER_ONNX_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
NUMERIC_RECOGNIZER_ONNX_SHA256 = "9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84"
AMBIGUITY_RECOGNIZER_ONNX_SHA256 = "b8e2773ca3966469081875fc36b3981ef4eb458356d8dfdae2be2722602f0096"
AMBIGUITY_PUBLIC_REPORT_SHA256 = "a69b11a164a87f9df959eda863f2263110c2eec3e1d3102f610f7ae32c4e99bd"
OFFICIAL_INFERENCE_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
SPACING_SOURCE_SHA256 = "a8eed54028c6d872942f5c2e54b8b7f8ad458ef6dd021b467639e821066c36e4"


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
        "validation", 112, 1_037_000,
        "six-role-context-composition-v5-validation",
        "resample-contrast-diagonal-fade-composition-v5-validation",
        (_MEDIUM, _SEMIBOLD, _REGULAR), (_MEDIUM_SHA, _SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "sealed_public", 144, 1_109_000,
        "six-role-context-composition-v5-public",
        "affine-quantize-edge-fade-composition-v5-public",
        (_SEMIBOLD, _MEDIUM, _REGULAR), (_SEMIBOLD_SHA, _MEDIUM_SHA, _REGULAR_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    return next(item for item in SPLITS if item.split == split)


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-production-composition-protocol.v5",
        "task": TASK,
        "revision": REVISION,
        "state": "fresh_splits_frozen_before_any_composed_model_execution",
        "predecessor": {
            "revision": "graphreader-v10-numeric-rescue-context-ambiguity-composition-v4",
            "validation_report_sha256": "075eb4cfee77591b8c2f16e3752a85364db261425ca477d99b26d940733a978e",
            "status": "failed",
            "fixture_bytes_reused": False,
            "public_archive_opened": False,
            "false_negatives": 4,
            "ambiguity_exact_match": 0.5,
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
            "replace the mismatched V4 ambiguity crop path with the exact checksum-bound source-group V3 adapter and ONNX",
            "for detector-rejected proposals only in the fixed 0.90 inclusive to 0.95 exclusive score band, accept only valid official-recognizer numeric text at x- or y-tick geometry",
            "never use the numeric specialist alone to accept a rejected proposal",
            "when official raw text is a valid graph number, reject a spacing transform that would make it nonnumeric",
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
