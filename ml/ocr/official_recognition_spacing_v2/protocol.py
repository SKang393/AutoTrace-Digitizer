# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for the bounded official-recognizer spacing repair."""

from __future__ import annotations


TASK = "ocr-recognition"
REVISION = "official-ppocrv5-image-spacing-v2"
CANDIDATE_ID = "P1"
EXPERIMENT_BUDGET = 3
MODEL_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
INFERENCE_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
SEED = 20261117
COUNTS = {"selection": 224, "sealed_public": 288}
GATES = {
    "exact_match_minimum": 0.90,
    "character_error_rate_maximum": 0.05,
    "role_accuracy_minimum": 0.90,
    "numeric_exact_match_minimum": 0.90,
    "word_exact_match_minimum": 0.90,
    "ambiguity_exact_match_minimum": 0.90,
    "spacing_changed_nonspace_truth_count": 0,
    "conversion_parity_maximum_absolute_error": 0.0001,
    "provider": "CPUExecutionProvider",
}
PUBLIC_GATE_CONFIG = {
    **GATES,
    "evaluation_limit": 1,
    "production_approval": False,
    "release_eligible": False,
}


def protocol_configuration(*, runner_source_bundle_sha256: str) -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-official-recognition-spacing-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "status": "p1_preregistered_before_inference",
        "defect_class": (
            "the exact official recognizer preserved O/o/l/I glyph identities but removed visible inter-glyph "
            "whitespace in both exposed ambiguity failures"
        ),
        "experiment_budget": EXPERIMENT_BUDGET,
        "currently_preregistered_candidate": CANDIDATE_ID,
        "consumed_candidates": [],
        "isolated_change": (
            "retain the exact official ONNX weights and CTC decoder, then restore whitespace only from generic "
            "large-gap evidence in the immutable source crop; no truth string, role, or label whitelist may "
            "participate in postprocessing"
        ),
        "optimizer_steps": 0,
        "weights_changed": False,
        "model_id": "en_PP-OCRv5_mobile_rec",
        "model_onnx_sha256": MODEL_SHA256,
        "inference_yaml_sha256": INFERENCE_YAML_SHA256,
        "runner_source_bundle_sha256": runner_source_bundle_sha256,
        "spacing_algorithm": {
            "id": "source-projection-large-gap-v1",
            "minimum_gap_pixels": 4,
            "minimum_gap_to_ink_height_ratio": 0.25,
            "minimum_source_groups": 3,
            "foreground_contrast_fraction": 0.30,
            "minimum_foreground_contrast": 10,
            "requires_raw_prediction_without_spaces": True,
            "requires_at_least_three_source_groups": True,
            "requires_source_group_count_not_greater_than_unicode_scalar_count": True,
            "partitioning": "width-proportional dynamic programming with nonempty text groups",
            "truth_or_role_input_forbidden": True,
        },
        "splits": {
            "selection": {
                "case_count": COUNTS["selection"],
                "seed": SEED,
                "renderer_family": "noto-variable-pad-spacing-v2-selection",
                "degradation_families": [
                    "mild-shear-v2", "box-resample-v2", "uneven-paper-v2", "contrast-dither-v2"
                ],
            },
            "sealed_public": {
                "case_count": COUNTS["sealed_public"],
                "seed": SEED + 200_000,
                "renderer_family": "noto-variable-pad-spacing-v2-public",
                "degradation_families": [
                    "subpixel-offset-v2", "area-resample-v2", "gamma-raster-v2", "row-fade-v2"
                ],
            },
        },
        "prior_exposed_selection_used_only_for_defect_diagnosis": {
            "revision": "official-ppocrv5-recognition-only-v1",
            "result_sha256": "69966537744b51287ff1163c8bebf1b90e91a3b4a8548f048198ff1c2d9d9f3f",
            "failed_case_ids": ["selection-recognition-0067", "selection-recognition-0135"],
            "raw_prediction": "OolI",
            "truth_text": "O o l I",
            "public_archive_opened": False,
        },
        "gates": GATES,
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


__all__ = [
    "CANDIDATE_ID", "COUNTS", "EXPERIMENT_BUDGET", "GATES", "INFERENCE_YAML_SHA256",
    "MODEL_SHA256", "PUBLIC_GATE_CONFIG", "REVISION", "SEED", "TASK", "protocol_configuration",
]
