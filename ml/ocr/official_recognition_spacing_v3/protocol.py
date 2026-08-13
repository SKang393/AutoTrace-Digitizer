# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen constants for conservative official-recognizer spacing V3."""

from __future__ import annotations


TASK = "ocr-recognition"
REVISION = "official-ppocrv5-conservative-spacing-v3"
CANDIDATE_ID = "P1"
EXPERIMENT_BUDGET = 3
MODEL_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
INFERENCE_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
TRIGGER_REPORT_SHA256 = "905bb12948ce7bdcdba95f4940e9b1b5f97017da6586c808ff5c43e128049ea9"
SEED = 20261231
COUNTS = {"selection": 256, "sealed_public": 320}
GATES = {
    "exact_match_minimum": 0.90,
    "character_error_rate_maximum": 0.05,
    "role_accuracy_minimum": 0.90,
    "numeric_exact_match_minimum": 0.90,
    "word_exact_match_minimum": 0.90,
    "ambiguity_exact_match_minimum": 0.90,
    "partial_spacing_exact_match_minimum": 0.90,
    "spacing_changed_nonspace_truth_count": 0,
    "nonspace_character_mutation_count": 0,
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
        "schema": "graphreader.ocr-official-recognition-spacing-protocol.v2",
        "task": TASK,
        "revision": REVISION,
        "status": "candidate_1_failed_selection",
        "defect_class": (
            "the V2 source-gap repair inserted spaces into compact words and rewrote lowercase l as capital I "
            "during the first checksum-bound production composition validation"
        ),
        "trigger_evidence": {
            "path": "ml/ocr/production_composition_v3/VALIDATION_REPORT.json",
            "sha256": TRIGGER_REPORT_SHA256,
            "recognition_exact_match": 0.96,
            "ambiguity_exact_match": 0.8181818181818182,
            "spacing_changed_nonspace_truth_count": 3,
            "public_archive_opened": False,
        },
        "prior_revision": "official-ppocrv5-image-spacing-v2",
        "prior_exposed_fixture_bytes_reused": False,
        "experiment_budget": EXPERIMENT_BUDGET,
        "currently_preregistered_candidate": None,
        "consumed_candidates": [CANDIDATE_ID],
        "isolated_change": (
            "raise the source-gap threshold from 0.25 to 0.40 of ink height, permit reconstruction from "
            "partially spaced raw output, and prohibit all recognized nonspace-character rewriting; retain "
            "the exact official ONNX weights, preprocessing, CTC decoder, and width-proportional partitioning"
        ),
        "optimizer_steps": 0,
        "weights_changed": False,
        "model_id": "en_PP-OCRv5_mobile_rec",
        "model_onnx_sha256": MODEL_SHA256,
        "inference_yaml_sha256": INFERENCE_YAML_SHA256,
        "runner_source_bundle_sha256": runner_source_bundle_sha256,
        "p1_failure_evidence": {
            "result_path": "ml/ocr/official_recognition_spacing_v3/P1_RESULT.json",
            "selection_report_sha256": "fe2030b63a5a50e76347fe7da827e2ca6084e7dacdbbccf89fb945fd9883f1e5",
            "exact_match": 0.875,
            "ambiguity_exact_match": 0.0,
            "compact_word_exact_match": 1.0,
            "spaced_word_exact_match": 1.0,
            "partial_spacing_exact_match": 1.0,
            "spacing_changed_nonspace_truth_count": 0,
            "nonspace_character_mutation_count": 0,
            "public_archive_opened": False,
        },
        "spacing_algorithm": {
            "id": "conservative-source-projection-large-gap-v3",
            "minimum_gap_pixels": 5,
            "minimum_gap_to_ink_height_ratio": 0.40,
            "minimum_source_groups": 2,
            "foreground_contrast_fraction": 0.30,
            "minimum_foreground_contrast": 10,
            "partitioning": "width-proportional dynamic programming with nonempty text groups",
            "existing_spaces_removed_before_partition_and_reconstructed_from_source": True,
            "recognized_nonspace_character_rewriting_forbidden": True,
            "truth_or_role_input_forbidden": True,
        },
        "splits": {
            "selection": {
                "case_count": COUNTS["selection"], "seed": SEED,
                "renderer_family": "noto-controlled-gap-spacing-v3-selection",
                "degradation_families": [
                    "fractional-shear-v3", "anisotropic-resample-v3", "paper-gradient-v3", "soft-threshold-v3"
                ],
            },
            "sealed_public": {
                "case_count": COUNTS["sealed_public"], "seed": SEED + 300_000,
                "renderer_family": "noto-controlled-gap-spacing-v3-public",
                "degradation_families": [
                    "fractional-shift-v3", "lanczos-cycle-v3", "gamma-band-v3", "column-fade-v3"
                ],
            },
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
    "MODEL_SHA256", "PUBLIC_GATE_CONFIG", "REVISION", "SEED", "TASK",
    "TRIGGER_REPORT_SHA256", "protocol_configuration",
]
