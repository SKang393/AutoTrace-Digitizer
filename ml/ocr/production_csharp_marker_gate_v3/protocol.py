# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen identities and mandatory gates for the V11 C# composition gate."""

from __future__ import annotations


TASK = "ocr-marker-production-composition"
REVISION = "graphreader-csharp-ocr-marker-composition-v3"
COMPOSITION_ID = "production-v11-ocr-to-normalized-marker-composed-v3"
SCENE_COUNT = 48
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
PLOT_BOUNDS = (104, 48, 510, 256)

MODEL_SHA256 = {
    "ocr_detector": "af13b387140d70946b23ff7349fed82649fde95eb6f6cabe90179b2914a16631",
    "official_recognizer": "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743",
    "numeric_recognizer": "9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84",
    "ambiguity_recognizer": "b8e2773ca3966469081875fc36b3981ef4eb458356d8dfdae2be2722602f0096",
    "marker_center": "017fca04fa3817596ce3088d73f51003dd3658bc56ec3130e25c92252e6bf739",
}

RENDERER_FAMILIES = (
    "orbit-shift-eight-role-marker-composite-v3-public-a",
    "orbit-shift-eight-role-marker-composite-v3-public-b",
    "orbit-shift-eight-role-marker-composite-v3-public-c",
    "orbit-shift-eight-role-marker-composite-v3-public-d",
    "orbit-shift-eight-role-marker-composite-v3-public-e",
    "orbit-shift-eight-role-marker-composite-v3-public-f",
)
DEGRADATION_FAMILIES = (
    "crossfield-gamma-ramp-v3-public",
    "elliptic-paper-shelf-v3-public",
    "staggered-row-quantization-v3-public",
    "counterphase-corner-fade-v3-public",
)


def protocol_configuration() -> dict[str, object]:
    """Return the immutable preregistration without opening fixture bytes."""
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-marker-production-composition-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "composition_id": COMPOSITION_ID,
        "state": "frozen_before_sealed_public_identity_materialization_or_model_execution",
        "models": {
            name: {"sha256": sha256, "provider": "CPUExecutionProvider"}
            for name, sha256 in MODEL_SHA256.items()
        },
        "split": {
            "name": "sealed_public",
            "scene_count": SCENE_COUNT,
            "scene_width": SCENE_WIDTH,
            "scene_height": SCENE_HEIGHT,
            "plot_bounds": list(PLOT_BOUNDS),
            "text_truths_per_scene": 8,
            "required_roles": [
                "y_tick", "x_tick", "axis_title", "phase_heading",
                "legend_text", "participant", "annotation", "other",
            ],
            "renderer_families": list(RENDERER_FAMILIES),
            "degradation_families": list(DEGRADATION_FAMILIES),
            "secret_seed_generated_once_and_not_serialized": True,
            "v1_v2_or_v11_fixture_bytes_reused": False,
            "predecessor_truth_or_scene_ids_reused": False,
        },
        "composition": {
            "ocr_candidate": "graph-text-composite-proposal-role-v11-p2",
            "ocr_text_mask_source": "accepted_CSharp_OcrResult_regions_and_masks_only",
            "artifact_mask_source": "checksum_bound_fixture_authored_structure_plane",
            "artifact_mask_production_approval": False,
            "marker_runtime": "normalized_marker_proposal_runtime_v1_with_frozen_1e-5_activation_boundary_tolerance",
            "marker_matching_tolerance_pixels": 5.0,
            "prohibited_hit_tolerance_pixels": 6.0,
        },
        "gates": {
            "exact_ocr_region_count_every_fixture": True,
            "ocr_false_regions": 0,
            "ocr_missed_regions": 0,
            "ocr_duplicate_regions": 0,
            "recognition_exact_match_minimum": 0.90,
            "character_error_rate_maximum": 0.05,
            "role_accuracy_minimum": 0.90,
            "every_role_observed": True,
            "numeric_exact_match_minimum": 0.90,
            "word_exact_match_minimum": 0.90,
            "ambiguity_exact_match_minimum": 0.90,
            "exact_marker_count_every_fixture": True,
            "marker_false_positives": 0,
            "marker_false_negatives": 0,
            "marker_duplicates": 0,
            "text_marker_creation_count": 0,
            "axis_hits": 0,
            "tick_hits": 0,
            "divider_hits": 0,
            "bracket_hits": 0,
            "arrow_hits": 0,
            "legend_hits": 0,
            "line_intersection_hits": 0,
            "direct_fixture_byte_execution_required": True,
            "direct_tensor_hash_per_model_call_required": True,
            "single_authorized_execution": True,
            "provider": "CPUExecutionProvider",
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "candidate_mode_only": True,
        "manifest_created": False,
        "model_store_promoted": False,
        "production_approval": False,
        "release_eligible": False,
        "blocking_gates_after_pass": [
            "approved_artifact_mask_provider",
            "approved_manifests_and_production_model_store",
            "packaging_discovery_and_clean_machine_execution",
            "private_chandler_automatic_validation",
        ],
    }


__all__ = [name for name in globals() if name.isupper()] + ["protocol_configuration"]
