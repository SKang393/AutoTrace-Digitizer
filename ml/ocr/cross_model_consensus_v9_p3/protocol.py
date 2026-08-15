# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen preregistration for the final P3 cross-model OCR candidate."""

from __future__ import annotations


TASK = "ocr-detection-recognition-composition"
REVISION = "graphreader-v10-v11-cross-model-consensus-v9-p3"
CANDIDATE_ID = "P3"
COMPOSITION_ID = "graphreader-v10-v11-cross-model-consensus-composition-v9-p3"
SELECTION_SCENE_COUNT = 192
PUBLIC_SCENE_COUNT = 256
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
PLOT_BOUNDS = (104, 48, 510, 256)
P2_PUBLIC_RESULT_SHA256 = (
    "11d448e69b5aa972139d518d8e920e3eeeec78ca7d66d2b0a9b0f7fe35c7be78"
)
V11_PUBLIC_RESULT_SHA256 = (
    "3e1da361972b8bd67677db333d55b0b3ad2b5c2f4ef79edcda058167d0a66986"
)
MODEL_SHA256 = {
    "primary_detector": "474b8468dbd91416f4e4978dafc46cb2317775d59d821c0470e0cd3e0f6203db",
    "role_detector": "af13b387140d70946b23ff7349fed82649fde95eb6f6cabe90179b2914a16631",
    "official_recognizer": "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743",
    "numeric_recognizer": "9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84",
    "ambiguity_recognizer": "b8e2773ca3966469081875fc36b3981ef4eb458356d8dfdae2be2722602f0096",
}
REQUIRED_ROLES = (
    "y_tick",
    "x_tick",
    "axis_title",
    "phase_heading",
    "legend_text",
    "participant",
    "annotation",
    "other",
)
SELECTION_RENDERER_FAMILIES = tuple(
    f"cross-model-consensus-selection-v9-p3-{suffix}" for suffix in "abcdefgh"
)
PUBLIC_RENDERER_FAMILIES = tuple(
    f"cross-model-consensus-public-v9-p3-{suffix}" for suffix in "ijklmnop"
)
SELECTION_DEGRADATION_FAMILIES = (
    "fine-column-lilt-v9-p3-selection",
    "shallow-corner-fade-v9-p3-selection",
    "low-rank-paper-slope-v9-p3-selection",
    "alternating-row-haze-v9-p3-selection",
)
PUBLIC_DEGRADATION_FAMILIES = (
    "oblique-paper-ripple-v9-p3-public",
    "off-center-radial-wash-v9-p3-public",
    "paired-column-drift-v9-p3-public",
    "asymmetric-row-shelf-v9-p3-public",
)


def configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-cross-model-consensus-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "composition_id": COMPOSITION_ID,
        "state": "preregistered_before_fixture_identity_or_model_execution",
        "candidate_budget": {
            "defect_class": "recognizer-confirmed selected-confidence acceptance",
            "candidate_number": 3,
            "candidate_limit": 3,
            "final_candidate": True,
            "optimizer_steps": 0,
        },
        "predecessor_aggregate_only": {
            "p2_public_result_sha256": P2_PUBLIC_RESULT_SHA256,
            "p2_case_level_evidence_used": False,
            "p2_public_gate_passed": False,
            "v11_public_result_sha256": V11_PUBLIC_RESULT_SHA256,
            "v11_public_gate_passed": True,
        },
        "decision": {
            "direct_selected_text_minimum_confidence": 0.75,
            "consensus_selected_text_minimum_confidence": 0.55,
            "role_match_minimum_intersection_over_union": 0.95,
            "non_direct_routes_retained": True,
            "v11_role_applies_only_to_geometry_matched_regions": True,
            "ordinary_production_reachable": False,
        },
        "models": {
            name: {"sha256": value, "provider": "CPUExecutionProvider"}
            for name, value in MODEL_SHA256.items()
        },
        "splits": {
            "selection": {
                "scene_count": SELECTION_SCENE_COUNT,
                "renderer_families": list(SELECTION_RENDERER_FAMILIES),
                "degradation_families": list(SELECTION_DEGRADATION_FAMILIES),
                "truth_visible_only_after_identity_freeze": True,
            },
            "sealed_public": {
                "scene_count": PUBLIC_SCENE_COUNT,
                "renderer_families": list(PUBLIC_RENDERER_FAMILIES),
                "degradation_families": list(PUBLIC_DEGRADATION_FAMILIES),
                "single_execution_only_after_selection_pass": True,
            },
            "scene_width": SCENE_WIDTH,
            "scene_height": SCENE_HEIGHT,
            "plot_bounds": list(PLOT_BOUNDS),
            "text_truths_per_scene": 8,
            "required_roles": list(REQUIRED_ROLES),
            "secret_seeds_generated_once_and_not_serialized": True,
            "p1_p2_fixture_bytes_truth_and_scene_ids_reused": False,
            "predecessor_public_bytes_used_for_selection_or_tuning": False,
        },
        "gates": {
            "exact_region_count_every_fixture": True,
            "false_regions": 0,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 0,
            "recognition_exact_match_minimum": 0.90,
            "character_error_rate_maximum": 0.05,
            "role_accuracy_minimum": 0.90,
            "per_role_accuracy_minimum": 0.90,
            "every_required_role_observed": True,
            "numeric_exact_match_minimum": 0.90,
            "word_exact_match_minimum": 0.90,
            "ambiguity_exact_match_minimum": 0.90,
            "all_five_payloads_execute_on_fixture_bytes": True,
            "direct_tensor_hash_per_model_call_required": True,
            "provider": "CPUExecutionProvider",
            "selection_execution_count": 1,
            "public_execution_count_after_selection_pass": 1,
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "fixture_identity_frozen": False,
        "selection_execution_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "artifact_mask_production_approval": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [name for name in globals() if name.isupper()] + ["configuration"]
