# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen contract for the selected-confidence P2 eight-role public gate."""

from __future__ import annotations


EVIDENCE_POLICY = "ml/policy/evidence-policy.json"
TASK = "ocr-detection-recognition-composition"
REVISION = "graphreader-v10-selected-confidence-eight-role-public-v9-p2"
CANDIDATE_ID = "P2"
COMPOSITION_ID = "graphreader-v10-selected-confidence-acceptance-composition-v9-p2"
SCENE_COUNT = 160
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
PLOT_BOUNDS = (104, 48, 510, 256)
P2_SELECTION_RESULT_SHA256 = (
    "288a04e4c1d4eb8c11aeae6cccb606b43dad4253c697e7703593e354c834e7b1"
)
MODEL_SHA256 = {
    "ocr_detector": "474b8468dbd91416f4e4978dafc46cb2317775d59d821c0470e0cd3e0f6203db",
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
RENDERER_FAMILIES = tuple(
    f"selected-confidence-eight-role-public-v9-p2-{suffix}"
    for suffix in "abcdefgh"
)
DEGRADATION_FAMILIES = (
    "diagonal-fiber-drift-v9-p2-public",
    "low-amplitude-ring-fade-v9-p2-public",
    "offset-row-shelf-v9-p2-public",
    "subpixel-paper-wave-v9-p2-public",
)


def configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-selected-confidence-public-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "composition_id": COMPOSITION_ID,
        "state": "preregistered_before_public_identity_materialization_or_model_execution",
        "predecessor_selection": {
            "result_sha256": P2_SELECTION_RESULT_SHA256,
            "selection_gates_passed": True,
            "case_level_evidence_used": False,
            "scene_count": 128,
            "truth_region_count": 640,
            "false_positives": 0,
            "false_negatives": 0,
            "duplicates": 0,
        },
        "models": {
            name: {"sha256": value, "provider": "CPUExecutionProvider"}
            for name, value in MODEL_SHA256.items()
        },
        "split": {
            "name": "sealed_public",
            "scene_count": SCENE_COUNT,
            "scene_width": SCENE_WIDTH,
            "scene_height": SCENE_HEIGHT,
            "plot_bounds": list(PLOT_BOUNDS),
            "text_truths_per_scene": 8,
            "required_roles": list(REQUIRED_ROLES),
            "renderer_families": list(RENDERER_FAMILIES),
            "degradation_families": list(DEGRADATION_FAMILIES),
            "secret_seed_generated_once_and_not_serialized": True,
            "p1_or_p2_selection_fixture_bytes_reused": False,
            "predecessor_truth_or_scene_ids_reused": False,
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
            "direct_fixture_byte_execution_required": True,
            "direct_tensor_hash_per_model_call_required": True,
            "single_authorized_execution": True,
            "provider": "CPUExecutionProvider",
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "marker_creation_evaluated": False,
        "artifact_mask_production_approval": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
        "blocking_gates_after_pass": [
            "marker_stage_direct_composition_evidence",
            "approved_artifact_mask_provider",
            "approved_production_model_store",
            "packaging_discovery_and_clean_machine_evidence",
            "private_chandler_automatic_validation",
        ],
    }


__all__ = [name for name in globals() if name.isupper()] + ["configuration"]
