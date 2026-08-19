# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen P2 identities and gates for selected-confidence acceptance."""

from __future__ import annotations


EVIDENCE_POLICY = "ml/policy/evidence-policy.json"
TASK = "ocr-detection-recognition-composition"
REVISION = "graphreader-v10-selected-confidence-acceptance-v9-p2"
CANDIDATE_ID = "P2"
COMPOSITION_ID = "graphreader-v10-selected-confidence-acceptance-composition-v9-p2"
SCENE_COUNT = 128
SCENE_WIDTH = 640
SCENE_HEIGHT = 320
PLOT_BOUNDS = (104, 48, 510, 256)
SELECTED_TEXT_MINIMUM_CONFIDENCE = 0.75
P1_RESULT_SHA256 = "49c4c84ca4667e2263a0b66a9ae054ec00a7e3ecb542a8d66d871ce36ff643b0"
RENDERER_FAMILIES = tuple(
    f"selected-confidence-shape-confound-v9-p2-selection-{suffix}"
    for suffix in "abcdefgh"
)
DEGRADATION_FAMILIES = (
    "oblique-paper-ripple-v9-p2-selection",
    "piecewise-contrast-shelf-v9-p2-selection",
    "radial-fade-quantization-v9-p2-selection",
    "row-drift-speckle-v9-p2-selection",
)
MODEL_SHA256 = {
    "ocr_detector": "474b8468dbd91416f4e4978dafc46cb2317775d59d821c0470e0cd3e0f6203db",
    "official_recognizer": "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743",
    "numeric_recognizer": "9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84",
    "ambiguity_recognizer": "b8e2773ca3966469081875fc36b3981ef4eb458356d8dfdae2be2722602f0096",
}


def configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-selected-confidence-acceptance-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_number": 2,
        "experiment_budget": 3,
        "composition_id": COMPOSITION_ID,
        "state": "p2_preregistered_before_selection_identity_materialization_or_model_execution",
        "predecessor_aggregate": {
            "candidate_id": "P1",
            "result_sha256": P1_RESULT_SHA256,
            "scene_count": 96,
            "truth_region_count": 480,
            "true_positives": 480,
            "false_positives": 4,
            "false_negatives": 0,
            "duplicates": 0,
            "case_level_evidence_used": False,
        },
        "isolated_change": (
            "retain the immutable P1 pipeline and require selected-text confidence at or "
            "above 0.75 only for its high detector acceptance route"
        ),
        "split": {
            "name": "visible_selection",
            "scene_count": SCENE_COUNT,
            "scene_width": SCENE_WIDTH,
            "scene_height": SCENE_HEIGHT,
            "plot_bounds": list(PLOT_BOUNDS),
            "text_truths_per_scene": 5,
            "required_roles": [
                "y_tick", "x_tick", "phase_heading", "annotation", "legend_text",
            ],
            "full_eight_role_coverage_proven": False,
            "renderer_families": list(RENDERER_FAMILIES),
            "degradation_families": list(DEGRADATION_FAMILIES),
            "secret_seed_generated_once_and_not_serialized": True,
            "p1_fixture_bytes_reused": False,
            "p1_truth_or_scene_ids_reused": False,
        },
        "models": {
            name: {"sha256": value, "provider": "CPUExecutionProvider"}
            for name, value in MODEL_SHA256.items()
        },
        "candidate": {
            "selected_text_minimum_confidence": SELECTED_TEXT_MINIMUM_CONFIDENCE,
            "applies_only_to_p1_recognizer_confirmed_detector_route": True,
            "p1_pipeline_immutable": True,
            "model_bytes_unchanged": True,
        },
        "selection_gates": {
            "exact_region_count_every_fixture": True,
            "false_regions": 0,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "recognition_exact_match_minimum": 0.90,
            "character_error_rate_maximum": 0.05,
            "role_accuracy_minimum": 0.90,
            "numeric_exact_match_minimum": 0.90,
            "word_exact_match_minimum": 0.90,
            "ambiguity_exact_match_minimum": 0.90,
            "direct_fixture_byte_execution_required": True,
            "direct_tensor_hash_per_model_call_required": True,
            "provider": "CPUExecutionProvider",
        },
        "selection_execution_authorized": False,
        "public_gate_authorized": False,
        "public_gate_evaluations": 0,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "manifest_created": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [name for name in globals() if name.isupper()] + ["configuration"]
