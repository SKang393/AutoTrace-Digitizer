# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration for OCR role-anchor set V23."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-role-anchor-set-v23"
SEED = 2_608_162_301
CANDIDATE_LIMIT = 3
FEATURE_COUNT = 31
ROLE_ORDER = (
    "YTick", "XTick", "AxisTitle", "PhaseHeading",
    "LegendText", "Participant", "Annotation", "Other",
)
THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75)
ROBUST_THRESHOLD_RUN_LENGTH = 3
RECOGNITION_EXACT_MINIMUM = 0.90
CHARACTER_ERROR_RATE_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
ROLE_CLASS_ACCURACY_MINIMUM = 0.85
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5
DETECTOR_PATH = (
    "ml/ocr/structural_veto_proposal_role_v17/artifacts/P3-run/"
    "graph-text-structural-veto-proposal-role-v17-p3.onnx"
)
DETECTOR_SHA256 = "ca32487f1df2c3fea1b8c2f51daf7578ed9756e9140d1b0eaf2a16b283591262"
RECOGNIZER_PATH = "ml/ocr/official_bakeoff/runs/conversion/en_PP-OCRv5_mobile_rec.onnx"
RECOGNIZER_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
RECOGNIZER_YAML_PATH = (
    "ml/ocr/official_bakeoff/runs/extracted/en_PP-OCRv5_mobile_rec_infer/inference.yml"
)
RECOGNIZER_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
V22_RESULT_PATH = "ml/ocr/scene_evidence_attention_v22/P3_RESULT.json"
V22_RESULT_SHA256 = "cd8903db6ec7c9e46e2e49938b6552bf09521e53aa277d01f514f460f566afe9"


@dataclass(frozen=True)
class SplitRegistration:
    name: str
    scene_count: int
    seed_offset: int
    renderer_families: tuple[str, ...]
    degradation_families: tuple[str, ...]
    font_paths: tuple[str, ...]


_REGULAR = "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"
_MEDIUM = "src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"
_SEMIBOLD = "src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"

_SPLITS = {
    "train": SplitRegistration(
        "train", 256, 8_113_000,
        (
            "role-anchor-grid-v23-train-a",
            "role-anchor-offset-v23-train-b",
            "role-anchor-inset-v23-train-c",
            "role-anchor-caption-v23-train-d",
        ),
        (
            "anisotropic-paper-v23-train",
            "column-tone-ramp-v23-train",
            "quantized-edge-v23-train",
            "sparse-softening-v23-train",
        ),
        (_REGULAR, _MEDIUM, _SEMIBOLD),
    ),
    "validation": SplitRegistration(
        "validation", 128, 8_419_000,
        (
            "role-anchor-wide-v23-selection-e",
            "role-anchor-floating-v23-selection-f",
            "role-anchor-staggered-v23-selection-g",
            "role-anchor-compact-v23-selection-h",
        ),
        (
            "cross-axis-soften-v23-selection",
            "localized-gain-step-v23-selection",
            "paired-scan-stripe-v23-selection",
            "weak-level-rounding-v23-selection",
        ),
        (_SEMIBOLD, _REGULAR),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public", 192, 8_827_000,
        (
            "role-anchor-tall-v23-public-i",
            "role-anchor-lateral-v23-public-j",
            "role-anchor-multiphase-v23-public-k",
            "role-anchor-asymmetric-v23-public-l",
        ),
        (
            "off-axis-tone-v23-public",
            "cross-channel-roundtrip-v23-public",
            "sparse-speckle-band-v23-public",
            "localized-contrast-fold-v23-public",
        ),
        (_MEDIUM, _SEMIBOLD),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V23 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-role-anchor-set-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_fixture_freeze_or_optimizer_execution",
        "defect_class": (
            "aggregate V22 P3 removed every false and prohibited region but retained one "
            "missed truth and 24 role errors across the fixed visible selection"
        ),
        "predecessor_aggregate_only": {
            "v22_result_path": V22_RESULT_PATH,
            "v22_result_sha256": V22_RESULT_SHA256,
            "v22_exact_scenes": 107,
            "v22_scene_count": 128,
            "v22_true_regions": 1023,
            "v22_truth_regions": 1024,
            "v22_false_regions": 0,
            "v22_missed_regions": 1,
            "v22_role_accuracy": 0.9765625,
            "v22_lowest_role": "Annotation",
            "v22_lowest_role_accuracy": 0.9296875,
            "case_level_evidence_used": False,
            "fixture_bytes_truth_scene_or_case_identity_reused": False,
        },
        "isolated_change": (
            "train a new project-owned role-anchor set model from scratch; learned role queries "
            "summarize the complete proposal set before each proposal is classified, replacing "
            "V22 all-proposal self-attention while retaining the frozen detector, recognizer, "
            "31 evidence features, thresholds, and gates"
        ),
        "architecture": {
            "id": "role-conditioned-scene-anchor-set-v1",
            "input": [1, "proposal_count", FEATURE_COUNT],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "quadratic_feature_lift": True,
            "scene_context": (
                "eight learned role queries pool permutation-invariant anchors over all proposals; "
                "each proposal then attends those anchors plus scene mean and maximum summaries"
            ),
            "proposal_and_role_heads": "separate binary proposal and eight-role heads",
            "dynamic_proposal_count": True,
            "ordinary_production_reachable": False,
            "model_license": "Apache-2.0",
        },
        "candidate_budget": {
            "candidate_limit": CANDIDATE_LIMIT,
            "candidate_number": 1,
            "optimizer_steps_maximum": 1280,
            "selection_execution_limit_per_candidate": 1,
            "public_execution_limit": 1,
        },
        "fixed_inputs": {
            "detector_path": DETECTOR_PATH,
            "detector_sha256": DETECTOR_SHA256,
            "recognizer_path": RECOGNIZER_PATH,
            "recognizer_sha256": RECOGNIZER_SHA256,
            "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
            "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
            "feature_groups": {
                "detector": 2,
                "role_probabilities": 8,
                "ctc_statistics": 8,
                "geometry": 9,
                "morphology": 4,
            },
            "provider": "CPUExecutionProvider",
        },
        "training": {
            "candidate": "P1",
            "seed": SEED,
            "epochs": 5,
            "scene_batch_size": 1,
            "learning_rate": 0.00030,
            "weight_decay": 0.0001,
            "proposal_objective": "class-balanced cross entropy plus scene-extrema margin",
            "role_objective": "class-balanced cross entropy over every positive proposal",
            "synthetic_only": True,
            "validation_or_public_pixels_used": False,
            "training_authorized": False,
        },
        "selection_thresholds": list(THRESHOLDS),
        "selection_gates": {
            "exact_region_and_role_every_scene": True,
            "false_regions": 0,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 0,
            "recognition_exact_minimum": RECOGNITION_EXACT_MINIMUM,
            "character_error_rate_maximum": CHARACTER_ERROR_RATE_MAXIMUM,
            "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
            "per_role_accuracy_minimum": ROLE_CLASS_ACCURACY_MINIMUM,
            "minimum_consecutive_passing_thresholds": ROBUST_THRESHOLD_RUN_LENGTH,
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
            "all_three_model_tensor_stream_hashes_required": True,
        },
        "splits": [asdict(item) for item in _SPLITS.values()],
        "split_policy": {
            "train_validation_public_family_ids_disjoint": True,
            "sealed_public_truth_hidden_until_one_time_gate": True,
            "predecessor_fixture_bytes_reused": False,
            "predecessor_case_identities_reused": False,
            "validation_or_public_pixels_used_for_training": False,
            "public_case_level_failure_analysis_permitted": False,
        },
        "data_scope": (
            "fresh project-owned procedural graph scenes only; no Chandler, Generalization, "
            "private or article images, external datasets, pretrained calibrator weights, or "
            "predecessor fixture bytes"
        ),
        "fixture_identity_frozen": False,
        "training_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "SplitRegistration", "protocol_configuration", "split_registration",
]
