# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration for OCR crop-evidence role-anchor V24."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-crop-evidence-role-anchor-v24"
SEED = 2_608_162_401
CANDIDATE_LIMIT = 3
FEATURE_COUNT = 31
CROP_CHANNELS = 2
CROP_HEIGHT = 32
CROP_WIDTH = 128
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
V23_RESULT_PATH = "ml/ocr/role_anchor_set_v23/P3_RESULT.json"
V23_RESULT_SHA256 = "83d7a3be46e082be3550144cb4bb1b0a287ada29fadbdcca231d2e27d7ad7422"


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
        "train", 256, 9_113_000,
        (
            "crop-evidence-grid-v24-train-a",
            "crop-evidence-offset-v24-train-b",
            "crop-evidence-inset-v24-train-c",
            "crop-evidence-caption-v24-train-d",
        ),
        (
            "local-gamma-v24-train",
            "column-bias-v24-train",
            "level-quantization-v24-train",
            "weak-blur-v24-train",
        ),
        (_REGULAR, _MEDIUM, _SEMIBOLD),
    ),
    "validation": SplitRegistration(
        "validation", 128, 9_419_000,
        (
            "crop-evidence-wide-v24-selection-e",
            "crop-evidence-floating-v24-selection-f",
            "crop-evidence-staggered-v24-selection-g",
            "crop-evidence-compact-v24-selection-h",
        ),
        (
            "row-soften-v24-selection",
            "center-gain-v24-selection",
            "paired-stripe-v24-selection",
            "weak-rounding-v24-selection",
        ),
        (_SEMIBOLD, _REGULAR),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public", 192, 9_827_000,
        (
            "crop-evidence-tall-v24-public-i",
            "crop-evidence-lateral-v24-public-j",
            "crop-evidence-multiphase-v24-public-k",
            "crop-evidence-asymmetric-v24-public-l",
        ),
        (
            "diagonal-tone-v24-public",
            "channel-roundtrip-v24-public",
            "sparse-band-v24-public",
            "contrast-fold-v24-public",
        ),
        (_MEDIUM, _SEMIBOLD),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V24 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-crop-evidence-role-anchor-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_fixture_freeze_or_optimizer_execution",
        "defect_class": (
            "V23 P1 and proposal-head-only P3 retained the same three false and "
            "prohibited regions while preserving all 1024 truths, indicating that the "
            "31 aggregate evidence values cannot distinguish the remaining structures"
        ),
        "predecessor_aggregate_only": {
            "v23_result_path": V23_RESULT_PATH,
            "v23_result_sha256": V23_RESULT_SHA256,
            "v23_exact_scenes": 121,
            "v23_scene_count": 128,
            "v23_true_regions": 1024,
            "v23_truth_regions": 1024,
            "v23_false_regions": 3,
            "v23_missed_regions": 0,
            "v23_prohibited_structure_hits": 3,
            "v23_role_accuracy": 0.99609375,
            "case_level_evidence_used": False,
            "fixture_bytes_truth_scene_or_case_identity_reused": False,
        },
        "isolated_change": (
            "train a new project-owned model from scratch that fuses each proposal's "
            "production tight and context raster crops with the existing 31 evidence values "
            "before the unchanged role-anchor set context and separate proposal and role heads"
        ),
        "architecture": {
            "id": "crop-evidence-role-conditioned-scene-anchor-set-v1",
            "evidence_input": [1, "proposal_count", FEATURE_COUNT],
            "crop_input": [1, "proposal_count", CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "crop_source": "first 128 columns of production V21 tight/context proposal encoding",
            "quadratic_evidence_lift": True,
            "shared_crop_encoder": True,
            "scene_context": (
                "eight learned role queries pool permutation-invariant anchors over fused "
                "proposal representations; each proposal also receives scene mean and maximum"
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
                "tight_and_context_raster_channels": 2,
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
            "detector_recognizer_evidence_and_crop_tensor_hashes_required": True,
        },
        "splits": [asdict(item) for item in _SPLITS.values()],
        "split_policy": {
            "train_validation_public_family_ids_disjoint": True,
            "sealed_public_truth_hidden_until_one_time_gate": True,
            "predecessor_fixture_bytes_reused": False,
            "predecessor_case_identities_reused": False,
            "validation_or_public_pixels_used_for_training": False,
        },
        "data_scope": (
            "fresh project-owned procedural graph scenes only; no Chandler, Generalization, "
            "private or article images, external datasets, pretrained calibrator weights, "
            "or predecessor fixture bytes"
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
