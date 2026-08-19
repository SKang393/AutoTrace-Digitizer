# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration for OCR evidence-rescue V25."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-evidence-rescue-v25"
SEED = 2_608_172_501
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
PARENT_CHECKPOINT_PATH = (
    "ml/ocr/crop_evidence_role_anchor_v24/artifacts/P2-run/"
    "graph-text-crop-evidence-role-anchor-v24-p2.pt"
)
PARENT_CHECKPOINT_SHA256 = "8362c155285683130fd3e173db87d40ead0bd6df934e5b239bf455c2f9a0eb73"
PARENT_ONNX_PATH = (
    "ml/ocr/crop_evidence_role_anchor_v24/artifacts/P2-run/"
    "graph-text-crop-evidence-role-anchor-v24-p2.onnx"
)
PARENT_ONNX_SHA256 = "c3276b2109509dddae5b6aea8a7a8ee2ee82960dcbe659834ef1a7cb6c3ea7e6"
PARENT_RESULT_PATH = "ml/ocr/crop_evidence_role_anchor_v24/P2_RESULT.json"
PARENT_RESULT_SHA256 = "c102bf6e2ccc26f401cd23666c81f5d8cdff8c9f2ab530b153dcc50b2f6ce317"

PARENT_ACCEPTANCE_MINIMUM = 0.35
CTC_SELECTED_MEAN_MINIMUM = 0.70
CTC_TOP1_MEAN_MINIMUM = 0.80
CTC_MARGIN_MEAN_MINIMUM = 0.25
CTC_ENTROPY_MEAN_MAXIMUM = 0.45
CTC_BLANK_RATIO_MAXIMUM = 0.75
CTC_LENGTH_FRACTION_MINIMUM = 1.0 / 16.0
CTC_ALNUM_FRACTION_MINIMUM = 0.80
ACCEPTED_LOGIT_MAGNITUDE = 8.0


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
        "train", 256, 10_117_000,
        (
            "evidence-rescue-grid-v25-train-a",
            "evidence-rescue-offset-v25-train-b",
            "evidence-rescue-inset-v25-train-c",
            "evidence-rescue-caption-v25-train-d",
        ),
        (
            "local-contrast-v25-train",
            "column-wave-v25-train",
            "level-rounding-v25-train",
            "edge-soften-v25-train",
        ),
        (_REGULAR, _MEDIUM, _SEMIBOLD),
    ),
    "validation": SplitRegistration(
        "validation", 128, 10_531_000,
        (
            "evidence-rescue-wide-v25-selection-e",
            "evidence-rescue-floating-v25-selection-f",
            "evidence-rescue-staggered-v25-selection-g",
            "evidence-rescue-compact-v25-selection-h",
        ),
        (
            "row-bias-v25-selection",
            "center-fade-v25-selection",
            "paired-column-v25-selection",
            "weak-quantization-v25-selection",
        ),
        (_SEMIBOLD, _REGULAR),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public", 192, 10_977_000,
        (
            "evidence-rescue-tall-v25-public-i",
            "evidence-rescue-lateral-v25-public-j",
            "evidence-rescue-multiphase-v25-public-k",
            "evidence-rescue-asymmetric-v25-public-l",
        ),
        (
            "diagonal-bias-v25-public",
            "channel-fold-v25-public",
            "sparse-row-v25-public",
            "contrast-quantize-v25-public",
        ),
        (_MEDIUM, _SEMIBOLD),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V25 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-evidence-rescue-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_fixture_freeze_or_candidate_execution",
        "defect_class": (
            "V23 P3 retained all 1024 visible-selection truths with three false and "
            "prohibited regions, while V24 P2 removed every false region but missed "
            "two truths; a recognition-evidence rescue is required only for the "
            "parent-accepted proposals rejected by the crop residual"
        ),
        "predecessor_aggregate_only": {
            "v23_true_regions": 1024,
            "v23_false_regions": 3,
            "v23_missed_regions": 0,
            "v23_prohibited_structure_hits": 3,
            "v24_p2_true_regions": 1022,
            "v24_p2_false_regions": 0,
            "v24_p2_missed_regions": 2,
            "v24_p2_prohibited_structure_hits": 0,
            "v24_p2_role_accuracy": 0.9931640625,
            "case_level_evidence_used": False,
            "fixture_bytes_truth_scene_or_case_identity_reused": False,
        },
        "isolated_change": (
            "reuse the exact frozen V24 P2 crop-residual model, preserve its roles, "
            "retain every proposal it accepts, and rescue only parent-accepted "
            "rejections that satisfy a fixed generic CTC confidence and alphanumeric "
            "evidence contract"
        ),
        "architecture": {
            "id": "frozen-crop-residual-ctc-evidence-rescue-v1",
            "evidence_input": [1, "proposal_count", FEATURE_COUNT],
            "crop_input": [1, "proposal_count", CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "optimizer_steps": 0,
            "parent_weights_frozen": True,
            "roles_preserved_exactly": True,
            "candidate_acceptance": (
                "V24 P2 probability >= 0.35 OR V23 parent probability >= 0.35 and "
                "all preregistered generic CTC evidence conditions pass"
            ),
            "ordinary_production_reachable": False,
            "model_license": "Apache-2.0",
        },
        "candidate_budget": {
            "candidate_limit": CANDIDATE_LIMIT,
            "candidate_number": 1,
            "optimizer_steps_maximum": 0,
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
            "parent_checkpoint_path": PARENT_CHECKPOINT_PATH,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_onnx_path": PARENT_ONNX_PATH,
            "parent_onnx_sha256": PARENT_ONNX_SHA256,
            "parent_result_path": PARENT_RESULT_PATH,
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "provider": "CPUExecutionProvider",
        },
        "rescue_contract": {
            "parent_acceptance_minimum": PARENT_ACCEPTANCE_MINIMUM,
            "ctc_selected_mean_minimum": CTC_SELECTED_MEAN_MINIMUM,
            "ctc_top1_mean_minimum": CTC_TOP1_MEAN_MINIMUM,
            "ctc_margin_mean_minimum": CTC_MARGIN_MEAN_MINIMUM,
            "ctc_entropy_mean_maximum": CTC_ENTROPY_MEAN_MAXIMUM,
            "ctc_blank_ratio_maximum": CTC_BLANK_RATIO_MAXIMUM,
            "ctc_length_fraction_minimum": CTC_LENGTH_FRACTION_MINIMUM,
            "ctc_alnum_fraction_minimum": CTC_ALNUM_FRACTION_MINIMUM,
            "accepted_logit_magnitude": ACCEPTED_LOGIT_MAGNITUDE,
            "feature_indices": {
                "ctc_selected_mean": 10,
                "ctc_top1_mean": 11,
                "ctc_margin_mean": 12,
                "ctc_entropy_mean": 13,
                "ctc_blank_ratio": 14,
                "ctc_length_fraction": 15,
                "ctc_digit_fraction": 16,
                "ctc_alpha_fraction": 17,
            },
        },
        "training": {
            "candidate": "P1",
            "seed": SEED,
            "optimizer_steps": 0,
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
            "detector_recognizer_parent_and_candidate_tensor_hashes_required": True,
        },
        "splits": [asdict(item) for item in _SPLITS.values()],
        "split_policy": {
            "train_validation_public_family_ids_disjoint": True,
            "sealed_public_truth_hidden_until_one_time_gate": True,
            "predecessor_fixture_bytes_reused": False,
            "predecessor_case_identities_reused": False,
            "validation_or_public_pixels_used_for_design": False,
        },
        "data_scope": (
            "fresh project-owned procedural graph scenes only; no Chandler, "
            "Generalization, private or article images, external datasets, "
            "predecessor fixture bytes, or predecessor case identities"
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


__all__ = [
    "ACCEPTED_LOGIT_MAGNITUDE", "CANDIDATE_LIMIT", "CROP_CHANNELS",
    "CROP_HEIGHT", "CROP_WIDTH", "DETECTOR_PATH", "DETECTOR_SHA256",
    "FEATURE_COUNT", "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR",
    "PARENT_CHECKPOINT_PATH", "PARENT_CHECKPOINT_SHA256", "PARENT_ONNX_PATH",
    "PARENT_ONNX_SHA256", "PARENT_RESULT_PATH", "PARENT_RESULT_SHA256",
    "RECOGNIZER_PATH", "RECOGNIZER_SHA256", "RECOGNIZER_YAML_PATH",
    "RECOGNIZER_YAML_SHA256", "REVISION", "ROLE_ORDER", "SEED", "TASK",
    "THRESHOLDS", "protocol_configuration", "split_registration",
]
