# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen preregistration for OCR large-margin proposal calibrator V20."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-margin-calibrator-v20"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20262220
FEATURE_COUNT = 31
DETECTOR_FLOOR = 0.56
TRAINING_NEGATIVE_CAP_PER_SCENE = 8
THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75)
ROBUST_THRESHOLD_RUN_LENGTH = 3
TRUTH_MATCH_IOU_MINIMUM = 0.5
RECOGNITION_EXACT_MINIMUM = 0.90
CHARACTER_ERROR_RATE_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
ROLE_CLASS_ACCURACY_MINIMUM = 0.85
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5
POSITIVE_LOGIT_MARGIN = 1.7346010553881064
NEGATIVE_LOGIT_MARGIN = -1.7346010553881064
MARGIN_LOSS_WEIGHT = 0.25
ROLE_ORDER = (
    "YTick", "XTick", "AxisTitle", "PhaseHeading",
    "LegendText", "Participant", "Annotation", "Other",
)
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
TRIGGER_RESULT_PATH = "ml/ocr/proposal_confirmation_calibrator_v19/P3_RESULT.json"
TRIGGER_RESULT_SHA256 = "fbbd30ce8f078e62eccdacd5fa178d4128619c214fa162146411f46f38eb6232"
LICENSE_PATH = "LICENSES/PaddlePaddle-PP-OCRv5-Models-Apache-2.0.txt"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
NOTICE_PATH = "LICENSES/PaddlePaddle-PP-OCRv5-Models-Notice.txt"
NOTICE_SHA256 = "8d81f5d0c58547cce471c24f82efe768a9d907d06764f67e90cc680c6d777729"


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
        "train", 192, 6_101_000, "margin-calibrator-multiband-v20-train",
        "photometric-margin-pressure-v20-train", (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "validation", 128, 6_307_000, "margin-calibrator-offset-v20-validation",
        "blur-quantization-margin-v20-validation", (_SEMIBOLD, _REGULAR),
        (_SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "sealed_public", 192, 6_521_000, "margin-calibrator-interleave-v20-public",
        "illumination-scanline-margin-v20-public", (_MEDIUM, _REGULAR),
        (_MEDIUM_SHA, _REGULAR_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V20 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-margin-calibrator-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "fresh_stored_splits_pending_freeze_candidate_p1_execution_blocked",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "execution_authorized": False,
        "execution_blocker": (
            "The V20 protocol, fresh stored splits, candidate configuration, source-bound runner, "
            "and one-use public gate must be committed before a separate P1 authorization."
        ),
        "defect_class": (
            "aggregate consumed V19 P3 evidence had a zero-error operating point at threshold "
            "0.65 but no required three-consecutive-threshold zero-error window"
        ),
        "trigger_evidence": {
            "result_path": TRIGGER_RESULT_PATH,
            "result_sha256": TRIGGER_RESULT_SHA256,
            "scene_count": 128,
            "exact_scene_count_at_selected_threshold": 128,
            "truth_regions": 1024,
            "true_positives": 1024,
            "false_regions": 0,
            "missed_regions": 0,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 0,
            "selected_threshold": 0.65,
            "passing_threshold_window": [],
            "lower_threshold_false_region_count": 1,
            "upper_threshold_missed_region_count": 3,
            "recognition_exact": 0.97265625,
            "character_error_rate": 0.004033419763756842,
            "role_accuracy": 1.0,
            "case_level_details_used": False,
            "fixture_bytes_scene_truth_or_case_identity_used": False,
            "consumed_v19_candidate_or_gate_rerun_authorized": False,
        },
        "isolated_change": (
            "train a new quadratic proposal calibrator from scratch on fresh stored V20 scenes "
            "using a preregistered symmetric large-margin loss around the unchanged two-class "
            "logit difference while retaining exact detector, recognizer, features, thresholds, "
            "role logits, and mandatory gates"
        ),
        "architecture": "proposal-evidence-quadratic-margin-mlp-v1",
        "fixed_inputs": {
            "detector_path": DETECTOR_PATH,
            "detector_sha256": DETECTOR_SHA256,
            "detector_floor": DETECTOR_FLOOR,
            "recognizer_path": RECOGNIZER_PATH,
            "recognizer_sha256": RECOGNIZER_SHA256,
            "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
            "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
            "provider": "CPUExecutionProvider",
            "all_parameters_frozen": True,
        },
        "calibrator": {
            "input": ["proposal_count", FEATURE_COUNT],
            "feature_groups": {
                "detector": 2, "role_probabilities": 8, "ctc_statistics": 8,
                "geometry": 9, "morphology": 4,
            },
            "quadratic_lift": True,
            "output": ["proposal_count", 2],
            "hidden_width": 32,
            "activation": "ReLU",
            "model_license": "Apache-2.0",
            "role_logits_modified": False,
        },
        "training": {
            "candidate": "P1",
            "seed": SEED,
            "epochs": 24,
            "batch_size": 256,
            "learning_rate": 0.0008,
            "weight_decay": 0.0001,
            "negative_cap_per_scene": TRAINING_NEGATIVE_CAP_PER_SCENE,
            "negative_sampling": "highest-frozen-detector-probability-nontruth-v1",
            "negative_class_weight": 4.0,
            "positive_logit_margin": POSITIVE_LOGIT_MARGIN,
            "negative_logit_margin": NEGATIVE_LOGIT_MARGIN,
            "margin_loss_weight": MARGIN_LOSS_WEIGHT,
            "validation_or_public_pixels_used": False,
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
            "selected_threshold": "interior-midpoint-of-longest-passing-run",
            "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
            "all_three_model_tensor_stream_hashes_required": True,
        },
        "splits": [asdict(item) for item in SPLITS],
        "split_policy": {
            "train_validation_public_family_ids_disjoint": True,
            "sealed_public_truth_hidden_until_one_time_gate": True,
            "predecessor_fixture_bytes_reused": False,
            "v19_fixture_bytes_scene_truth_or_case_identity_reused": False,
            "validation_or_public_pixels_used_for_training": False,
            "public_case_level_failure_analysis_permitted": False,
        },
        "data_scope": (
            "fresh procedural scientific graph scenes only; no Chandler, Generalization, private "
            "or article images, external datasets, downloaded training data, predecessor fixture "
            "bytes, or V19 validation case identities"
        ),
        "license_path": LICENSE_PATH,
        "license_sha256": LICENSE_SHA256,
        "notice_path": NOTICE_PATH,
        "notice_sha256": NOTICE_SHA256,
        "marker_creation_gate_required_before_approval": True,
        "manifest_created": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [name for name in globals() if name.isupper()] + ["protocol_configuration", "split_registration"]
