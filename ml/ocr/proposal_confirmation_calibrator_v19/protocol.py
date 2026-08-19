# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen preregistration for OCR proposal confirmation calibrator V19."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-proposal-confirmation-calibrator-v19"
PUBLIC_REVISION = f"{REVISION}-public-v1"
EXPERIMENT_BUDGET = 3
SEED = 20262219
FEATURE_COUNT = 31
DETECTOR_FLOOR = 0.56
TRAINING_NEGATIVE_CAP_PER_SCENE = 4
THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75)
ROBUST_THRESHOLD_RUN_LENGTH = 3
TRUTH_MATCH_IOU_MINIMUM = 0.5
RECOGNITION_EXACT_MINIMUM = 0.90
CHARACTER_ERROR_RATE_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
ROLE_CLASS_ACCURACY_MINIMUM = 0.85
ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR = 1e-5
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
TRIGGER_RESULT_PATH = "ml/ocr/recognition_confirmed_proposal_role_v18/P1_RESULT.json"
TRIGGER_RESULT_SHA256 = "8b77be5cf32db4d7519035476dd155f1726fd545bfca0014d82d8a2922e38dba"
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
        "train", 192, 4_903_000, "proposal-calibrator-lattice-v19-train",
        "ctc-structure-cross-v19-train", (_REGULAR, _MEDIUM, _SEMIBOLD),
        (_REGULAR_SHA, _MEDIUM_SHA, _SEMIBOLD_SHA),
    ),
    SplitRegistration(
        "validation", 128, 5_117_000, "proposal-calibrator-collision-v19-validation",
        "ctc-geometry-envelope-v19-validation", (_SEMIBOLD, _REGULAR),
        (_SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "sealed_public", 192, 5_333_000, "proposal-calibrator-interleave-v19-public",
        "ctc-edge-pressure-v19-public", (_MEDIUM, _REGULAR),
        (_MEDIUM_SHA, _REGULAR_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V19 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-proposal-confirmation-calibrator-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "stored_splits_frozen_candidate_p1_preregistered_execution_blocked",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": ["P1", "P2", "P3"],
        "currently_preregistered_candidate": "P1",
        "execution_authorized": False,
        "execution_blocker": (
            "The frozen V19 stored splits, candidate configuration, source-bound runner, and "
            "one-use public gate must be committed before a separate P1 authorization."
        ),
        "defect_class": (
            "aggregate consumed V18 P1 evidence passed recognition and role thresholds but "
            "retained one prohibited false region and missed three truths, showing that one fixed "
            "recognition-confidence threshold cannot jointly preserve recall and exclusion"
        ),
        "trigger_evidence": {
            "result_path": TRIGGER_RESULT_PATH,
            "result_sha256": TRIGGER_RESULT_SHA256,
            "scene_count": 192,
            "exact_scene_count": 189,
            "truth_regions": 1536,
            "true_positives": 1533,
            "false_regions": 1,
            "missed_regions": 3,
            "duplicate_regions": 0,
            "prohibited_structure_hits": 1,
            "recognition_exact": 0.96875,
            "character_error_rate": 0.004607852548718441,
            "role_accuracy": 0.998046875,
            "minimum_role_accuracy": 0.9895833333333334,
            "case_level_details_used": False,
            "fixture_bytes_scene_truth_or_case_identity_used": False,
            "consumed_v18_candidate_or_gate_rerun_authorized": False,
        },
        "isolated_change": (
            "freeze the exact V17 P3 detector and official recognizer, replace the failed scalar "
            "confidence veto with one small project-trained calibrator over generic detector, role, "
            "CTC, geometry, and morphology evidence extracted only from fresh V19 training scenes"
        ),
        "architecture": "proposal-evidence-mlp-calibrator-v1",
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
                "detector": 2,
                "role_probabilities": 8,
                "ctc_statistics": 8,
                "geometry": 9,
                "morphology": 4,
            },
            "output": ["proposal_count", 2],
            "hidden_width": 32,
            "activation": "ReLU",
            "model_license": "Apache-2.0",
            "role_logits_modified": False,
        },
        "training": {
            "candidate": "P1",
            "seed": SEED,
            "epochs": 20,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "negative_cap_per_scene": TRAINING_NEGATIVE_CAP_PER_SCENE,
            "negative_sampling": "highest-frozen-detector-probability-nontruth-v1",
            "negative_class_weight": 2.0,
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
            "v18_fixture_bytes_scene_truth_or_case_identity_reused": False,
            "validation_or_public_pixels_used_for_training": False,
        },
        "data_scope": (
            "fresh procedural scientific graph scenes only; no Chandler, Generalization, private "
            "or article images, external datasets, downloaded training data, predecessor fixture "
            "bytes, or V18 validation case identities"
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


__all__ = [
    "CHARACTER_ERROR_RATE_MAXIMUM", "DETECTOR_FLOOR", "DETECTOR_PATH", "DETECTOR_SHA256",
    "EXPERIMENT_BUDGET", "FEATURE_COUNT", "LICENSE_PATH", "LICENSE_SHA256", "NOTICE_PATH",
    "NOTICE_SHA256", "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR", "PUBLIC_REVISION",
    "RECOGNITION_EXACT_MINIMUM", "RECOGNIZER_PATH", "RECOGNIZER_SHA256",
    "RECOGNIZER_YAML_PATH", "RECOGNIZER_YAML_SHA256", "REVISION", "ROBUST_THRESHOLD_RUN_LENGTH",
    "ROLE_ACCURACY_MINIMUM", "ROLE_CLASS_ACCURACY_MINIMUM", "ROLE_ORDER", "SEED", "SPLITS",
    "TASK", "THRESHOLDS", "TRAINING_NEGATIVE_CAP_PER_SCENE", "TRIGGER_RESULT_PATH",
    "TRIGGER_RESULT_SHA256", "TRUTH_MATCH_IOU_MINIMUM", "protocol_configuration", "split_registration",
]
