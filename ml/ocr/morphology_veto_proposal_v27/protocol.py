# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration for OCR morphology-veto proposal V27."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-morphology-veto-proposal-v27"
SEED = 2_608_182_701
CANDIDATE_LIMIT = 3
FEATURE_COUNT = 31
STRUCTURE_FEATURE_COUNT = 24
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
OUTPUT_LOGIT_SCALE = 0.5

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
    "ml/ocr/scene_topology_proposal_v26/artifacts/P3-run/"
    "graph-text-scene-topology-proposal-v26-p3.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "fcf8806107dd1881c596be66f209c62170a892586ebf75efc6381f9acc4eb6ae"
)
PARENT_ONNX_PATH = (
    "ml/ocr/scene_topology_proposal_v26/artifacts/P3-run/"
    "graph-text-scene-topology-proposal-v26-p3.onnx"
)
PARENT_ONNX_SHA256 = (
    "9f4aa80722285207ec4be860782f5b443775d024817f035264eee07804aefe17"
)
TRIGGER_RESULT_PATH = "ml/ocr/scene_topology_proposal_v26/P3_RESULT.json"
TRIGGER_RESULT_SHA256 = "1b5f253c27e4e7a900c93264ebefbd5aa92f891a2472f9971f51da063057696f"


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
        "train", 256, 14_117_000,
        (
            "morphology-combs-v27-train-a",
            "morphology-rails-v27-train-b",
            "morphology-corners-v27-train-c",
            "morphology-loops-v27-train-d",
        ),
        (
            "local-contrast-v27-train",
            "alternating-columns-v27-train",
            "edge-softening-v27-train",
            "ink-threshold-v27-train",
        ),
        (_REGULAR, _MEDIUM, _SEMIBOLD),
    ),
    "validation": SplitRegistration(
        "validation", 128, 14_831_000,
        (
            "morphology-spurs-v27-selection-e",
            "morphology-frames-v27-selection-f",
            "morphology-hooks-v27-selection-g",
            "morphology-crossings-v27-selection-h",
        ),
        (
            "diagonal-luminance-v27-selection",
            "row-quantization-v27-selection",
            "context-halo-v27-selection",
            "subpixel-column-v27-selection",
        ),
        (_SEMIBOLD, _REGULAR),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public", 192, 15_593_000,
        (
            "morphology-teeth-v27-public-i",
            "morphology-enclosures-v27-public-j",
            "morphology-junctions-v27-public-k",
            "morphology-ladders-v27-public-l",
        ),
        (
            "paired-gradient-v27-public",
            "sparse-row-v27-public",
            "asymmetric-gamma-v27-public",
            "band-luminance-v27-public",
        ),
        (_MEDIUM, _SEMIBOLD),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V27 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-morphology-veto-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_fixture_freeze_or_candidate_execution",
        "defect_class": (
            "V26 P3 retained all 1024 visible-selection truths but left one "
            "prohibited false region at two adjacent high thresholds, produced no "
            "three-threshold zero-error window, and exceeded strict CPU ONNX parity "
            "despite ORT_ENABLE_BASIC"
        ),
        "predecessor_aggregate_only": {
            "v26_scene_count": 128,
            "v26_exact_scene_count": 122,
            "v26_true_regions": 1024,
            "v26_false_regions": 1,
            "v26_missed_regions": 0,
            "v26_duplicate_regions": 0,
            "v26_prohibited_structure_hits": 1,
            "v26_recognition_exact": 0.97265625,
            "v26_character_error_rate": 0.004640371229698376,
            "v26_role_accuracy": 0.9951171875,
            "v26_onnx_parity_maximum_absolute_error": 0.000011444091796875,
            "case_level_evidence_used": False,
            "fixture_bytes_truth_scene_or_case_identity_reused": False,
        },
        "isolated_change": (
            "freeze the exact consumed V26 P3 parent, add a project-owned residual "
            "veto head over 24 explicit binary projection and morphology features, "
            "and apply a fixed 0.5 output-logit scale for deterministic CPU parity"
        ),
        "architecture": {
            "id": "frozen-v26-morphology-veto-scaled-v1",
            "evidence_input": [1, "proposal_count", FEATURE_COUNT],
            "crop_input": [1, "proposal_count", CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
            "structure_input": [1, "proposal_count", STRUCTURE_FEATURE_COUNT],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "parent_weights_frozen": True,
            "parent_role_argmax_preserved_exactly": True,
            "veto_head_initialization": "deterministic-from-scratch",
            "structure_features": (
                "per tight/context channel: binary ink fraction, active row and "
                "column fractions, row and column peaks, horizontal and vertical "
                "spans, row and column transition densities, edge, center, and "
                "corner occupancy"
            ),
            "runtime_numeric_precision": "float32",
            "output_logit_scale": OUTPUT_LOGIT_SCALE,
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
            "execution_providers": ["CPUExecutionProvider"],
            "ordinary_production_reachable": False,
            "model_license": "Apache-2.0",
        },
        "candidate_budget": {
            "candidate_limit": CANDIDATE_LIMIT,
            "candidate_number": 1,
            "optimizer_steps_maximum": 1024,
            "selection_execution_limit_per_candidate": 1,
            "public_execution_limit": 1,
        },
        "candidate_p1": {
            "seed": SEED,
            "epochs": 4,
            "expected_optimizer_steps": 1024,
            "learning_rate": 0.0005,
            "weight_decay": 0.0005,
            "gradient_clip_norm": 5.0,
            "proposal_cross_entropy_weight": 1.0,
            "false_positive_weight": 4.0,
            "positive_logit_margin_floor": 2.0,
            "negative_logit_margin_ceiling": -2.0,
            "positive_floor_weight": 2.0,
            "negative_ceiling_weight": 4.0,
            "scene_separation_logit_margin_minimum": 4.0,
            "scene_separation_weight": 2.0,
            "complete_proposal_negative_cap_per_scene": 10000,
            "recognition_batch_size": 64,
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
            "trigger_result_path": TRIGGER_RESULT_PATH,
            "trigger_result_sha256": TRIGGER_RESULT_SHA256,
            "provider": "CPUExecutionProvider",
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
            "public_case_level_failure_analysis_permitted": False,
        },
        "data_scope": (
            "fresh project-owned procedural graph scenes only; no Chandler, "
            "Generalization, private or article images, external datasets, "
            "V26 fixture bytes, or V26 case identities"
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
    "CANDIDATE_LIMIT", "CROP_CHANNELS", "CROP_HEIGHT", "CROP_WIDTH",
    "DETECTOR_PATH", "DETECTOR_SHA256", "FEATURE_COUNT",
    "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR", "OUTPUT_LOGIT_SCALE",
    "PARENT_CHECKPOINT_PATH",
    "PARENT_CHECKPOINT_SHA256", "PARENT_ONNX_PATH", "PARENT_ONNX_SHA256",
    "RECOGNIZER_PATH", "RECOGNIZER_SHA256", "RECOGNIZER_YAML_PATH",
    "RECOGNIZER_YAML_SHA256", "REVISION", "ROLE_ORDER", "SEED",
    "STRUCTURE_FEATURE_COUNT", "TASK", "THRESHOLDS", "TRIGGER_RESULT_PATH",
    "TRIGGER_RESULT_SHA256", "protocol_configuration", "split_registration",
]
