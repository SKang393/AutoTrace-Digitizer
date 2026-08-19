# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration for OCR relational-neighborhood proposal V28."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-relational-neighborhood-proposal-v28"
SEED = 2_608_182_801
CANDIDATE_LIMIT = 3
FEATURE_COUNT = 31
RELATION_FEATURE_COUNT = 19
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
ROLE_PARENT_CHECKPOINT_PATH = (
    "ml/ocr/crop_evidence_role_anchor_v24/artifacts/P2-run/"
    "graph-text-crop-evidence-role-anchor-v24-p2.pt"
)
ROLE_PARENT_CHECKPOINT_SHA256 = (
    "8362c155285683130fd3e173db87d40ead0bd6df934e5b239bf455c2f9a0eb73"
)
ROLE_PARENT_ONNX_PATH = (
    "ml/ocr/crop_evidence_role_anchor_v24/artifacts/P2-run/"
    "graph-text-crop-evidence-role-anchor-v24-p2.onnx"
)
ROLE_PARENT_ONNX_SHA256 = (
    "c3276b2109509dddae5b6aea8a7a8ee2ee82960dcbe659834ef1a7cb6c3ea7e6"
)
TRIGGER_RESULT_PATH = "ml/ocr/morphology_veto_proposal_v27/P3_RESULT.json"
TRIGGER_RESULT_SHA256 = "8989a7765d2f64b10039ef26359d95534281be0c788cb4dc20a28ac3067e566d"


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
        "train", 256, 16_349_000,
        (
            "relational-neighborhood-clusters-v28-train-a",
            "relational-neighborhood-margins-v28-train-b",
            "relational-neighborhood-rows-v28-train-c",
            "relational-neighborhood-columns-v28-train-d",
        ),
        (
            "checker-luminance-v28-train",
            "local-gamma-v28-train",
            "paired-columns-v28-train",
            "soft-quantization-v28-train",
        ),
        (_REGULAR, _MEDIUM, _SEMIBOLD),
    ),
    "validation": SplitRegistration(
        "validation", 128, 17_071_000,
        (
            "relational-neighborhood-chains-v28-selection-e",
            "relational-neighborhood-satellites-v28-selection-f",
            "relational-neighborhood-bands-v28-selection-g",
            "relational-neighborhood-corners-v28-selection-h",
        ),
        (
            "radial-luminance-v28-selection",
            "paired-rows-v28-selection",
            "directional-halo-v28-selection",
            "asymmetric-quantization-v28-selection",
        ),
        (_SEMIBOLD, _REGULAR),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public", 192, 17_813_000,
        (
            "relational-neighborhood-lattices-v28-public-i",
            "relational-neighborhood-islands-v28-public-j",
            "relational-neighborhood-bridges-v28-public-k",
            "relational-neighborhood-rings-v28-public-l",
        ),
        (
            "cross-gradient-v28-public",
            "sparse-columns-v28-public",
            "piecewise-gamma-v28-public",
            "paired-bands-v28-public",
        ),
        (_MEDIUM, _SEMIBOLD),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V28 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-relational-neighborhood-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_fixture_freeze_or_candidate_execution",
        "defect_class": (
            "V27 P1, P2, and P3 each retained the same three prohibited false "
            "regions and produced no three-threshold zero-error window despite "
            "strict CPU ONNX parity and exact role preservation"
        ),
        "predecessor_aggregate_only": {
            "v27_scene_count": 128,
            "v27_exact_scene_count": 123,
            "v27_true_regions": 1024,
            "v27_false_regions": 3,
            "v27_missed_regions": 0,
            "v27_duplicate_regions": 0,
            "v27_prohibited_structure_hits": 3,
            "v27_recognition_exact": 0.98046875,
            "v27_character_error_rate": 0.0033123550844650546,
            "v27_role_accuracy": 0.998046875,
            "v27_onnx_parity_maximum_absolute_error": 4.76837158203125e-06,
            "v27_candidates_consumed": 3,
            "case_level_evidence_used": False,
            "fixture_bytes_truth_scene_or_case_identity_reused": False,
        },
        "isolated_change": (
            "replace per-proposal morphology vetoing with a project-owned proposal "
            "head trained from scratch over explicit ordered pair geometry and "
            "permutation-equivariant scene message passing; freeze only the exact "
            "consumed V24 role anchor"
        ),
        "architecture": {
            "id": "pairwise-geometry-relational-neighborhood-v1",
            "evidence_input": [1, "proposal_count", FEATURE_COUNT],
            "crop_input": [1, "proposal_count", CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
            "relation_input": [
                1, "proposal_count", "proposal_count", RELATION_FEATURE_COUNT,
            ],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "proposal_weights_initialized_from_scratch": True,
            "role_parent_weights_frozen": True,
            "role_parent_argmax_preserved_exactly": True,
            "pair_features": (
                "ordered normalized center offsets and magnitudes, log size ratios, "
                "axis overlaps, IoU, axis gaps, row and column alignment, neighbor "
                "plot position, shared plot-side indicators, and identity"
            ),
            "message_passing": (
                "two permutation-equivariant attention blocks with learned pairwise "
                "geometry bias and pairwise value embeddings"
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
            "objective": "relational-neighborhood-asymmetric-scene-margin-v1",
            "objective_formula": (
                "weighted proposal cross entropy plus positive-floor, negative-"
                "ceiling, per-scene separation, and prohibited-negative top-k "
                "margin penalties"
            ),
            "seed": SEED,
            "epochs": 4,
            "expected_optimizer_steps": 1024,
            "learning_rate": 0.0004,
            "weight_decay": 0.0005,
            "gradient_clip_norm": 5.0,
            "proposal_cross_entropy_weight": 1.0,
            "false_positive_weight": 5.0,
            "positive_logit_margin_floor": 2.0,
            "negative_logit_margin_ceiling": -2.0,
            "positive_floor_weight": 2.0,
            "negative_ceiling_weight": 5.0,
            "scene_separation_logit_margin_minimum": 4.0,
            "scene_separation_weight": 2.5,
            "hard_negative_top_k": 4,
            "hard_negative_weight": 3.0,
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
            "role_parent_checkpoint_path": ROLE_PARENT_CHECKPOINT_PATH,
            "role_parent_checkpoint_sha256": ROLE_PARENT_CHECKPOINT_SHA256,
            "role_parent_onnx_path": ROLE_PARENT_ONNX_PATH,
            "role_parent_onnx_sha256": ROLE_PARENT_ONNX_SHA256,
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
            "detector_recognizer_role_parent_and_candidate_tensor_hashes_required": True,
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
            "Generalization, private or article images, external datasets, V27 "
            "fixture bytes, or V27 case identities"
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
    "RECOGNIZER_PATH", "RECOGNIZER_SHA256", "RECOGNIZER_YAML_PATH",
    "RECOGNIZER_YAML_SHA256", "RELATION_FEATURE_COUNT", "REVISION",
    "ROLE_ORDER", "ROLE_PARENT_CHECKPOINT_PATH", "ROLE_PARENT_CHECKPOINT_SHA256",
    "ROLE_PARENT_ONNX_PATH", "ROLE_PARENT_ONNX_SHA256", "SEED", "TASK",
    "THRESHOLDS", "TRIGGER_RESULT_PATH", "TRIGGER_RESULT_SHA256",
    "protocol_configuration", "split_registration",
]
