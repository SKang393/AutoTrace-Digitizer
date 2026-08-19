# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration for OCR scene-topology proposal V26."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-scene-topology-proposal-v26"
SEED = 2_608_182_601
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
TRIGGER_RESULT_PATH = "ml/ocr/evidence_rescue_v25/P3_RESULT.json"
TRIGGER_RESULT_SHA256 = "010034ad8a38dbf2671f6ef5cd7fb8695b0accdf6b4cdb6d960811b2d2a9bca3"


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
        "train", 384, 12_113_000,
        (
            "topology-shelves-v26-train-a",
            "topology-gutters-v26-train-b",
            "topology-islands-v26-train-c",
            "topology-ribbons-v26-train-d",
        ),
        (
            "anisotropic-gamma-v26-train",
            "subpixel-row-phase-v26-train",
            "ink-spread-v26-train",
            "background-slope-v26-train",
        ),
        (_REGULAR, _MEDIUM, _SEMIBOLD),
    ),
    "validation": SplitRegistration(
        "validation", 128, 12_641_000,
        (
            "topology-terraces-v26-selection-e",
            "topology-corridors-v26-selection-f",
            "topology-anchors-v26-selection-g",
            "topology-pockets-v26-selection-h",
        ),
        (
            "cross-axis-shading-v26-selection",
            "alternating-ink-v26-selection",
            "soft-shoulder-v26-selection",
            "quantized-gradient-v26-selection",
        ),
        (_SEMIBOLD, _REGULAR),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public", 192, 13_277_000,
        (
            "topology-courtyards-v26-public-i",
            "topology-lanes-v26-public-j",
            "topology-bridges-v26-public-k",
            "topology-bays-v26-public-l",
        ),
        (
            "oblique-luminance-v26-public",
            "paired-band-v26-public",
            "sparse-column-v26-public",
            "asymmetric-quantization-v26-public",
        ),
        (_MEDIUM, _SEMIBOLD),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V26 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "schema": "graphreader.ocr-scene-topology-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_fixture_freeze_or_candidate_execution",
        "defect_class": (
            "V25 P3 retained 1017 of 1024 visible-selection truths with one false "
            "prohibited region and seven misses at every fixed threshold despite "
            "passing recognition, role, and ONNX parity gates"
        ),
        "predecessor_aggregate_only": {
            "v25_scene_count": 128,
            "v25_exact_scene_count": 112,
            "v25_true_regions": 1017,
            "v25_false_regions": 1,
            "v25_missed_regions": 7,
            "v25_duplicate_regions": 0,
            "v25_prohibited_structure_hits": 1,
            "v25_recognition_exact": 0.96484375,
            "v25_character_error_rate": 0.010632995514205018,
            "v25_role_accuracy": 0.9833984375,
            "case_level_evidence_used": False,
            "fixture_bytes_truth_scene_or_case_identity_reused": False,
        },
        "isolated_change": (
            "replace the exhausted residual proposal path with a project-owned "
            "proposal head trained from scratch over complete production proposals, "
            "axial crop topology, generic evidence, and scene aggregates while "
            "preserving the exact frozen V24 role logits"
        ),
        "architecture": {
            "id": "frozen-role-axial-topology-proposal-v1",
            "evidence_input": [1, "proposal_count", FEATURE_COUNT],
            "crop_input": [1, "proposal_count", CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "proposal_head_initialization": "deterministic-from-scratch",
            "role_parent_weights_frozen": True,
            "roles_preserved_exactly": True,
            "crop_features": (
                "depthwise spatial stem plus global mean/max, eight-bin row, and "
                "sixteen-bin column projections"
            ),
            "scene_features": "per-scene evidence mean and maximum",
            "ordinary_production_reachable": False,
            "model_license": "Apache-2.0",
        },
        "candidate_budget": {
            "candidate_limit": CANDIDATE_LIMIT,
            "candidate_number": 1,
            "optimizer_steps_maximum": 2304,
            "selection_execution_limit_per_candidate": 1,
            "public_execution_limit": 1,
        },
        "candidate_p1": {
            "seed": SEED,
            "epochs": 6,
            "expected_optimizer_steps": 2304,
            "learning_rate": 0.0004,
            "weight_decay": 0.001,
            "gradient_clip_norm": 5.0,
            "proposal_cross_entropy_weight": 1.0,
            "positive_logit_margin_floor": 2.5,
            "negative_logit_margin_ceiling": -2.5,
            "positive_floor_weight": 3.0,
            "negative_ceiling_weight": 3.0,
            "scene_separation_logit_margin_minimum": 5.0,
            "scene_separation_weight": 1.5,
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
    "CANDIDATE_LIMIT", "CROP_CHANNELS", "CROP_HEIGHT", "CROP_WIDTH",
    "DETECTOR_PATH", "DETECTOR_SHA256", "FEATURE_COUNT",
    "ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR", "RECOGNIZER_PATH",
    "RECOGNIZER_SHA256", "RECOGNIZER_YAML_PATH", "RECOGNIZER_YAML_SHA256",
    "REVISION", "ROLE_ORDER", "ROLE_PARENT_CHECKPOINT_PATH",
    "ROLE_PARENT_CHECKPOINT_SHA256", "ROLE_PARENT_ONNX_PATH",
    "ROLE_PARENT_ONNX_SHA256", "SEED", "TASK", "THRESHOLDS",
    "TRIGGER_RESULT_PATH", "TRIGGER_RESULT_SHA256", "protocol_configuration",
    "split_registration",
]
