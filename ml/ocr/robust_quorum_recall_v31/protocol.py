# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fail-closed preregistration for OCR robust quorum recall V31."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-robust-quorum-recall-v31"
SEED = 2_608_193_101
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
    "ml/ocr/official_bakeoff/runs/extracted/en_PP-OCRv5_mobile_rec_infer/"
    "inference.yml"
)
RECOGNIZER_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
V30_CHECKPOINT_PATH = (
    "ml/ocr/unanimous_structure_veto_v30/artifacts/P1-run/"
    "graph-text-unanimous-structure-veto-v30-p1.pt"
)
V30_CHECKPOINT_SHA256 = (
    "a91044ee4621fe914377ad522d7ee3d9036f0a5e5714168684c34d2c3b8a9ceb"
)
V30_ONNX_PATH = (
    "ml/ocr/unanimous_structure_veto_v30/artifacts/P1-run/"
    "graph-text-unanimous-structure-veto-v30-p1.onnx"
)
V30_ONNX_SHA256 = (
    "78425c5b4a45ef2cbf99086243af0ede96c91b2b6afcdac1daa71bfeb5e55c18"
)
TRIGGER_RESULT_PATH = "ml/ocr/unanimous_structure_veto_v30/PUBLIC_GATE_RESULT.json"
TRIGGER_RESULT_SHA256 = (
    "c070c1acd1b803f579529055949e363b97d24cb4207233a35d86b87b2e691e3c"
)


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
        "train", 384, 23_711_000,
        (
            "quorum-recall-text-fields-v31-train-a",
            "quorum-recall-line-clusters-v31-train-b",
            "quorum-recall-bracket-fields-v31-train-c",
            "quorum-recall-legend-crossings-v31-train-d",
        ),
        (
            "anisotropic-soften-v31-train",
            "offset-quantization-v31-train",
            "ranked-stroke-v31-train",
            "local-gamma-v31-train",
        ),
        (_REGULAR, _MEDIUM, _SEMIBOLD),
    ),
    "validation": SplitRegistration(
        "validation", 192, 24_487_000,
        (
            "quorum-recall-isolated-text-v31-selection-e",
            "quorum-recall-dense-connectors-v31-selection-f",
            "quorum-recall-axis-adjacency-v31-selection-g",
            "quorum-recall-structure-boundaries-v31-selection-h",
        ),
        (
            "radial-soften-v31-selection",
            "paired-gamma-v31-selection",
            "staggered-block-v31-selection",
            "asymmetric-quantization-v31-selection",
        ),
        (_SEMIBOLD, _REGULAR),
    ),
    "sealed_public": SplitRegistration(
        "sealed_public", 256, 25_303_000,
        (
            "quorum-recall-structure-lattices-v31-public-i",
            "quorum-recall-crossing-bands-v31-public-j",
            "quorum-recall-label-adjacency-v31-public-k",
            "quorum-recall-mixed-fields-v31-public-l",
        ),
        (
            "oblique-soften-v31-public",
            "phase-shift-grid-v31-public",
            "local-rank-v31-public",
            "edge-quantization-v31-public",
        ),
        (_MEDIUM, _SEMIBOLD),
    ),
}


def split_registration(split: str) -> SplitRegistration:
    try:
        return _SPLITS[split]
    except KeyError as error:
        raise ValueError(f"unsupported OCR V31 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-robust-quorum-recall-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "preregistered_before_fixture_freeze_or_candidate_execution",
        "defect_class": (
            "V30 passed 255 of 256 truth-hidden public scenes with zero false "
            "regions, zero duplicates, zero prohibited hits, and one missed "
            "truth at every fixed threshold"
        ),
        "predecessor_aggregate_only": {
            "v30_public_scene_count": 256,
            "v30_public_exact_scene_count": 255,
            "v30_public_truth_regions": 2048,
            "v30_public_true_positives": 2047,
            "v30_public_false_regions": 0,
            "v30_public_missed_regions": 1,
            "v30_public_duplicate_regions": 0,
            "v30_public_prohibited_structure_hits": 0,
            "v30_public_recognition_exact": 0.98095703125,
            "v30_public_character_error_rate": 0.0033769523005487546,
            "v30_public_role_accuracy": 0.99951171875,
            "v30_public_minimum_role_accuracy": 0.99609375,
            "v30_public_archive_read_count": 1,
            "v30_public_gate_consumed": True,
            "case_level_evidence_used": False,
            "fixture_bytes_truth_scene_or_case_identity_reused": False,
        },
        "isolated_change": (
            "reuse the exact project-owned V30 route weights with zero optimizer "
            "steps, replace strict minimum-margin unanimity with the median "
            "positive-vs-negative margin across the three routes, and retain "
            "deterministic geometry roles"
        ),
        "architecture": {
            "id": "two-of-three-robust-route-quorum-v1",
            "evidence_input": [1, "proposal_count", FEATURE_COUNT],
            "crop_input": [1, "proposal_count", CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH],
            "relation_input": [
                1, "proposal_count", "proposal_count", RELATION_FEATURE_COUNT,
            ],
            "output": [1, "proposal_count", 2 + len(ROLE_ORDER)],
            "route_weight_source": V30_CHECKPOINT_PATH,
            "route_weight_sha256": V30_CHECKPOINT_SHA256,
            "optimizer_steps": 0,
            "proposal_consensus": (
                "median positive-vs-negative margin across attention, invariant "
                "relation-summary, and local-structure routes"
            ),
            "role_output": "source-declared plot-relative deterministic partition",
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
            "optimizer_steps_maximum": 0,
            "selection_execution_limit_per_candidate": 1,
            "public_execution_limit": 1,
        },
        "candidate_p1": {
            "objective": "zero-training-two-of-three-route-quorum-v1",
            "seed": SEED,
            "expected_optimizer_steps": 0,
            "predecessor_checkpoint_reused": True,
        },
        "fixed_inputs": {
            "detector_path": DETECTOR_PATH,
            "detector_sha256": DETECTOR_SHA256,
            "recognizer_path": RECOGNIZER_PATH,
            "recognizer_sha256": RECOGNIZER_SHA256,
            "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
            "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
            "v30_checkpoint_path": V30_CHECKPOINT_PATH,
            "v30_checkpoint_sha256": V30_CHECKPOINT_SHA256,
            "v30_onnx_path": V30_ONNX_PATH,
            "v30_onnx_sha256": V30_ONNX_SHA256,
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
            "detector_recognizer_candidate_and_route_tensor_hashes_required": True,
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
            "Generalization, private or article images, external datasets, V30 "
            "fixture bytes, V30 case identities, truth rows, or predictions"
        ),
        "fixture_identity_frozen": False,
        "candidate_execution_authorized": False,
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
    "ROLE_ORDER", "SEED", "TASK", "THRESHOLDS", "TRIGGER_RESULT_PATH",
    "TRIGGER_RESULT_SHA256", "V30_CHECKPOINT_PATH", "V30_CHECKPOINT_SHA256",
    "V30_ONNX_PATH", "V30_ONNX_SHA256", "protocol_configuration",
    "split_registration",
]
