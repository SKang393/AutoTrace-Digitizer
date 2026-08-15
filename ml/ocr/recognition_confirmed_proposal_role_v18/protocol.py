# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for recognition-confirmed OCR proposal and role V18."""

from __future__ import annotations

from dataclasses import asdict, dataclass


TASK = "ocr-detection-recognition"
REVISION = "graph-text-recognition-confirmed-proposal-role-v18"
PUBLIC_REVISION = f"{REVISION}-public-v1"
CANDIDATE_ID = "P1"
EXPERIMENT_BUDGET = 1
DETECTOR_THRESHOLD = 0.64
RECOGNITION_CONFIDENCE_THRESHOLD = 0.60
TRUTH_MATCH_IOU_MINIMUM = 0.5
RECOGNITION_EXACT_MINIMUM = 0.90
CHARACTER_ERROR_RATE_MAXIMUM = 0.05
ROLE_ACCURACY_MINIMUM = 0.90
ROLE_CLASS_ACCURACY_MINIMUM = 0.85
ROLE_ORDER = (
    "YTick", "XTick", "AxisTitle", "PhaseHeading",
    "LegendText", "Participant", "Annotation", "Other",
)

DETECTOR_PATH = (
    "ml/ocr/structural_veto_proposal_role_v17/artifacts/P3-run/"
    "graph-text-structural-veto-proposal-role-v17-p3.onnx"
)
DETECTOR_SHA256 = "ca32487f1df2c3fea1b8c2f51daf7578ed9756e9140d1b0eaf2a16b283591262"
DETECTOR_RESULT_PATH = "ml/ocr/structural_veto_proposal_role_v17/P3_RESULT.json"
DETECTOR_RESULT_SHA256 = "917353b0449b884bd2536e4b4581b700c80cbbf3b787235eca146975e414d8f1"
RECOGNIZER_PATH = "ml/ocr/official_bakeoff/runs/conversion/en_PP-OCRv5_mobile_rec.onnx"
RECOGNIZER_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
RECOGNIZER_YAML_PATH = (
    "ml/ocr/official_bakeoff/runs/extracted/en_PP-OCRv5_mobile_rec_infer/inference.yml"
)
RECOGNIZER_YAML_SHA256 = "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
FEASIBILITY_PATH = "ml/ocr/recognition_confirmed_proposal_role_v18/FEASIBILITY_EVIDENCE.json"
FEASIBILITY_SHA256 = "cd309eb202cffe97f6b43ef2d337a300eb3da955e8a85d0b0c10e6d1f425164e"
NOTICE_PATH = "LICENSES/PaddlePaddle-PP-OCRv5-Models-Notice.txt"
NOTICE_SHA256 = "8d81f5d0c58547cce471c24f82efe768a9d907d06764f67e90cc680c6d777729"
LICENSE_PATH = "LICENSES/PaddlePaddle-PP-OCRv5-Models-Apache-2.0.txt"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"


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
        "validation", 192, 4_217_000, "recognition-confirmed-lattice-v18-validation",
        "recognition-confidence-envelope-v18-validation", (_SEMIBOLD, _REGULAR),
        (_SEMIBOLD_SHA, _REGULAR_SHA),
    ),
    SplitRegistration(
        "sealed_public", 256, 4_491_000, "recognition-confirmed-interleave-v18-public",
        "recognition-confidence-edge-band-v18-public", (_MEDIUM, _REGULAR),
        (_MEDIUM_SHA, _REGULAR_SHA),
    ),
)


def split_registration(split: str) -> SplitRegistration:
    try:
        return next(item for item in SPLITS if item.split == split)
    except StopIteration as error:
        raise ValueError(f"Unknown OCR V18 split: {split}") from error


def protocol_configuration() -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-recognition-confirmed-proposal-role-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "state": "design_preregistered_before_stored_split_materialization",
        "defect_class": (
            "the exhausted V17 P3 detector retained one prohibited false proposal at its last "
            "all-truth threshold, while an aggregate visible-only feasibility pass showed the "
            "exact official recognizer confidence separated that false proposal from every truth"
        ),
        "isolated_change": (
            "reuse the exact immutable V17 P3 proposal and role ONNX at threshold 0.64 and the "
            "exact official PP-OCRv5 English mobile recognition ONNX, rejecting only proposals "
            "whose nonblank collapsed CTC path has mean selected-character probability below 0.60"
        ),
        "architecture": "exact-v17-p3-plus-official-recognizer-confidence-confirmation-v1",
        "experiment_budget": EXPERIMENT_BUDGET,
        "candidate_ids": [CANDIDATE_ID],
        "currently_preregistered_candidate": None,
        "execution_authorized": False,
        "execution_blocker": (
            "Fresh V18 validation and truth-hidden public fixture bytes, their checksums, the "
            "zero-optimizer candidate configuration, runner sources, and one-use gate must be "
            "committed before a separate authorization commit."
        ),
        "optimizer_steps": 0,
        "fixed_composition": {
            "detector_path": DETECTOR_PATH,
            "detector_sha256": DETECTOR_SHA256,
            "detector_result_path": DETECTOR_RESULT_PATH,
            "detector_result_sha256": DETECTOR_RESULT_SHA256,
            "detector_probability_threshold": DETECTOR_THRESHOLD,
            "recognizer_path": RECOGNIZER_PATH,
            "recognizer_sha256": RECOGNIZER_SHA256,
            "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
            "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
            "recognition_confidence_measure": "mean-softmax-probability-of-collapsed-nonblank-ctc-path",
            "recognition_confidence_threshold": RECOGNITION_CONFIDENCE_THRESHOLD,
            "empty_ctc_path_confidence": 0.0,
            "provider": "CPUExecutionProvider",
            "session_options": "single-threaded-sequential-deterministic",
        },
        "feasibility_evidence": {
            "path": FEASIBILITY_PATH,
            "sha256": FEASIBILITY_SHA256,
            "visible_v17_validation_only": True,
            "case_level_details_used": False,
            "fixture_bytes_or_case_identity_reused_by_v18": False,
            "public_or_private_data_used": False,
            "additional_parameter_sweeps_authorized": False,
        },
        "artifact_provenance": {
            "detector_license": "Apache-2.0 project-trained weights",
            "recognizer_license": "Apache-2.0 official PaddlePaddle model weights",
            "license_path": LICENSE_PATH,
            "license_sha256": LICENSE_SHA256,
            "notice_path": NOTICE_PATH,
            "notice_sha256": NOTICE_SHA256,
            "model_store_promotion_authorized": False,
            "packaging_authorized": False,
        },
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
            "provider": "CPUExecutionProvider",
            "direct_fixture_byte_execution_required": True,
            "detector_and_recognizer_tensor_stream_hashes_required": True,
            "case_level_details_emitted": False,
        },
        "splits": [asdict(item) for item in SPLITS],
        "split_policy": {
            "validation_and_public_family_ids_disjoint": True,
            "sealed_public_truth_hidden_until_one_time_gate": True,
            "predecessor_fixture_bytes_reused": False,
            "v17_fixture_bytes_scene_truth_or_case_identity_reused": False,
            "validation_or_public_pixels_used_for_threshold_selection": False,
            "public_case_level_failure_analysis_permitted": False,
        },
        "data_scope": (
            "fresh procedural scientific graph scenes only; no Chandler, Generalization, private "
            "or article images, external datasets, downloaded training data, predecessor fixture "
            "bytes, or V17 validation case identities"
        ),
        "marker_creation_gate_required_before_approval": True,
        "manifest_created": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "CANDIDATE_ID", "CHARACTER_ERROR_RATE_MAXIMUM", "DETECTOR_PATH", "DETECTOR_RESULT_PATH",
    "DETECTOR_RESULT_SHA256", "DETECTOR_SHA256", "DETECTOR_THRESHOLD", "EXPERIMENT_BUDGET",
    "FEASIBILITY_PATH", "FEASIBILITY_SHA256", "LICENSE_PATH", "LICENSE_SHA256", "NOTICE_PATH",
    "NOTICE_SHA256", "PUBLIC_REVISION", "RECOGNITION_CONFIDENCE_THRESHOLD",
    "RECOGNITION_EXACT_MINIMUM", "RECOGNIZER_PATH", "RECOGNIZER_SHA256",
    "RECOGNIZER_YAML_PATH", "RECOGNIZER_YAML_SHA256", "REVISION", "ROLE_ACCURACY_MINIMUM",
    "ROLE_CLASS_ACCURACY_MINIMUM", "ROLE_ORDER", "SPLITS", "TASK", "TRUTH_MATCH_IOU_MINIMUM",
    "protocol_configuration", "split_registration",
]
