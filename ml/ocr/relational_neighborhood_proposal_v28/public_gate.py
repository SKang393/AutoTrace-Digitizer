# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for the selected OCR V28 P3 payload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    require_committed_sources,
    sha256_file,
    source_bundle_sha256,
)
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _calibrated_records,
    _cpu_session,
    _is_ancestor,
    _read_json,
    _repository_head,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, metrics_pass
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive
from .pipeline import extract_relational_evidence
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    ROBUST_THRESHOLD_RUN_LENGTH,
    ROLE_PARENT_ONNX_PATH,
    ROLE_PARENT_ONNX_SHA256,
    TASK,
    THRESHOLDS,
)
from .train_p1 import _candidate_session, _validate_stored_split
from .train_p2 import (
    P1_ONNX_PATH,
    P1_ONNX_SHA256,
    _positive_probabilities,
)
from .train_p3 import RUNNER_SOURCE_PATHS as P3_RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/relational_neighborhood_proposal_v28")
LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")
SPLIT_SEAL_PATH = ROOT / "SPLIT_SEAL.json"
SPLIT_SEAL_SHA256 = "c968aeb5ec0a3440a9fa76b3a346d3652238230599043f801b4fa46ed9eef9bf"
PUBLIC_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
PUBLIC_OUTPUT_PATH = ROOT / "artifacts/public-gate-v1/report.json"
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v28-public.zip")
PUBLIC_ARCHIVE_SHA256 = "db00a6bffda5cefe3ecd747d89f930946782f05e7c5a5f013abf06d2a07e0946"
PUBLIC_MANIFEST_SHA256 = "1b4673389d32e898decb66389b473a6e536a779307950276c323a23ce7efd4f4"
PUBLIC_REVISION = "graph-text-relational-neighborhood-proposal-v28-public-v1"

P3_RESULT_PATH = ROOT / "P3_RESULT.json"
P3_RESULT_SHA256 = "0b56a7a72d4ed8cd2733cd80cfcad4c88ee0acbf404a87660b57cfe07df7bcf2"
P3_REPORT_PATH = ROOT / "artifacts/P3-run/candidate-report.json"
P3_REPORT_SHA256 = "16d60e8d74ea38910ef97fe09365eb37102c377a8018421566d48dcfdcce11fd"
P3_CHECKPOINT_PATH = (
    ROOT / "artifacts/P3-run/graph-text-relational-neighborhood-proposal-v28-p3.pt"
)
P3_CHECKPOINT_SHA256 = "3d3f9e147a3606af2d398ee5930a3bb51a60e6b0b41cc436d48ba1b26714e46b"
P3_ONNX_PATH = (
    ROOT / "artifacts/P3-run/graph-text-relational-neighborhood-proposal-v28-p3.onnx"
)
P3_ONNX_SHA256 = "4179534c1abfe7dd22e041d452d52269550f8a13471cbd18bacb0becd18b45af"
P3_OPENED_SEAL_PATH = (
    Path("ml/markers/training-seals/ocr-detection-recognition")
    / REVISION / "P3/opened.json"
)
P3_OPENED_SEAL_SHA256 = "dd9a66ee10586a6756fb7559ed50d0d4df68baaa4289b9f2a2b7408db840c7a8"
P3_RESULT_SEAL_PATH = (
    Path("ml/markers/training-seals/ocr-detection-recognition")
    / REVISION / "P3/result.json"
)
P3_RESULT_SEAL_SHA256 = "46f2f8a63784124cb8251ac0fe670451e53f9b345aa30b19da241957015d8067"
P3_SOURCE_COMMIT = "c5d6efebf2d4798137060aa905dfdf29cf698f27"

EVALUATOR_SOURCE_PATHS = tuple(dict.fromkeys((
    *P3_RUNNER_SOURCE_PATHS,
    P3_RESULT_PATH,
    ROOT / "public_gate.py",
)))
EXPECTED_CANDIDATE_HASH_KEYS = (
    "detector_onnx_sha256",
    "recognizer_onnx_sha256",
    "role_parent_onnx_sha256",
    "p1_onnx_sha256",
    "candidate_onnx_sha256",
    "candidate_checkpoint_sha256",
    "selection_result_sha256",
    "selection_report_sha256",
    "selection_opened_seal_sha256",
    "selection_result_seal_sha256",
)
GATE_CONFIG: dict[str, object] = {
    "evaluation_limit": 1,
    "minimum_consecutive_thresholds": ROBUST_THRESHOLD_RUN_LENGTH,
    "exact_region_and_role_every_scene_at_each_threshold": True,
    "false_regions": 0,
    "missed_regions": 0,
    "duplicate_regions": 0,
    "prohibited_structure_hits": 0,
    "recognition_exact_minimum": 0.90,
    "character_error_rate_maximum": 0.05,
    "role_accuracy_minimum": 0.90,
    "per_role_accuracy_minimum": 0.85,
    "provider": "CPUExecutionProvider",
    "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
    "p1_proposal_stream_preservation_required": True,
    "direct_fixture_byte_execution_required": True,
    "complete_production_proposal_stream_required": True,
    "detector_recognizer_candidate_relation_and_p1_tensor_hashes_required": True,
    "case_level_failure_analysis_permitted": False,
}


def _public_window(selection: dict[str, object]) -> tuple[float, ...]:
    selected = float(selection.get("selected_threshold", -1.0))
    window = tuple(float(value) for value in selection.get("passing_threshold_window", []))
    if (
        selected not in THRESHOLDS
        or selected not in window
        or len(window) < ROBUST_THRESHOLD_RUN_LENGTH
        or any(value not in THRESHOLDS for value in window)
    ):
        raise RuntimeError("OCR V28 selection has no preregistered robust threshold window")
    selected_index = window.index(selected)
    lower = max(0, selected_index - 1)
    result = window[lower:lower + ROBUST_THRESHOLD_RUN_LENGTH]
    if len(result) < ROBUST_THRESHOLD_RUN_LENGTH:
        result = window[-ROBUST_THRESHOLD_RUN_LENGTH:]
    return result


def _selected_result_is_terminal(selection: dict[str, Any]) -> bool:
    metrics = selection.get("selection_metrics", {})
    roles = metrics.get("per_role_accuracy", {})
    comparisons = selection.get("threshold_comparisons", [])
    return bool(
        selection.get("schema")
        == "graphreader.ocr-relational-neighborhood-selection-result.v1"
        and selection.get("task") == TASK
        and selection.get("revision") == REVISION
        and selection.get("candidate_id") == "P3"
        and selection.get("status") == "selected"
        and selection.get("candidate_consumed") is True
        and selection.get("selection_gate_passed") is True
        and selection.get("optimizer_steps") == 0
        and selection.get("weights_changed") is False
        and selection.get("training_role_matches") == 2048
        and selection.get("training_role_mismatches") == 0
        and selection.get("frozen_p1_parameters_preserved") is True
        and selection.get("p1_proposal_decisions_preserved") is True
        and selection.get("p1_full_output_stream_preserved") is True
        and selection.get("p1_reexecution_parity_passed") is True
        and selection.get("onnx_parity_passed") is True
        and float(selection.get("onnx_parity_maximum_absolute_error", 1.0)) <= 1e-5
        and selection.get("provider") == "CPUExecutionProvider"
        and selection.get("candidate_onnx_graph_optimization_level") == "ORT_DISABLE_ALL"
        and selection.get("source_commit") == P3_SOURCE_COMMIT
        and selection.get("onnx_sha256") == P3_ONNX_SHA256
        and selection.get("checkpoint_sha256") == P3_CHECKPOINT_SHA256
        and selection.get("report_sha256") == P3_REPORT_SHA256
        and selection.get("training_opened_seal_sha256") == P3_OPENED_SEAL_SHA256
        and selection.get("training_result_seal_sha256") == P3_RESULT_SEAL_SHA256
        and metrics.get("direct_stored_fixture_byte_execution") is True
        and metrics.get("scene_count") == metrics.get("exact_scene_count") == 128
        and metrics.get("true_positives") == metrics.get("truth_region_count") == 1024
        and metrics.get("false_positives") == metrics.get("false_negatives") == 0
        and metrics.get("duplicate_region_count") == 0
        and metrics.get("prohibited_structure_hits") == 0
        and metrics.get("recognition_exact") == 0.97265625
        and metrics.get("character_error_rate") == 0.004634994206257242
        and metrics.get("role_accuracy") == 1.0
        and isinstance(roles, dict)
        and set(roles) == {
            "YTick", "XTick", "AxisTitle", "PhaseHeading",
            "LegendText", "Participant", "Annotation", "Other",
        }
        and set(roles.values()) == {1.0}
        and len(comparisons) == len(THRESHOLDS)
        and all(
            item.get("threshold") == threshold
            and item.get("exact_scene_count") == 128
            and item.get("false_positives") == 0
            and item.get("false_negatives") == 0
            and item.get("duplicate_region_count") == 0
            and item.get("prohibited_structure_hits") == 0
            for item, threshold in zip(comparisons, THRESHOLDS, strict=True)
        )
        and selection.get("case_level_details_emitted") is False
        and selection.get("public_gate_archive_opened") is False
        and selection.get("public_gate_authorized") is False
        and selection.get("public_gate_evaluations") == 0
        and selection.get("private_validation_authorized") is False
        and selection.get("production_approval") is False
        and selection.get("release_eligible") is False
        and "cases" not in selection
        and "predictions" not in selection
    )


def _candidate_hashes() -> dict[str, str]:
    values = {
        "detector_onnx_sha256": sha256_file(REPO_ROOT / DETECTOR_PATH),
        "recognizer_onnx_sha256": sha256_file(REPO_ROOT / RECOGNIZER_PATH),
        "role_parent_onnx_sha256": sha256_file(REPO_ROOT / ROLE_PARENT_ONNX_PATH),
        "p1_onnx_sha256": sha256_file(REPO_ROOT / P1_ONNX_PATH),
        "candidate_onnx_sha256": sha256_file(REPO_ROOT / P3_ONNX_PATH),
        "candidate_checkpoint_sha256": sha256_file(REPO_ROOT / P3_CHECKPOINT_PATH),
        "selection_result_sha256": sha256_file(REPO_ROOT / P3_RESULT_PATH),
        "selection_report_sha256": sha256_file(REPO_ROOT / P3_REPORT_PATH),
        "selection_opened_seal_sha256": sha256_file(REPO_ROOT / P3_OPENED_SEAL_PATH),
        "selection_result_seal_sha256": sha256_file(REPO_ROOT / P3_RESULT_SEAL_PATH),
    }
    return values


def _gate_metrics_pass(metrics: dict[str, object]) -> bool:
    return bool(
        metrics_pass(metrics)
        and metrics.get("scene_count") == 192
        and metrics.get("direct_stored_fixture_byte_execution") is True
    )


def preflight() -> dict[str, Any]:
    require_committed_sources(
        REPO_ROOT,
        (LEDGER_PATH, PUBLIC_CONFIG_PATH, P3_RESULT_PATH, ROOT / "public_gate.py"),
    )
    if (REPO_ROOT / PUBLIC_OUTPUT_PATH).exists():
        raise RuntimeError("OCR V28 public gate output already exists")
    config = _read_json(REPO_ROOT / PUBLIC_CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-relational-neighborhood-public-gate-config.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "candidate_id": "P3",
        "evaluation_limit": 1,
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_sha256": PUBLIC_ARCHIVE_SHA256,
        "expected_dataset_manifest_sha256": PUBLIC_MANIFEST_SHA256,
        "split_seal_path": SPLIT_SEAL_PATH.as_posix(),
        "split_seal_sha256": SPLIT_SEAL_SHA256,
        "selection_result_path": P3_RESULT_PATH.as_posix(),
        "selection_result_sha256": P3_RESULT_SHA256,
        "selection_report_path": P3_REPORT_PATH.as_posix(),
        "selection_report_sha256": P3_REPORT_SHA256,
        "candidate_onnx_path": P3_ONNX_PATH.as_posix(),
        "candidate_onnx_sha256": P3_ONNX_SHA256,
        "candidate_checkpoint_path": P3_CHECKPOINT_PATH.as_posix(),
        "candidate_checkpoint_sha256": P3_CHECKPOINT_SHA256,
        "provider": "CPUExecutionProvider",
        "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
        "public_execution_authorized": True,
        "case_level_failure_analysis_permitted": False,
        "marker_creation_authorized": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V28 public gate configuration changed: {key}")
    if config.get("expected_candidate_hash_keys") != list(EXPECTED_CANDIDATE_HASH_KEYS):
        raise RuntimeError("OCR V28 public gate candidate hash schema changed")
    selection = _read_json(REPO_ROOT / P3_RESULT_PATH)
    if sha256_file(REPO_ROOT / P3_RESULT_PATH) != P3_RESULT_SHA256:
        raise RuntimeError("OCR V28 selected aggregate result changed")
    if not _selected_result_is_terminal(selection):
        raise RuntimeError("OCR V28 public gate requires the exact consumed P3 selection")
    if sha256_file(REPO_ROOT / P3_REPORT_PATH) != P3_REPORT_SHA256:
        raise RuntimeError("OCR V28 selected candidate report changed")
    candidate_report = _read_json(REPO_ROOT / P3_REPORT_PATH)
    if (
        candidate_report.get("status") != "selected"
        or candidate_report.get("selection_gate_passed") is not True
        or candidate_report.get("onnx_sha256") != P3_ONNX_SHA256
        or candidate_report.get("checkpoint_sha256") != P3_CHECKPOINT_SHA256
        or candidate_report.get("case_level_details_emitted") is not False
        or candidate_report.get("public_gate_archive_opened") is not False
        or "cases" in candidate_report
        or "predictions" in candidate_report
    ):
        raise RuntimeError("OCR V28 selected candidate report is not terminal")
    expected_hashes = dict(zip(EXPECTED_CANDIDATE_HASH_KEYS, (
        DETECTOR_SHA256,
        RECOGNIZER_SHA256,
        ROLE_PARENT_ONNX_SHA256,
        P1_ONNX_SHA256,
        P3_ONNX_SHA256,
        P3_CHECKPOINT_SHA256,
        P3_RESULT_SHA256,
        P3_REPORT_SHA256,
        P3_OPENED_SEAL_SHA256,
        P3_RESULT_SEAL_SHA256,
    ), strict=True))
    candidate_hashes = _candidate_hashes()
    if candidate_hashes != expected_hashes:
        raise RuntimeError("OCR V28 public gate candidate payloads changed")
    if sha256_file(REPO_ROOT / RECOGNIZER_YAML_PATH) != RECOGNIZER_YAML_SHA256:
        raise RuntimeError("OCR V28 recognizer preprocessing contract changed")
    if sha256_file(REPO_ROOT / SPLIT_SEAL_PATH) != SPLIT_SEAL_SHA256:
        raise RuntimeError("OCR V28 split seal changed")
    split_seal = _read_json(REPO_ROOT / SPLIT_SEAL_PATH)
    registered = split_seal["splits"]["sealed_public"]
    if (
        registered.get("archive_path") != PUBLIC_ARCHIVE_PATH.as_posix()
        or registered.get("archive_sha256") != PUBLIC_ARCHIVE_SHA256
        or registered.get("manifest_sha256") != PUBLIC_MANIFEST_SHA256
        or registered.get("proposal_summary", {}).get("scene_count") != 192
        or split_seal.get("public_execution_authorized") is not False
        or split_seal.get("public_evaluations") != 0
        or split_seal.get("chandler_used") is not False
        or split_seal.get("private_data") is not False
    ):
        raise RuntimeError("OCR V28 sealed public registration changed")
    head = _repository_head()
    runner_commit = str(config.get("runner_source_commit", ""))
    if not runner_commit or not _is_ancestor(runner_commit, head):
        raise RuntimeError("OCR V28 public runner source commit is not an ancestor")
    evaluator_bundle = source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    if config.get("expected_evaluator_source_bundle_sha256") != evaluator_bundle:
        raise RuntimeError("OCR V28 public evaluator source bundle changed")
    ledger = _read_json(REPO_ROOT / LEDGER_PATH)
    entry = next(
        (
            item for item in ledger["revisions"]
            if item.get("task") == TASK and item.get("revision") == REVISION
        ),
        None,
    )
    if (
        entry is None
        or entry.get("status") != "candidate_3_selected_public_gate_pending"
        or entry.get("consumed_candidate_ids") != ["P1", "P2", "P3"]
        or entry.get("selection_evaluations") != 3
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_authorized_candidate_id") != "P3"
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_archive_opened") is not False
        or entry.get("public_gate_config_path") != PUBLIC_CONFIG_PATH.as_posix()
        or entry.get("public_gate_config_sha256")
        != sha256_file(REPO_ROOT / PUBLIC_CONFIG_PATH)
        or entry.get("public_gate_runner_source_commit") != runner_commit
        or entry.get("public_gate_runner_source_bundle_sha256") != evaluator_bundle
        or entry.get("execution_authorized") is not False
        or entry.get("authorized_candidate_id") is not None
        or entry.get("marker_creation_evaluated") is not False
        or entry.get("private_validation") is not False
        or entry.get("production_approval") is not False
        or entry.get("release_eligible") is not False
    ):
        raise RuntimeError("OCR V28 public gate is not authorized by the canonical ledger")
    return {
        "candidate_hashes": candidate_hashes,
        "config": config,
        "head": head,
        "public_registration": registered,
        "public_window": _public_window(selection),
        "selection": selection,
        "split_seal": split_seal,
    }


def evaluate_public() -> dict[str, object]:
    evidence = preflight()
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes=evidence["candidate_hashes"],
        dataset_manifest_sha256=PUBLIC_MANIFEST_SHA256,
        split_config_path=PUBLIC_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    output_path = REPO_ROOT / PUBLIC_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phase = "open_frozen_public_archive_once"
    try:
        archive_path = REPO_ROOT / PUBLIC_ARCHIVE_PATH
        archive_payload = archive_path.read_bytes()
        if sha256(archive_payload).hexdigest() != PUBLIC_ARCHIVE_SHA256:
            raise RuntimeError("OCR V28 public archive changed")
        public_scenes = load_archive(BytesIO(archive_payload))  # type: ignore[arg-type]
        _validate_stored_split(
            public_scenes, evidence["public_registration"], "sealed_public",
        )

        phase = "direct_public_onnx_execution"
        detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        candidate_session = _candidate_session(REPO_ROOT / P3_ONNX_PATH)
        p1_session = _candidate_session(REPO_ROOT / P1_ONNX_PATH)
        detector_input = detector_session.get_inputs()[0].name
        recognizer_input = recognizer_session.get_inputs()[0].name
        alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)

        def detector_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(detector_session.run(None, {
                detector_input: np.ascontiguousarray(values),
            })[0], dtype=np.float32)

        def recognizer_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(recognizer_session.run(None, {
                recognizer_input: np.ascontiguousarray(values),
            })[0], dtype=np.float32)

        (
            values,
            crops,
            _,
            records,
            relations,
            scene_slices,
            runtime_evidence,
        ) = extract_relational_evidence(
            public_scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=10_000,
            recognition_batch_size=64,
        )
        registered_summary = evidence["public_registration"]["proposal_summary"]
        for key in ("scene_count", "proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if runtime_evidence.get(key) != registered_summary.get(key):
                raise RuntimeError(f"OCR V28 public proposal stream changed: {key}")

        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        relation_inputs = sha256()
        candidate_outputs = sha256()
        p1_outputs = sha256()
        p1_proposals = sha256()
        p3_proposals = sha256()
        proposal_logit_mismatches = 0
        threshold_acceptance_mismatches = 0
        for scene_index, scene_slice in enumerate(scene_slices):
            scene_values = np.ascontiguousarray(values[scene_slice][None, ...])
            scene_crops = np.ascontiguousarray(crops[scene_slice][None, ...])
            scene_relations = np.ascontiguousarray(relations[scene_index][None, ...])
            inputs = {
                "proposal_evidence": scene_values,
                "proposal_crops": scene_crops,
                "proposal_relations": scene_relations,
            }
            actual = np.asarray(candidate_session.run(None, inputs)[0], dtype=np.float32)
            actual_p1 = np.asarray(p1_session.run(None, inputs)[0], dtype=np.float32)
            if actual.shape != actual_p1.shape or actual.shape[2] != 10:
                raise RuntimeError("OCR V28 public candidate output contract changed")
            proposal_logit_mismatches += int(np.count_nonzero(
                actual[:, :, :2] != actual_p1[:, :, :2]
            ))
            p3_probability = _positive_probabilities(actual[0, :, :2])
            p1_probability = _positive_probabilities(actual_p1[0, :, :2])
            for threshold in evidence["public_window"]:
                threshold_acceptance_mismatches += int(np.count_nonzero(
                    (p3_probability >= threshold) != (p1_probability >= threshold)
                ))
            outputs.append(actual[0])
            evidence_inputs.update(scene_values.tobytes(order="C"))
            crop_inputs.update(scene_crops.tobytes(order="C"))
            relation_inputs.update(scene_relations.tobytes(order="C"))
            candidate_outputs.update(actual.tobytes(order="C"))
            p1_outputs.update(actual_p1.tobytes(order="C"))
            p1_proposals.update(
                np.ascontiguousarray(actual_p1[:, :, :2]).tobytes(order="C")
            )
            p3_proposals.update(
                np.ascontiguousarray(actual[:, :, :2]).tobytes(order="C")
            )
        if len(outputs) != len(public_scenes):
            raise RuntimeError("OCR V28 public inference call count changed")
        p1_proposal_hash = p1_proposals.hexdigest()
        p3_proposal_hash = p3_proposals.hexdigest()
        proposal_preserved = bool(
            proposal_logit_mismatches == 0
            and threshold_acceptance_mismatches == 0
            and p1_proposal_hash == p3_proposal_hash
        )
        runtime_evidence.update({
            "candidate_inference_calls": len(scene_slices),
            "p1_inference_calls": len(scene_slices),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_relation_input_tensor_stream_sha256": relation_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "p1_output_tensor_stream_sha256": p1_outputs.hexdigest(),
            "p1_proposal_output_tensor_stream_sha256": p1_proposal_hash,
            "p3_proposal_output_tensor_stream_sha256": p3_proposal_hash,
            "p1_proposal_logit_mismatch_count": proposal_logit_mismatches,
            "p1_threshold_acceptance_mismatch_count": threshold_acceptance_mismatches,
            "p1_proposal_decisions_preserved": proposal_preserved,
            "candidate_onnx_sha256": P3_ONNX_SHA256,
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
            "provider": "CPUExecutionProvider",
            "public_archive_read_count": 1,
        })
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(records, flat_output)
        comparisons = evaluate_thresholds(
            public_scenes,
            calibrated_records,
            flat_output[:, :2],
            evidence["public_window"],
            runtime_evidence,
        )
        passed = bool(
            proposal_preserved
            and len(comparisons) == ROBUST_THRESHOLD_RUN_LENGTH
            and all(_gate_metrics_pass(item["metrics"]) for item in comparisons)
        )
        selected_threshold = float(evidence["selection"]["selected_threshold"])
        selected_metrics = next(
            item["metrics"]
            for item in comparisons
            if item["threshold"] == selected_threshold
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-relational-neighborhood-public-gate.v1",
            "task": TASK,
            "revision": PUBLIC_REVISION,
            "candidate_id": "P3",
            "status": "pass" if passed else "fail",
            "evaluation_count": 1,
            "candidate_hashes": evidence["candidate_hashes"],
            "fixture_archive_sha256": PUBLIC_ARCHIVE_SHA256,
            "fixture_manifest_sha256": PUBLIC_MANIFEST_SHA256,
            "provider": "CPUExecutionProvider",
            "selected_threshold": selected_threshold,
            "public_threshold_window": list(evidence["public_window"]),
            "metrics": selected_metrics,
            "threshold_comparisons": comparisons,
            "gate_requirements": GATE_CONFIG,
            "seal_binding": gate.binding,
            "canonical_seal_key": gate.key,
            "direct_fixture_byte_execution": True,
            "public_archive_read_count": 1,
            "case_level_details_emitted": False,
            "marker_creation_evaluated": False,
            "marker_creation_gate_required_before_production_approval": True,
            "private_validation_authorized": False,
            "production_approval": False,
            "release_eligible": False,
            "synthetic_only": True,
            "chandler_included": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        output_path.write_bytes(canonical_json_bytes(report))
        complete_gate_seal(
            gate, status=str(report["status"]), report_sha256=sha256_file(output_path),
        )
        return report
    except Exception as error:
        failure: dict[str, object] = {
            "schema": "graphreader.ocr-relational-neighborhood-public-gate-failure.v1",
            "task": TASK,
            "revision": PUBLIC_REVISION,
            "candidate_id": "P3",
            "status": "failed_runner",
            "evaluation_count": 1,
            "phase": phase,
            "exception_type": type(error).__name__,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_hashes": evidence["candidate_hashes"],
            "seal_binding": gate.binding,
            "canonical_seal_key": gate.key,
            "case_level_details_emitted": False,
            "marker_creation_evaluated": False,
            "private_validation_authorized": False,
            "production_approval": False,
            "release_eligible": False,
        }
        output_path.write_bytes(canonical_json_bytes(failure))
        complete_gate_seal(
            gate, status="failed_runner", report_sha256=sha256_file(output_path),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        evidence = preflight()
        print(json.dumps({
            "candidate_id": "P3",
            "head": evidence["head"],
            "public_window": list(evidence["public_window"]),
            "ready": True,
        }, sort_keys=True))
        return 0
    report = evaluate_public()
    print(json.dumps({
        "status": report["status"],
        "evaluation_count": report["evaluation_count"],
        "selected_threshold": report["selected_threshold"],
        "public_threshold_window": report["public_threshold_window"],
        "metrics": report["metrics"],
        "case_level_details_emitted": False,
        "production_approval": False,
        "release_eligible": False,
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATOR_SOURCE_PATHS",
    "EXPECTED_CANDIDATE_HASH_KEYS",
    "GATE_CONFIG",
    "PUBLIC_CONFIG_PATH",
    "PUBLIC_OUTPUT_PATH",
    "PUBLIC_REVISION",
    "_gate_metrics_pass",
    "_public_window",
    "_selected_result_is_terminal",
    "evaluate_public",
    "preflight",
]
