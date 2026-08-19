# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for the selected OCR V29 P1 payload."""

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
    ROLE_PARENT_CHECKPOINT_PATH,
    ROLE_PARENT_CHECKPOINT_SHA256,
    ROLE_PARENT_ONNX_PATH,
    ROLE_PARENT_ONNX_SHA256,
    TASK,
    THRESHOLDS,
)
from .train_p1 import (
    RUNNER_SOURCE_PATHS as P1_RUNNER_SOURCE_PATHS,
    _candidate_session,
    _validate_stored_split,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/dual_route_consensus_proposal_v29")
LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")
SPLIT_SEAL_PATH = ROOT / "SPLIT_SEAL.json"
SPLIT_SEAL_SHA256 = "e0f30b2d57c3c22e6e1121e71248af4c6ff65e48d4781e815f4cf1f6bcdf41bf"
PUBLIC_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
PUBLIC_OUTPUT_PATH = ROOT / "artifacts/public-gate-v1/report.json"
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v29-public.zip")
PUBLIC_ARCHIVE_SHA256 = "b642bd38d7e30e0a650fde87c7f99e41443ef9d4566880421deeb461f04ae7e9"
PUBLIC_MANIFEST_SHA256 = "c8ec15b43dd42c0ba8da0e9fc3aef7217d13582931bf8ee2fad9bbd00e01002a"
PUBLIC_REVISION = "graph-text-dual-route-consensus-proposal-v29-public-v1"

P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P1_RESULT_SHA256 = "16fc3cab1c357a81bab2bdd0ae2f44b50f4a38a6c2d4e5bab688d9e9f4a2396e"
P1_REPORT_PATH = ROOT / "artifacts/P1-run/candidate-report.json"
P1_REPORT_SHA256 = "49b7cd6d2645d7e5bda5c787a0bad9547cb2a244dc694c4ec77a17266672ee4b"
P1_CHECKPOINT_PATH = (
    ROOT / "artifacts/P1-run/graph-text-dual-route-consensus-proposal-v29-p1.pt"
)
P1_CHECKPOINT_SHA256 = "1dd8fc613815402fdad389f38652dbf75e35f8e25ed2f56dd06f06be6196336f"
P1_ONNX_PATH = (
    ROOT / "artifacts/P1-run/graph-text-dual-route-consensus-proposal-v29-p1.onnx"
)
P1_ONNX_SHA256 = "a1ce725897f44d43a6db0852638abb3787c9be917bba0d412f0b1a798831f223"
P1_OPENED_SEAL_PATH = (
    Path("ml/markers/training-seals/ocr-detection-recognition")
    / REVISION / "P1/opened.json"
)
P1_OPENED_SEAL_SHA256 = "8e5a1194cbfbb4422da6f428b90e7ba0e0b6bd3b680fda11c641b6e76ef01bcc"
P1_RESULT_SEAL_PATH = (
    Path("ml/markers/training-seals/ocr-detection-recognition")
    / REVISION / "P1/result.json"
)
P1_RESULT_SEAL_SHA256 = "25f4407719abc06fd32d160f410af71575a7095dd859173f00b78793e59f21f8"

EVALUATOR_SOURCE_PATHS = tuple(dict.fromkeys((
    *P1_RUNNER_SOURCE_PATHS,
    P1_RESULT_PATH,
    ROOT / "public_gate.py",
)))
EXPECTED_CANDIDATE_HASH_KEYS = (
    "detector_onnx_sha256",
    "recognizer_onnx_sha256",
    "role_parent_checkpoint_sha256",
    "role_parent_onnx_sha256",
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
    "direct_fixture_byte_execution_required": True,
    "complete_production_proposal_stream_required": True,
    "detector_recognizer_candidate_and_relation_tensor_hashes_required": True,
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
        raise RuntimeError("OCR V29 selection has no preregistered robust threshold window")
    selected_index = window.index(selected)
    lower = max(0, selected_index - 1)
    result = window[lower:lower + ROBUST_THRESHOLD_RUN_LENGTH]
    if len(result) < ROBUST_THRESHOLD_RUN_LENGTH:
        result = window[-ROBUST_THRESHOLD_RUN_LENGTH:]
    return result


def _selected_result_is_terminal(selection: dict[str, Any]) -> bool:
    metrics = selection.get("selection_metrics", {})
    roles = metrics.get("per_role_accuracy", {})
    return bool(
        selection.get("schema")
        == "graphreader.ocr-dual-route-consensus-selection-result.v1"
        and selection.get("task") == TASK
        and selection.get("revision") == REVISION
        and selection.get("candidate_id") == "P1"
        and selection.get("status") == "selected"
        and selection.get("candidate_consumed") is True
        and selection.get("selection_gate_passed") is True
        and selection.get("optimizer_steps") == 1280
        and selection.get("source_bundle_sha256")
        == "2834392f7484fc9882bdef28a30e4cdb331847376b5a8c0781951af56cc5ee99"
        and selection.get("split_seal_sha256") == SPLIT_SEAL_SHA256
        and selection.get("onnx_sha256") == P1_ONNX_SHA256
        and selection.get("checkpoint_sha256") == P1_CHECKPOINT_SHA256
        and selection.get("report_sha256") == P1_REPORT_SHA256
        and selection.get("training_opened_seal_sha256") == P1_OPENED_SEAL_SHA256
        and selection.get("training_result_seal_sha256") == P1_RESULT_SEAL_SHA256
        and selection.get("onnx_parity_passed") is True
        and float(selection.get("onnx_parity_maximum_absolute_error", 1.0)) <= 1e-5
        and selection.get("provider") == "CPUExecutionProvider"
        and selection.get("candidate_onnx_graph_optimization_level") == "ORT_DISABLE_ALL"
        and selection.get("frozen_v24_role_parent_preserved") is True
        and metrics.get("direct_stored_fixture_byte_execution") is True
        and metrics.get("scene_count") == metrics.get("exact_scene_count") == 160
        and metrics.get("true_positives") == metrics.get("truth_region_count") == 1280
        and metrics.get("false_positives") == metrics.get("false_negatives") == 0
        and metrics.get("duplicate_region_count") == 0
        and metrics.get("prohibited_structure_hits") == 0
        and metrics.get("recognition_exact") == 0.97109375
        and metrics.get("character_error_rate") == 0.004912373871481678
        and metrics.get("role_accuracy") == 1.0
        and metrics.get("deterministic_role_mismatch_count") == 0
        and isinstance(roles, dict)
        and set(roles) == {
            "YTick", "XTick", "AxisTitle", "PhaseHeading",
            "LegendText", "Participant", "Annotation", "Other",
        }
        and set(roles.values()) == {1.0}
        and selection.get("passing_threshold_window") == list(THRESHOLDS)
        and selection.get("selected_threshold") == 0.55
        and selection.get("case_level_details_emitted") is False
        and selection.get("public_gate_archive_opened") is False
        and selection.get("public_gate_authorized") is False
        and selection.get("public_gate_evaluations") == 0
        and selection.get("marker_creation_evaluated") is False
        and selection.get("private_validation_authorized") is False
        and selection.get("production_approval") is False
        and selection.get("release_eligible") is False
        and not any(key in selection for key in (
            "cases", "predictions", "truths", "fixture_bytes", "scene_ids",
        ))
    )


def _candidate_hashes() -> dict[str, str]:
    return {
        "detector_onnx_sha256": sha256_file(REPO_ROOT / DETECTOR_PATH),
        "recognizer_onnx_sha256": sha256_file(REPO_ROOT / RECOGNIZER_PATH),
        "role_parent_checkpoint_sha256": sha256_file(
            REPO_ROOT / ROLE_PARENT_CHECKPOINT_PATH
        ),
        "role_parent_onnx_sha256": sha256_file(REPO_ROOT / ROLE_PARENT_ONNX_PATH),
        "candidate_onnx_sha256": sha256_file(REPO_ROOT / P1_ONNX_PATH),
        "candidate_checkpoint_sha256": sha256_file(REPO_ROOT / P1_CHECKPOINT_PATH),
        "selection_result_sha256": sha256_file(REPO_ROOT / P1_RESULT_PATH),
        "selection_report_sha256": sha256_file(REPO_ROOT / P1_REPORT_PATH),
        "selection_opened_seal_sha256": sha256_file(
            REPO_ROOT / P1_OPENED_SEAL_PATH
        ),
        "selection_result_seal_sha256": sha256_file(
            REPO_ROOT / P1_RESULT_SEAL_PATH
        ),
    }


def _gate_metrics_pass(metrics: dict[str, object]) -> bool:
    return bool(
        metrics_pass(metrics)
        and metrics.get("scene_count") == 224
        and metrics.get("direct_stored_fixture_byte_execution") is True
    )


def _validate_config(config: dict[str, object], *, require_authorized: bool) -> None:
    expected = {
        "schema": "graphreader.ocr-dual-route-consensus-public-gate-config.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "candidate_id": "P1",
        "evaluation_limit": 1,
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_sha256": PUBLIC_ARCHIVE_SHA256,
        "expected_dataset_manifest_sha256": PUBLIC_MANIFEST_SHA256,
        "split_seal_path": SPLIT_SEAL_PATH.as_posix(),
        "split_seal_sha256": SPLIT_SEAL_SHA256,
        "selection_result_path": P1_RESULT_PATH.as_posix(),
        "selection_result_sha256": P1_RESULT_SHA256,
        "selection_report_path": P1_REPORT_PATH.as_posix(),
        "selection_report_sha256": P1_REPORT_SHA256,
        "candidate_onnx_path": P1_ONNX_PATH.as_posix(),
        "candidate_onnx_sha256": P1_ONNX_SHA256,
        "candidate_checkpoint_path": P1_CHECKPOINT_PATH.as_posix(),
        "candidate_checkpoint_sha256": P1_CHECKPOINT_SHA256,
        "provider": "CPUExecutionProvider",
        "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
        "case_level_failure_analysis_permitted": False,
        "marker_creation_authorized": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V29 public gate configuration changed: {key}")
    if config.get("expected_candidate_hash_keys") != list(EXPECTED_CANDIDATE_HASH_KEYS):
        raise RuntimeError("OCR V29 public gate candidate hash schema changed")
    expected_gate_hash = sha256(canonical_json_bytes(dict(GATE_CONFIG))).hexdigest()
    if config.get("expected_gate_config_sha256") != expected_gate_hash:
        raise RuntimeError("OCR V29 public gate metric contract changed")
    if require_authorized and config.get("public_execution_authorized") is not True:
        raise RuntimeError("OCR V29 public gate is not separately authorized")


def preflight() -> dict[str, Any]:
    require_committed_sources(
        REPO_ROOT,
        (LEDGER_PATH, PUBLIC_CONFIG_PATH, P1_RESULT_PATH, ROOT / "public_gate.py"),
    )
    if (REPO_ROOT / PUBLIC_OUTPUT_PATH).exists():
        raise RuntimeError("OCR V29 public gate output already exists")
    config = _read_json(REPO_ROOT / PUBLIC_CONFIG_PATH)
    _validate_config(config, require_authorized=True)
    selection = _read_json(REPO_ROOT / P1_RESULT_PATH)
    if sha256_file(REPO_ROOT / P1_RESULT_PATH) != P1_RESULT_SHA256:
        raise RuntimeError("OCR V29 selected aggregate result changed")
    if not _selected_result_is_terminal(selection):
        raise RuntimeError("OCR V29 public gate requires the exact consumed P1 selection")
    if sha256_file(REPO_ROOT / P1_REPORT_PATH) != P1_REPORT_SHA256:
        raise RuntimeError("OCR V29 selected candidate report changed")
    candidate_report = _read_json(REPO_ROOT / P1_REPORT_PATH)
    report_comparisons = candidate_report.get("threshold_comparisons", [])
    if (
        candidate_report.get("status") != "selected"
        or candidate_report.get("selection_gate_passed") is not True
        or candidate_report.get("onnx_sha256") != P1_ONNX_SHA256
        or candidate_report.get("checkpoint_sha256") != P1_CHECKPOINT_SHA256
        or candidate_report.get("case_level_details_emitted") is not False
        or candidate_report.get("public_gate_archive_opened") is not False
        or len(report_comparisons) != len(THRESHOLDS)
        or any(
            item.get("threshold") != threshold
            or item.get("metrics", {}).get("exact_scene_count") != 160
            or item.get("metrics", {}).get("false_positives") != 0
            or item.get("metrics", {}).get("false_negatives") != 0
            or item.get("metrics", {}).get("duplicate_region_count") != 0
            or item.get("metrics", {}).get("prohibited_structure_hits") != 0
            for item, threshold in zip(report_comparisons, THRESHOLDS, strict=True)
        )
        or "cases" in candidate_report
        or "predictions" in candidate_report
    ):
        raise RuntimeError("OCR V29 selected candidate report is not terminal")
    expected_hashes = dict(zip(EXPECTED_CANDIDATE_HASH_KEYS, (
        DETECTOR_SHA256,
        RECOGNIZER_SHA256,
        ROLE_PARENT_CHECKPOINT_SHA256,
        ROLE_PARENT_ONNX_SHA256,
        P1_ONNX_SHA256,
        P1_CHECKPOINT_SHA256,
        P1_RESULT_SHA256,
        P1_REPORT_SHA256,
        P1_OPENED_SEAL_SHA256,
        P1_RESULT_SEAL_SHA256,
    ), strict=True))
    candidate_hashes = _candidate_hashes()
    if candidate_hashes != expected_hashes:
        raise RuntimeError("OCR V29 public gate candidate payloads changed")
    if sha256_file(REPO_ROOT / RECOGNIZER_YAML_PATH) != RECOGNIZER_YAML_SHA256:
        raise RuntimeError("OCR V29 recognizer preprocessing contract changed")
    if sha256_file(REPO_ROOT / SPLIT_SEAL_PATH) != SPLIT_SEAL_SHA256:
        raise RuntimeError("OCR V29 split seal changed")
    split_seal = _read_json(REPO_ROOT / SPLIT_SEAL_PATH)
    registered = split_seal["splits"]["sealed_public"]
    summary = registered.get("proposal_summary", {})
    if (
        registered.get("archive_path") != PUBLIC_ARCHIVE_PATH.as_posix()
        or registered.get("archive_sha256") != PUBLIC_ARCHIVE_SHA256
        or registered.get("manifest_sha256") != PUBLIC_MANIFEST_SHA256
        or summary.get("scene_count") != 224
        or summary.get("proposal_count") != 7392
        or summary.get("positive_proposal_count") != 1792
        or summary.get("negative_proposal_count") != 5600
        or split_seal.get("public_execution_authorized") is not False
        or split_seal.get("public_evaluations") != 0
        or split_seal.get("chandler_used") is not False
        or split_seal.get("private_data") is not False
    ):
        raise RuntimeError("OCR V29 sealed public registration changed")
    head = _repository_head()
    runner_commit = str(config.get("runner_source_commit", ""))
    if not runner_commit or not _is_ancestor(runner_commit, head):
        raise RuntimeError("OCR V29 public runner source commit is not an ancestor")
    evaluator_bundle = source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    if config.get("expected_evaluator_source_bundle_sha256") != evaluator_bundle:
        raise RuntimeError("OCR V29 public evaluator source bundle changed")
    ledger = _read_json(REPO_ROOT / LEDGER_PATH)
    entry = next((
        item for item in ledger["revisions"]
        if item.get("task") == TASK and item.get("revision") == REVISION
    ), None)
    if (
        entry is None
        or entry.get("status") != "candidate_1_selected_public_gate_pending"
        or entry.get("consumed_candidate_ids") != ["P1"]
        or entry.get("candidate_1_selection_evaluations") != 1
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_authorized_candidate_id") != "P1"
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
        raise RuntimeError("OCR V29 public gate is not authorized by the canonical ledger")
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
            raise RuntimeError("OCR V29 public archive changed")
        public_scenes = load_archive(BytesIO(archive_payload))  # type: ignore[arg-type]
        _validate_stored_split(
            public_scenes, evidence["public_registration"], "sealed_public",
        )

        phase = "direct_public_onnx_execution"
        detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        candidate_session = _candidate_session(REPO_ROOT / P1_ONNX_PATH)
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
        for key in (
            "scene_count", "proposal_count",
            "positive_proposal_count", "negative_proposal_count",
        ):
            if runtime_evidence.get(key) != registered_summary.get(key):
                raise RuntimeError(f"OCR V29 public proposal stream changed: {key}")
        runtime_evidence.pop("proposal_relation_scene_shapes", None)

        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        relation_inputs = sha256()
        candidate_outputs = sha256()
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
            if actual.shape != (1, scene_values.shape[1], 10):
                raise RuntimeError("OCR V29 public candidate output contract changed")
            outputs.append(actual[0])
            evidence_inputs.update(scene_values.tobytes(order="C"))
            crop_inputs.update(scene_crops.tobytes(order="C"))
            relation_inputs.update(scene_relations.tobytes(order="C"))
            candidate_outputs.update(actual.tobytes(order="C"))
        if len(outputs) != len(public_scenes):
            raise RuntimeError("OCR V29 public inference call count changed")
        runtime_evidence.update({
            "candidate_inference_calls": len(scene_slices),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_relation_input_tensor_stream_sha256": relation_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "candidate_onnx_sha256": P1_ONNX_SHA256,
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
            len(comparisons) == ROBUST_THRESHOLD_RUN_LENGTH
            and all(_gate_metrics_pass(item["metrics"]) for item in comparisons)
        )
        selected_threshold = float(evidence["selection"]["selected_threshold"])
        selected_metrics = next(
            item["metrics"]
            for item in comparisons
            if item["threshold"] == selected_threshold
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-dual-route-consensus-public-gate.v1",
            "task": TASK,
            "revision": PUBLIC_REVISION,
            "candidate_id": "P1",
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
            "schema": "graphreader.ocr-dual-route-consensus-public-gate-failure.v1",
            "task": TASK,
            "revision": PUBLIC_REVISION,
            "candidate_id": "P1",
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
            "candidate_id": "P1",
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
    "_validate_config",
    "evaluate_public",
    "preflight",
]
