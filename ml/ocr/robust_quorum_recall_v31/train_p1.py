# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-training robust quorum selection for OCR V31 P1."""

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
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import (
    acquire_training_candidate,
    complete_training_candidate,
    void_candidate,
)
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _calibrated_records,
    _cpu_session,
    _is_ancestor,
    _read_json,
    _repository_head,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from ml.ocr.relational_neighborhood_proposal_v28.train_p1 import (
    _candidate_session,
    _export,
)

from .dataset import load_archive, proposal_summary, split_fingerprint
from .model import RobustQuorumRecallProposalNet
from .pipeline import extract_relational_evidence
from .prepare_split import SOURCE_PATHS
from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    DETECTOR_PATH,
    DETECTOR_SHA256,
    FEATURE_COUNT,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    RELATION_FEATURE_COUNT,
    REVISION,
    SEED,
    TASK,
    THRESHOLDS,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    V30_CHECKPOINT_PATH,
    V30_CHECKPOINT_SHA256,
    V30_ONNX_PATH,
    V30_ONNX_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/robust_quorum_recall_v31")
CANDIDATE_ID = "P1"
CONFIG_PATH = ROOT / "training/p1.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v31-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v31-public.zip")
RUNNER_SOURCE_PATHS = SOURCE_PATHS


def _validate_stored_split(
    scenes: tuple[Any, ...], registered: dict[str, Any], name: str,
) -> None:
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != registered["split_fingerprint"]
        or any(summary[key] != registered["proposal_summary"][key] for key in summary)
    ):
        raise RuntimeError(f"OCR V31 {name} stored fixtures violate the seal")


def _trigger_is_terminal(trigger: dict[str, Any]) -> bool:
    metrics = trigger.get("metrics", {})
    comparisons = trigger.get("threshold_comparisons", [])
    return bool(
        trigger.get("schema")
        == "graphreader.ocr-unanimous-structure-veto-public-result.v1"
        and trigger.get("revision") == "graph-text-unanimous-structure-veto-v30"
        and trigger.get("status") == "failed_public_gate"
        and trigger.get("candidate_consumed") is True
        and trigger.get("public_gate_passed") is False
        and trigger.get("public_archive_consumed") is True
        and trigger.get("public_archive_read_count") == 1
        and trigger.get("case_level_failure_analysis_performed") is False
        and trigger.get("next_revision_may_reuse_public_bytes") is False
        and trigger.get("public_failure_tuning_authorized") is False
        and metrics.get("scene_count") == 256
        and metrics.get("exact_scene_count") == 255
        and metrics.get("truth_region_count") == 2048
        and metrics.get("true_positives") == 2047
        and metrics.get("false_positives") == 0
        and metrics.get("false_negatives") == 1
        and metrics.get("duplicate_region_count") == 0
        and metrics.get("prohibited_structure_hits") == 0
        and len(comparisons) == 3
        and "cases" not in trigger
        and "predictions" not in trigger
        and "truths" not in trigger
    )


def preflight(*, require_authorized: bool = True) -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-robust-quorum-recall-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "two-of-three-robust-route-quorum-v1",
        "objective": "zero-training-two-of-three-route-quorum-v1",
        "model_license": "Apache-2.0",
        "seed": SEED,
        "expected_optimizer_steps": 0,
        "predecessor_checkpoint_reused": True,
        "feature_count": FEATURE_COUNT,
        "crop_channels": CROP_CHANNELS,
        "crop_height": CROP_HEIGHT,
        "crop_width": CROP_WIDTH,
        "relation_feature_count": RELATION_FEATURE_COUNT,
        "runtime_numeric_precision": "float32",
        "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
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
        "selection_thresholds": list(THRESHOLDS),
        "split_seal_path": SEAL_PATH.as_posix(),
        "selection_fixture_archive_path": SELECTION_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "selection_evaluation_limit": 1,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "marker_creation_evaluated": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V31 P1 config field mismatch: {key}")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V31 P1 runner source bundle changed")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V31 split seal changed before P1")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    required_seal = {
        "schema": "graphreader.ocr-robust-quorum-recall-split-seal.v1",
        "revision": REVISION,
        "optimizer_steps_at_freeze": 0,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "candidate_execution_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "private_data": False,
        "chandler_used": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in required_seal.items():
        if seal.get(key) != value:
            raise RuntimeError(f"OCR V31 split seal field changed: {key}")
    head = _repository_head()
    if not _is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V31 split source commit is not an ancestor")
    if seal.get("source_bundle_sha256") != config.get("split_source_bundle_sha256"):
        raise RuntimeError("OCR V31 split source bundle changed")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V31 split source inventory is absent")
    for relative, expected_hash in sealed_sources.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V31 frozen split source changed: {relative}")
    for split, (path, key) in {
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(key):
            raise RuntimeError(f"OCR V31 {split} archive changed before P1")
    for relative, expected_hash in {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        V30_CHECKPOINT_PATH: V30_CHECKPOINT_SHA256,
        V30_ONNX_PATH: V30_ONNX_SHA256,
        TRIGGER_RESULT_PATH: TRIGGER_RESULT_SHA256,
    }.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V31 frozen input changed: {relative}")
    if not _trigger_is_terminal(_read_json(REPO_ROOT / TRIGGER_RESULT_PATH)):
        raise RuntimeError("OCR V31 aggregate-only V30 trigger changed")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V31 P1 output already exists")
    ledger = _read_json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(
        (
            item for item in ledger.get("revisions", [])
            if item.get("task") == TASK and item.get("revision") == REVISION
        ),
        None,
    )
    if entry is None or entry.get("status") != "candidate_1_preregistered":
        raise RuntimeError("OCR V31 P1 ledger state is not preregistered")
    if require_authorized and (
        entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or config.get("candidate_execution_authorized") is not True
    ):
        raise RuntimeError("OCR V31 P1 execution is not separately authorized")
    if not require_authorized and config.get("candidate_execution_authorized") not in (
        False, True,
    ):
        raise RuntimeError("OCR V31 P1 authorization field is invalid")
    return {"config": config, "seal": seal, "entry": entry, "head": head}


def evaluate_candidate(output_dir: Path) -> dict[str, object]:
    evidence = preflight(require_authorized=True)
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    started = time.perf_counter()
    phase = "initialize"
    report_path = output_dir / "candidate-report.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        config = evidence["config"]
        phase = "load_exact_v30_routes"
        payload = torch.load(
            REPO_ROOT / V30_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
        state = payload.get("state_dict")
        if not isinstance(state, dict) or not state:
            raise RuntimeError("OCR V31 exact V30 route state is missing")
        model = RobustQuorumRecallProposalNet(seed=int(config["seed"]))
        model.load_state_dict(state, strict=True)
        model.eval()
        optimizer_steps = 0
        checkpoint_path = output_dir / "graph-text-robust-quorum-recall-v31-p1.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)

        phase = "single_visible_selection"
        archive_payload = (REPO_ROOT / SELECTION_ARCHIVE_PATH).read_bytes()
        if sha256(archive_payload).hexdigest() != config["selection_fixture_archive_sha256"]:
            raise RuntimeError("OCR V31 selection archive changed during read")
        selection_scenes = load_archive(BytesIO(archive_payload))
        _validate_stored_split(
            selection_scenes,
            evidence["seal"]["splits"]["validation"],
            "validation",
        )
        detector_runner = _cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_runner = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)
        (
            selection_values,
            selection_crops,
            _,
            selection_records,
            selection_relations,
            selection_slices,
            selection_evidence,
        ) = extract_relational_evidence(
            selection_scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=10000,
            recognition_batch_size=64,
        )
        registered = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V31 incomplete selection stream: {key}")
        first = selection_slices[0]
        onnx_path = output_dir / "graph-text-robust-quorum-recall-v31-p1.onnx"
        _export(
            model,
            torch.from_numpy(selection_values[first]).unsqueeze(0),
            torch.from_numpy(selection_crops[first]).unsqueeze(0),
            torch.from_numpy(selection_relations[0]).unsqueeze(0),
            onnx_path,
        )
        candidate_session = _candidate_session(onnx_path)
        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        relation_inputs = sha256()
        candidate_outputs = sha256()
        attention_outputs = sha256()
        summary_outputs = sha256()
        local_outputs = sha256()
        parity_error = 0.0
        role_mismatches = 0
        with torch.inference_mode():
            for scene_index, scene_slice in enumerate(selection_slices):
                values = np.ascontiguousarray(selection_values[scene_slice][None, ...])
                crops = np.ascontiguousarray(selection_crops[scene_slice][None, ...])
                relations = np.ascontiguousarray(selection_relations[scene_index][None, ...])
                torch_values = torch.from_numpy(values)
                torch_crops = torch.from_numpy(crops)
                torch_relations = torch.from_numpy(relations)
                expected_output = model(
                    torch_values, torch_crops, torch_relations,
                ).numpy()
                consensus, attention, summary, local_route = model.proposal_routes(
                    torch_values, torch_crops, torch_relations,
                )
                actual_output = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                    "proposal_relations": relations,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
                )
                expected_roles = np.argmax(model.role_logits(torch_values).numpy(), axis=2)
                role_mismatches += int(np.count_nonzero(
                    np.argmax(actual_output[:, :, 2:], axis=2) != expected_roles,
                ))
                outputs.append(actual_output[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                relation_inputs.update(relations.tobytes(order="C"))
                candidate_outputs.update(actual_output.tobytes(order="C"))
                attention_outputs.update(attention.numpy().tobytes(order="C"))
                summary_outputs.update(summary.numpy().tobytes(order="C"))
                local_outputs.update(local_route.numpy().tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(selection_records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(selection_slices),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_relation_input_tensor_stream_sha256": relation_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "attention_route_output_tensor_stream_sha256": attention_outputs.hexdigest(),
            "summary_route_output_tensor_stream_sha256": summary_outputs.hexdigest(),
            "local_route_output_tensor_stream_sha256": local_outputs.hexdigest(),
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
            "deterministic_role_mismatch_count": role_mismatches,
            "selection_archive_read_count": 1,
            "direct_stored_fixture_byte_execution": True,
        })
        comparisons = evaluate_thresholds(
            selection_scenes,
            calibrated_records,
            flat_output[:, :2],
            tuple(float(value) for value in config["selection_thresholds"]),
            selection_evidence,
        )
        robust = select_robust_window(comparisons)
        selected = robust[0] if robust else max(
            comparisons,
            key=lambda item: (
                item["metrics"]["exact_scene_count"],
                -item["metrics"]["false_positives"],
                -item["metrics"]["false_negatives"],
                item["metrics"]["recognition_exact"],
                item["metrics"]["role_accuracy"],
            ),
        )
        window = robust[1] if robust else ()
        parity_passed = parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        roles_exact = role_mismatches == 0
        passed = robust is not None and parity_passed and roles_exact
        report: dict[str, object] = {
            "schema": "graphreader.ocr-robust-quorum-recall-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": optimizer_steps,
            "weights_changed": False,
            "predecessor_checkpoint_reused": True,
            "predecessor_checkpoint_sha256": V30_CHECKPOINT_SHA256,
            "objective": config["objective"],
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": parity_error,
            "onnx_parity_passed": parity_passed,
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
            "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "passing_threshold_window": list(window),
            "selected_threshold": selected["threshold"],
            "selection_metrics": selected["metrics"],
            "selection_evidence": selection_evidence,
            "selection_archive_read_count": 1,
            "training_authorization": authorization.binding,
            "split_seal_sha256": config["split_seal_sha256"],
            "selection_fixture_archive_sha256": config[
                "selection_fixture_archive_sha256"
            ],
            "case_level_details_emitted": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "marker_creation_evaluated": False,
            "private_validation_authorized": False,
            "manifest_creation_authorized": False,
            "model_store_promotion_authorized": False,
            "production_approval": False,
            "release_eligible": False,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(
            authorization,
            status=str(report["status"]),
            report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-robust-quorum-recall-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": 0,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "training_authorization": authorization.binding,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "marker_creation_evaluated": False,
            "production_approval": False,
            "release_eligible": False,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        void_candidate(authorization, error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        evidence = preflight(require_authorized=True)
        print(json.dumps({
            "head": evidence["head"],
            "runner_source_bundle_sha256": source_bundle_sha256(
                REPO_ROOT, RUNNER_SOURCE_PATHS,
            ),
            "ready": True,
        }, sort_keys=True))
        return 0
    report = evaluate_candidate(REPO_ROOT / CANONICAL_OUTPUT)
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report["selected_threshold"],
        "passing_threshold_window": report["passing_threshold_window"],
        "onnx_parity_maximum_absolute_error": report[
            "onnx_parity_maximum_absolute_error"
        ],
        "selection_metrics": report["selection_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT", "CONFIG_PATH", "RUNNER_SOURCE_PATHS",
    "_trigger_is_terminal", "evaluate_candidate", "preflight",
]
