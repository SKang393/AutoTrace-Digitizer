# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer parent-recall residual-veto candidate for V24 P3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive
from .model_p2 import FrozenRoleAnchorCropResidualNet
from .model_p3 import ParentRecallResidualVetoNet
from .pipeline import extract_crop_evidence
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    TASK,
    THRESHOLDS,
)
from .train_p1 import (
    _calibrated_records,
    _cpu_session,
    _export,
    _feature_groups,
    _is_ancestor,
    _read_json,
    _repository_head,
    _validate_stored_split,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/crop_evidence_role_anchor_v24")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
P2_REPORT_PATH = ROOT / "artifacts/P2-run/candidate-report.json"
P2_CHECKPOINT_PATH = ROOT / "artifacts/P2-run/graph-text-crop-evidence-role-anchor-v24-p2.pt"
P2_ONNX_PATH = ROOT / "artifacts/P2-run/graph-text-crop-evidence-role-anchor-v24-p2.onnx"
PARENT_RESULT_PATH = Path("ml/ocr/role_anchor_set_v23/P3_RESULT.json")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-public.zip")
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "model_p2.py",
    ROOT / "model_p3.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p1.py",
    ROOT / "train_p3.py",
    P2_RESULT_PATH,
    PARENT_RESULT_PATH,
    Path("ml/ocr/role_anchor_set_v23/model.py"),
    Path("ml/ocr/role_anchor_set_v23/protocol.py"),
    Path("ml/ocr/margin_calibrator_v20/dataset.py"),
    Path("ml/ocr/margin_calibrator_v20/pipeline.py"),
    Path("ml/ocr/margin_calibrator_v20/protocol.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/dataset.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/protocol.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _load_p2_model(config: dict[str, Any]) -> FrozenRoleAnchorCropResidualNet:
    checkpoint = torch.load(
        REPO_ROOT / P2_CHECKPOINT_PATH, map_location="cpu", weights_only=True,
    )
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("OCR V24 P2 checkpoint has no state_dict")
    model = FrozenRoleAnchorCropResidualNet(
        residual_scale=float(config["p2_crop_residual_scale"]),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-crop-evidence-role-anchor-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "p2-parent-recall-crop-residual-veto-v1",
        "objective": "zero-optimizer-parent-recall-crop-residual-veto-v1",
        "model_license": "Apache-2.0",
        "optimizer_steps": 0,
        "expected_optimizer_steps": 0,
        "weights_changed": False,
        "parent_probability_minimum": 0.35,
        "crop_residual_margin_minimum": -0.25,
        "accepted_logit_magnitude": 8.0,
        "recognition_batch_size": 64,
        "complete_proposal_negative_cap_per_scene": 100000,
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "p2_result_path": P2_RESULT_PATH.as_posix(),
        "p2_crop_residual_scale": 0.0625,
        "p2_report_path": P2_REPORT_PATH.as_posix(),
        "p2_checkpoint_path": P2_CHECKPOINT_PATH.as_posix(),
        "p2_onnx_path": P2_ONNX_PATH.as_posix(),
        "parent_result_path": PARENT_RESULT_PATH.as_posix(),
        "onnx_parity_maximum_absolute_error": 1e-5,
        "validation_or_public_pixels_used_for_training": False,
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
            raise RuntimeError(f"OCR V24 P3 config field mismatch: {key}")
    if config.get("selection_thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V24 P3 thresholds changed")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V24 P3 runner source bundle changed")
    exact_inputs = {
        SEAL_PATH: config["split_seal_sha256"],
        P2_RESULT_PATH: config["p2_result_sha256"],
        P2_REPORT_PATH: config["p2_report_sha256"],
        P2_CHECKPOINT_PATH: config["p2_checkpoint_sha256"],
        P2_ONNX_PATH: config["p2_onnx_sha256"],
        PARENT_RESULT_PATH: config["parent_result_sha256"],
        Path(DETECTOR_PATH): DETECTOR_SHA256,
        Path(RECOGNIZER_PATH): RECOGNIZER_SHA256,
        Path(RECOGNIZER_YAML_PATH): RECOGNIZER_YAML_SHA256,
    }
    for relative, expected_hash in exact_inputs.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V24 P3 frozen input changed: {relative.as_posix()}")
    p2 = _read_json(REPO_ROOT / P2_RESULT_PATH)
    metrics = p2.get("selection_metrics", {})
    if (
        p2.get("status") != "failed_selection"
        or p2.get("candidate_consumed") is not True
        or metrics.get("true_positives") != 1022
        or metrics.get("false_positives") != 0
        or metrics.get("false_negatives") != 2
        or metrics.get("prohibited_structure_hits") != 0
        or metrics.get("role_accuracy") != 0.9931640625
        or p2.get("parent_role_maximum_absolute_error") != 0.0
        or p2.get("onnx_parity_passed") is not True
        or p2.get("public_gate_archive_opened") is not False
        or p2.get("case_level_details_emitted") is not False
    ):
        raise RuntimeError("OCR V24 P3 aggregate-only P2 trigger changed")
    parent = _read_json(REPO_ROOT / PARENT_RESULT_PATH)
    parent_metrics = parent.get("selection_metrics", {})
    if (
        parent.get("candidate_id") != "P3"
        or parent.get("status") != "failed_selection"
        or parent_metrics.get("true_positives") != 1024
        or parent_metrics.get("false_positives") != 3
        or parent_metrics.get("false_negatives") != 0
        or parent_metrics.get("prohibited_structure_hits") != 3
        or parent.get("public_gate_archive_opened") is not False
        or parent.get("case_level_details_emitted") is not False
    ):
        raise RuntimeError("OCR V24 P3 aggregate-only V23 parent trigger changed")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    head = _repository_head()
    if not _is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V24 split seal source commit is not an ancestor")
    for relative, expected_hash in seal.get("source_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V24 frozen split source changed: {relative}")
    bindings = {
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }
    for split, (relative, config_key) in bindings.items():
        actual = sha256_file(REPO_ROOT / relative)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config[config_key]:
            raise RuntimeError(f"OCR V24 {split} archive changed before P3")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V24 P3 output already exists")
    return {"config": config, "head": head, "seal": seal}


def run_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V24 P3 output exists: {output_dir}")
    evidence = preflight()
    config = evidence["config"]
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "load_frozen_selection"
    try:
        selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
        _validate_stored_split(
            selection_scenes, evidence["seal"]["splits"]["validation"], "validation",
        )
        detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        detector_input = detector_session.get_inputs()[0].name
        recognizer_input = recognizer_session.get_inputs()[0].name
        alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)

        def detector_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(
                detector_session.run(None, {detector_input: np.ascontiguousarray(values)})[0],
                dtype=np.float32,
            )

        def recognizer_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(
                recognizer_session.run(None, {recognizer_input: np.ascontiguousarray(values)})[0],
                dtype=np.float32,
            )

        phase = "direct_selection_feature_and_crop_execution"
        values, crops, _, records, selection_evidence = extract_crop_evidence(
            selection_scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        groups = _feature_groups(records, len(selection_scenes))
        registered = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V24 P3 incomplete selection proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in selection_scenes)
        if sum(record.truth_index >= 0 for record in records) != expected_truths:
            raise RuntimeError("OCR V24 P3 production stream omitted a validation truth")

        phase = "zero_optimizer_consensus_export"
        p2_model = _load_p2_model(config)
        model = ParentRecallResidualVetoNet(
            p2_model,
            parent_probability_minimum=float(config["parent_probability_minimum"]),
            crop_residual_margin_minimum=float(config["crop_residual_margin_minimum"]),
            accepted_logit_magnitude=float(config["accepted_logit_magnitude"]),
        ).eval()
        first = groups[0]
        example_values = torch.from_numpy(values[first]).unsqueeze(0)
        example_crops = torch.from_numpy(crops[first]).unsqueeze(0)
        onnx_path = output_dir / "graph-text-crop-evidence-role-anchor-v24-p3.onnx"
        _export(model, example_values, example_crops, onnx_path)
        candidate_session = _cpu_session(onnx_path)
        if {item.name for item in candidate_session.get_inputs()} != {
            "proposal_evidence", "proposal_crops",
        }:
            raise RuntimeError("OCR V24 P3 ONNX input identity changed")

        phase = "single_visible_selection"
        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        candidate_outputs = sha256()
        parity_error = 0.0
        p2_role_error = 0.0
        with torch.inference_mode():
            for indices in groups:
                evidence_values = np.ascontiguousarray(values[indices][None, ...])
                crop_values = np.ascontiguousarray(crops[indices][None, ...])
                torch_values = torch.from_numpy(evidence_values)
                torch_crops = torch.from_numpy(crop_values)
                expected_output = model(torch_values, torch_crops).numpy()
                p2_output = p2_model(torch_values, torch_crops).numpy()
                actual_output = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": evidence_values,
                    "proposal_crops": crop_values,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
                )
                p2_role_error = max(
                    p2_role_error,
                    float(np.max(np.abs(expected_output[:, :, 2:] - p2_output[:, :, 2:]))),
                )
                outputs.append(actual_output[0])
                evidence_inputs.update(evidence_values.tobytes(order="C"))
                crop_inputs.update(crop_values.tobytes(order="C"))
                candidate_outputs.update(np.ascontiguousarray(actual_output).tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(groups),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "p2_role_maximum_absolute_error": p2_role_error,
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
        roles_preserved = p2_role_error == 0.0
        passed = robust is not None and parity_passed and roles_preserved
        report: dict[str, object] = {
            "schema": "graphreader.ocr-crop-evidence-role-anchor-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": 0,
            "weights_changed": False,
            "objective": config["objective"],
            "consensus_gate": {
                "parent_probability_minimum": config["parent_probability_minimum"],
                "crop_residual_margin_minimum": config["crop_residual_margin_minimum"],
                "accepted_logit_magnitude": config["accepted_logit_magnitude"],
            },
            "p2_result_sha256": config["p2_result_sha256"],
            "p2_report_sha256": config["p2_report_sha256"],
            "p2_checkpoint_sha256": config["p2_checkpoint_sha256"],
            "p2_onnx_sha256": config["p2_onnx_sha256"],
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": parity_error,
            "onnx_parity_passed": parity_passed,
            "p2_role_maximum_absolute_error": p2_role_error,
            "p2_roles_preserved": roles_preserved,
            "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "passing_threshold_window": list(window),
            "selected_threshold": selected["threshold"],
            "selection_metrics": selected["metrics"],
            "training_authorization": authorization.binding,
            "split_seal_sha256": config["split_seal_sha256"],
            "selection_fixture_archive_sha256": config["selection_fixture_archive_sha256"],
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
            authorization, status=str(report["status"]), report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-crop-evidence-role-anchor-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": 0,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "case_level_details_emitted": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "production_approval": False,
            "release_eligible": False,
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(
            authorization, status="failed_runner", report_sha256=sha256_file(report_path),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    if args.preflight == args.execute:
        parser.error("select exactly one of --preflight or --execute")
    if args.preflight:
        evidence = preflight()
        print(json.dumps({
            "head": evidence["head"],
            "ready": True,
            "runner_source_bundle_sha256": source_bundle_sha256(
                REPO_ROOT, RUNNER_SOURCE_PATHS,
            ),
        }, sort_keys=True))
        return 0
    report = run_candidate(REPO_ROOT / args.output)
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": report["optimizer_steps"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
        "p2_role_maximum_absolute_error": report["p2_role_maximum_absolute_error"],
        "passing_threshold_window": report["passing_threshold_window"],
        "selected_threshold": report["selected_threshold"],
        "selection_metrics": report["selection_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
