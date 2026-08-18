# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer visible selection for OCR V25 P1."""

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
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _calibrated_records,
    _cpu_session,
    _export,
    _feature_groups,
    _is_ancestor,
    _read_json,
    _repository_head,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive, proposal_summary, split_fingerprint
from .model import FrozenCropResidualCtcRescueNet
from .pipeline import extract_crop_evidence
from .protocol import (
    ACCEPTED_LOGIT_MAGNITUDE,
    DETECTOR_PATH,
    DETECTOR_SHA256,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    PARENT_ACCEPTANCE_MINIMUM,
    PARENT_CHECKPOINT_PATH,
    PARENT_CHECKPOINT_SHA256,
    PARENT_ONNX_PATH,
    PARENT_ONNX_SHA256,
    PARENT_RESULT_PATH,
    PARENT_RESULT_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    TASK,
    THRESHOLDS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/evidence_rescue_v25")
CANDIDATE_ID = "P1"
CONFIG_PATH = ROOT / "training/p1.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v25-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v25-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v25-public.zip")
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "model.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p1.py",
    Path(PARENT_RESULT_PATH),
    Path("ml/ocr/crop_evidence_role_anchor_v24/model_p2.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/pipeline.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/protocol.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/train_p1.py"),
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


def _load_parent_state() -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        REPO_ROOT / PARENT_CHECKPOINT_PATH, map_location="cpu", weights_only=True,
    )
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("OCR V25 parent checkpoint has no state_dict")
    return state_dict


def _validate_stored_split(
    scenes: tuple[Any, ...], registered: dict[str, Any], name: str,
) -> None:
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != registered["split_fingerprint"]
        or any(summary[key] != registered["proposal_summary"][key] for key in summary)
    ):
        raise RuntimeError(f"OCR V25 {name} stored fixtures violate the seal")


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    required = {
        "schema": "graphreader.ocr-evidence-rescue-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "objective": "zero-optimizer-frozen-parent-ctc-evidence-rescue-v1",
        "optimizer_steps": 0,
        "provider": "CPUExecutionProvider",
        "selection_thresholds": list(THRESHOLDS),
        "complete_proposal_negative_cap_per_scene": 10000,
        "recognition_batch_size": 64,
        "parent_acceptance_minimum": PARENT_ACCEPTANCE_MINIMUM,
        "accepted_logit_magnitude": ACCEPTED_LOGIT_MAGNITUDE,
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
        "parent_result_path": PARENT_RESULT_PATH,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "split_seal_path": SEAL_PATH.as_posix(),
        "train_fixture_archive_path": TRAIN_ARCHIVE_PATH.as_posix(),
        "selection_fixture_archive_path": SELECTION_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "validation_or_public_pixels_used_for_design": False,
        "case_level_predecessor_evidence_used": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(f"OCR V25 P1 config changed: {key}")
    head = _repository_head()
    source_commit = config.get("split_source_commit")
    if not isinstance(source_commit, str) or not _is_ancestor(source_commit, head):
        raise RuntimeError("OCR V25 split source commit is not an ancestor of HEAD")
    runner_hash = source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    if config.get("expected_runner_source_bundle_sha256") != runner_hash:
        raise RuntimeError("OCR V25 P1 runner source bundle changed")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    if (
        seal.get("schema") != "graphreader.ocr-evidence-rescue-split-seal.v1"
        or seal.get("revision") != REVISION
        or seal.get("source_commit") != source_commit
        or seal.get("source_bundle_sha256") != config.get("split_source_bundle_sha256")
        or sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256")
        or seal.get("optimizer_steps_at_freeze") != 0
        or seal.get("selection_evaluations") != 0
        or seal.get("public_evaluations") != 0
    ):
        raise RuntimeError("OCR V25 split seal changed")
    expected_archives = {
        TRAIN_ARCHIVE_PATH: config.get("train_fixture_archive_sha256"),
        SELECTION_ARCHIVE_PATH: config.get("selection_fixture_archive_sha256"),
        PUBLIC_ARCHIVE_PATH: config.get("public_fixture_archive_sha256"),
    }
    exact_inputs = {
        Path(DETECTOR_PATH): DETECTOR_SHA256,
        Path(RECOGNIZER_PATH): RECOGNIZER_SHA256,
        Path(RECOGNIZER_YAML_PATH): RECOGNIZER_YAML_SHA256,
        Path(PARENT_CHECKPOINT_PATH): PARENT_CHECKPOINT_SHA256,
        Path(PARENT_ONNX_PATH): PARENT_ONNX_SHA256,
        Path(PARENT_RESULT_PATH): PARENT_RESULT_SHA256,
        **expected_archives,
    }
    for relative, expected_hash in exact_inputs.items():
        if not isinstance(expected_hash, str) or sha256_file(REPO_ROOT / relative) != expected_hash:
            raise RuntimeError(f"OCR V25 frozen input changed: {relative.as_posix()}")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V25 P1 output already exists")
    return {"config": config, "head": head, "seal": seal}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V25 P1 output exists: {output_dir}")
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
    phase = "load_frozen_parent"
    optimizer_steps = 0
    try:
        model = FrozenCropResidualCtcRescueNet()
        model.load_parent_state_dict(_load_parent_state())
        model.eval()
        checkpoint_path = output_dir / "graph-text-evidence-rescue-v25-p1.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)

        phase = "candidate_onnx_export"
        example_evidence = torch.zeros(1, 3, 31, dtype=torch.float32)
        example_crops = torch.zeros(1, 3, 2, 32, 128, dtype=torch.float32)
        onnx_path = output_dir / "graph-text-evidence-rescue-v25-p1.onnx"
        _export(model, example_evidence, example_crops, onnx_path)
        candidate_session = _cpu_session(onnx_path)
        if {item.name for item in candidate_session.get_inputs()} != {
            "proposal_evidence", "proposal_crops",
        }:
            raise RuntimeError("OCR V25 P1 ONNX input identity changed")

        phase = "direct_stored_fixture_selection"
        selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
        _validate_stored_split(
            selection_scenes, evidence["seal"]["splits"]["validation"], "validation",
        )
        detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        detector_input = detector_session.get_inputs()[0].name
        recognizer_input = recognizer_session.get_inputs()[0].name
        detector_runner = lambda values: detector_session.run(None, {detector_input: values})[0]
        recognizer_runner = lambda values: recognizer_session.run(None, {recognizer_input: values})[0]
        alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)
        selection_values, selection_crops, _, records, selection_evidence = (
            extract_crop_evidence(
                selection_scenes,
                detector_runner,
                recognizer_runner,
                alphabet,
                mode="train",
                negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
                recognition_batch_size=int(config["recognition_batch_size"]),
            )
        )
        registered = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V25 incomplete selection proposal stream: {key}")

        groups = _feature_groups(records, len(selection_scenes))
        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        candidate_outputs = sha256()
        parent_outputs = sha256()
        parity_error = 0.0
        role_error = 0.0
        with torch.inference_mode():
            for indices in groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                crops = np.ascontiguousarray(selection_crops[indices][None, ...])
                torch_values = torch.from_numpy(values)
                torch_crops = torch.from_numpy(crops)
                expected = model(torch_values, torch_crops).numpy()
                parent = model.parent(torch_values, torch_crops).numpy()
                actual = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                })[0], dtype=np.float32)
                parity_error = max(parity_error, float(np.max(np.abs(expected - actual))))
                role_error = max(
                    role_error, float(np.max(np.abs(actual[:, :, 2:] - parent[:, :, 2:]))),
                )
                outputs.append(actual[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                candidate_outputs.update(np.ascontiguousarray(actual).tobytes(order="C"))
                parent_outputs.update(np.ascontiguousarray(parent).tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(groups),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "parent_output_tensor_stream_sha256": parent_outputs.hexdigest(),
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "parent_role_maximum_absolute_error": role_error,
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
        roles_preserved = role_error == 0.0
        passed = robust is not None and parity_passed and roles_preserved
        report: dict[str, object] = {
            "schema": "graphreader.ocr-evidence-rescue-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": optimizer_steps,
            "weights_changed": False,
            "objective": config["objective"],
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": parity_error,
            "onnx_parity_passed": parity_passed,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_role_maximum_absolute_error": role_error,
            "parent_roles_preserved": roles_preserved,
            "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "passing_threshold_window": list(window),
            "selected_threshold": selected["threshold"],
            "selection_metrics": selected["metrics"],
            "selection_evidence": selection_evidence,
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
            "schema": "graphreader.ocr-evidence-rescue-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": optimizer_steps,
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
        complete_training_candidate(
            authorization, status="failed_runner", report_sha256=sha256_file(report_path),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        value = preflight()
        print(json.dumps({
            "repository_head": value["head"],
            "split_seal_sha256": sha256_file(REPO_ROOT / SEAL_PATH),
            "runner_source_bundle_sha256": source_bundle_sha256(
                REPO_ROOT, RUNNER_SOURCE_PATHS,
            ),
            "candidate_id": CANDIDATE_ID,
            "optimizer_steps": 0,
        }, indent=2, sort_keys=True))
        return 0
    report = train_candidate(REPO_ROOT / CANONICAL_OUTPUT)
    print(json.dumps({
        "status": report["status"],
        "selection_gate_passed": report.get("selection_gate_passed", False),
        "optimizer_steps": report["optimizer_steps"],
        "onnx_parity_maximum_absolute_error": report.get(
            "onnx_parity_maximum_absolute_error"
        ),
        "selection_metrics": report.get("selection_metrics"),
        "elapsed_ms": report["elapsed_ms"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
