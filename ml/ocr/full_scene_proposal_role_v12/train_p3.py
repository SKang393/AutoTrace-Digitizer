# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer P3 structural repair for OCR V12."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_split, proposal_summary, proposal_targets, proposals, split_fingerprint
from .pipeline_p3 import evaluate_thresholds
from .protocol import ROLE_ACCURACY_MINIMUM, ROLE_CLASS_ACCURACY_MINIMUM, REVISION, TASK
from .structural_guard import (
    MAXIMUM_INK_DENSITY,
    MINIMUM_WIDTH_HEIGHT_RATIO,
    REQUIRED_COMPONENT_COUNT,
    is_rejected_structure,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/full_scene_proposal_role_v12")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "pipeline_p3.py", ROOT / "protocol.py",
    ROOT / "structural_guard.py", ROOT / "train_p3.py",
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _guard_safety(scenes: tuple[object, ...]) -> dict[str, int]:
    guarded_positive = guarded_negative = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        labels, _ = proposal_targets(scene, candidates)
        for candidate, label in zip(candidates, labels, strict=True):
            if is_rejected_structure(candidate):
                guarded_positive += int(label == 1)
                guarded_negative += int(label == 0)
    return {"guarded_positive_proposals": guarded_positive, "guarded_negative_proposals": guarded_negative}


def run_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V12 P3 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started, phase = time.perf_counter(), "initialization"
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
            raise RuntimeError("OCR V12 P3 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V12 P3 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V12 P3 public seal changed")
        if config["p2_result_sha256"] != sha256_file(REPO_ROOT / P2_RESULT_PATH):
            raise RuntimeError("OCR V12 P2 result changed")
        p2 = json.loads((REPO_ROOT / P2_RESULT_PATH).read_text(encoding="utf-8"))
        p2_report_path = REPO_ROOT / p2["report_path"]
        p2_onnx_path = REPO_ROOT / "ml/ocr/full_scene_proposal_role_v12/artifacts/P2-run/graph-text-full-scene-proposal-role-v12-p2.onnx"
        if sha256_file(p2_report_path) != config["p2_report_sha256"] or sha256_file(p2_onnx_path) != config["p2_onnx_sha256"]:
            raise RuntimeError("OCR V12 P2 report or ONNX bytes changed")
        if p2.get("onnx_parity_passed") is not True or p2.get("onnx_parity_maximum_absolute_error") != config["inherited_onnx_parity_maximum_absolute_error"]:
            raise RuntimeError("OCR V12 P2 parity evidence changed")
        validation = build_split("validation")
        training = build_split("train")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V12 P3 validation split changed")
        if proposal_summary(validation) != {key: selection["validation"][key] for key in proposal_summary(validation)}:
            raise RuntimeError("OCR V12 P3 validation proposals changed")
        guard_safety = {"train": _guard_safety(training), "validation": _guard_safety(validation)}
        if guard_safety != config["expected_guard_safety"]:
            raise RuntimeError(f"OCR V12 P3 structural guard safety changed: {guard_safety}")
        if any(item["guarded_positive_proposals"] for item in guard_safety.values()):
            raise RuntimeError("OCR V12 P3 structural guard rejects a visible truth proposal")
        session = ort.InferenceSession(str(p2_onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V12 P3 requires CPU selection execution")
        input_digest, output_digest, calls = sha256(), sha256(), 0

        def runner(input_values: np.ndarray) -> np.ndarray:
            nonlocal calls
            contiguous = np.ascontiguousarray(input_values, dtype=np.float32)
            input_digest.update(contiguous.tobytes(order="C"))
            output = np.asarray(session.run(None, {"region_proposals": contiguous})[0], dtype=np.float32)
            output_digest.update(np.ascontiguousarray(output).tobytes(order="C"))
            calls += 1
            return output

        phase = "selection"
        comparisons = evaluate_thresholds(validation, runner, tuple(float(value) for value in config["selection_thresholds"]))
        selected = max(comparisons, key=lambda item: (
            item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"],
            -item["metrics"]["false_negatives"], item["metrics"]["role_accuracy"], item["threshold"],
        ))
        metrics = selected["metrics"]
        passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["true_positives"] == metrics["truth_region_count"]
            and metrics["false_positives"] == metrics["false_negatives"] == metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
            and metrics["structural_guard_truth_match_count"] == 0
            and metrics["role_accuracy"] >= ROLE_ACCURACY_MINIMUM
            and min(metrics["per_role_accuracy"].values()) >= ROLE_CLASS_ACCURACY_MINIMUM
            and p2["onnx_parity_passed"] is True and calls == len(validation)
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-full-scene-proposal-role-candidate.v1", "task": TASK,
            "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "optimizer_steps": 0,
            "weights_changed": False, "onnx_bytes_changed": False,
            "source_candidate_id": "P2", "onnx_path": p2_onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(p2_onnx_path),
            "onnx_parity_maximum_absolute_error": p2["onnx_parity_maximum_absolute_error"],
            "onnx_parity_passed": p2["onnx_parity_passed"],
            "onnx_parity_evidence_result_path": P2_RESULT_PATH.as_posix(),
            "onnx_parity_recomputed": False, "provider": "CPUExecutionProvider",
            "structural_guard": {
                "component_count_equals": REQUIRED_COMPONENT_COUNT,
                "ink_density_maximum_inclusive": MAXIMUM_INK_DENSITY,
                "width_height_ratio_minimum_inclusive": MINIMUM_WIDTH_HEIGHT_RATIO,
            },
            "guard_safety": guard_safety, "threshold_comparisons": comparisons,
            "selected_threshold": selected["threshold"], "selection_metrics": metrics,
            "direct_execution": {
                "inference_calls": calls, "input_tensor_stream_sha256": input_digest.hexdigest(),
                "output_tensor_stream_sha256": output_digest.hexdigest(),
            },
            "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
            "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-full-scene-proposal-role-failure.v1", "task": TASK,
            "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "failed_runner",
            "phase": phase, "optimizer_steps": 0, "exception_type": type(error).__name__,
            "exception_message": str(error), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False, "release_eligible": False, "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False, "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = run_candidate(REPO_ROOT / args.output)
    print(json.dumps({
        "status": report["status"], "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report["selected_threshold"], "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
