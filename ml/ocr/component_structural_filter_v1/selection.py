# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-training selection runner for the OCR structural filter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.component_geometric_v4.dataset import build_split, split_fingerprint

from .pipeline import evaluate_samples
from .protocol import (
    CANDIDATE_ID,
    MARKER_EXCLUSION_ACCURACY_MINIMUM,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    REVISION,
    ROLE_ACCURACY_MINIMUM,
    SOURCE_CANDIDATE_ID,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_CONFIDENCE_THRESHOLD,
    SOURCE_ONNX_SHA256,
    SOURCE_REPORT_SHA256,
    SOURCE_REVISION,
    STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
    TASK,
    VALIDATION_CER_MAXIMUM,
    VALIDATION_EXACT_MATCH_MINIMUM,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("ml/ocr/component_structural_filter_v1/training/p1.json")
CANONICAL_OUTPUT = Path("ml/ocr/component_structural_filter_v1/artifacts/P1-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/ocr/component_geometric_v4/p3_dataset.py"),
    Path("ml/ocr/component_structural_filter_v1/pipeline.py"),
    Path("ml/ocr/component_structural_filter_v1/protocol.py"),
    Path("ml/ocr/component_structural_filter_v1/selection.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def select_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
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
    phase = "source_binding"
    try:
        source_result_path = REPO_ROOT / config["source_result_path"]
        source_report_path = REPO_ROOT / config["source_report_path"]
        source_onnx_path = REPO_ROOT / config["source_onnx_path"]
        if sha256_file(source_result_path) != config["source_result_sha256"]:
            raise RuntimeError("Tracked OCR V4 P3 result checksum mismatch")
        if sha256_file(source_report_path) != SOURCE_REPORT_SHA256:
            raise RuntimeError("Ignored OCR V4 P3 report checksum mismatch")
        if sha256_file(source_onnx_path) != SOURCE_ONNX_SHA256:
            raise RuntimeError("Ignored OCR V4 P3 ONNX checksum mismatch")
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if (
            source_report.get("task") != TASK
            or source_report.get("revision") != SOURCE_REVISION
            or source_report.get("candidate_id") != SOURCE_CANDIDATE_ID
            or source_report.get("status") != "failed_selection"
            or source_report.get("onnx_sha256") != SOURCE_ONNX_SHA256
            or source_report.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or float(source_report.get("selected_threshold")) != SOURCE_CONFIDENCE_THRESHOLD
            or source_report.get("onnx_parity_passed") is not True
            or float(source_report.get("onnx_parity_maximum_absolute_error")) > ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
            or int(source_report.get("public_gate_evaluations")) != 0
            or source_report.get("sealed_public_archive_opened") is not False
        ):
            raise RuntimeError("OCR V4 P3 report is not the exact fail-closed source candidate")
        selection_path = REPO_ROOT / config["source_selection_manifest_path"]
        if sha256_file(selection_path) != config["source_selection_manifest_sha256"]:
            raise RuntimeError("Source selection manifest checksum mismatch")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        source_seal_path = REPO_ROOT / config["source_sealed_public_test_seal_path"]
        if sha256_file(source_seal_path) != config["source_sealed_public_test_seal_sha256"]:
            raise RuntimeError("Source sealed-public seal checksum mismatch")
        source_seal = json.loads(source_seal_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / source_seal["fixture_archive_path"]) != source_seal["fixture_archive_sha256"]:
            raise RuntimeError("Sealed-public archive changed before structural-filter selection")

        phase = "validation_execution"
        validation_samples = build_split("validation")
        if split_fingerprint(validation_samples) != selection["validation_split_fingerprint"]:
            raise RuntimeError("Validation renderer no longer reproduces the frozen source split")
        session = ort.InferenceSession(str(source_onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Structural-filter selection requires CPUExecutionProvider only")
        input_digest = sha256()
        output_digest = sha256()
        inference_calls = 0

        def runner(value: np.ndarray) -> np.ndarray:
            nonlocal inference_calls
            contiguous = np.ascontiguousarray(value, dtype=np.float32)
            input_digest.update(contiguous.tobytes())
            output = np.asarray(session.run(None, {"glyphs": contiguous})[0], dtype=np.float32)
            output_digest.update(np.ascontiguousarray(output).tobytes())
            inference_calls += 1
            return output

        metrics = evaluate_samples(validation_samples, runner, SOURCE_CONFIDENCE_THRESHOLD)
        selection_passed = (
            float(metrics["exact_match"]) >= VALIDATION_EXACT_MATCH_MINIMUM
            and float(metrics["character_error_rate"]) <= VALIDATION_CER_MAXIMUM
            and float(metrics["role_accuracy"]) >= ROLE_ACCURACY_MINIMUM
            and float(metrics["marker_exclusion_accuracy"]) >= MARKER_EXCLUSION_ACCURACY_MINIMUM
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-component-structural-filter-selection-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "candidate_kind": "deterministic_postprocessor",
            "status": "selected" if selection_passed else "failed_selection",
            "selection_gate_passed": selection_passed,
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "training_authorization": authorization.binding,
            "optimizer_steps": 0,
            "weights_changed": False,
            "source_revision": SOURCE_REVISION,
            "source_candidate_id": SOURCE_CANDIDATE_ID,
            "source_report_path": config["source_report_path"],
            "source_report_sha256": SOURCE_REPORT_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_onnx_path": config["source_onnx_path"],
            "source_onnx_sha256": SOURCE_ONNX_SHA256,
            "source_onnx_parity_maximum_absolute_error": source_report["onnx_parity_maximum_absolute_error"],
            "source_onnx_parity_passed": True,
            "provider": "CPUExecutionProvider",
            "confidence_threshold": SOURCE_CONFIDENCE_THRESHOLD,
            "structural_rule": {
                "field": "component_height_ratio",
                "operator": ">=",
                "reject_minimum": STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
                "position": "before_classifier",
            },
            "validation_sample_count": len(validation_samples),
            "validation_split_fingerprint": selection["validation_split_fingerprint"],
            "metrics": metrics,
            "direct_execution": {
                "inference_calls": inference_calls,
                "input_tensor_stream_sha256": input_digest.hexdigest(),
                "output_tensor_stream_sha256": output_digest.hexdigest(),
            },
            "source_selection_manifest_sha256": config["source_selection_manifest_sha256"],
            "source_sealed_public_test_seal_sha256": config["source_sealed_public_test_seal_sha256"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-component-structural-filter-selection-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False,
            "optimizer_steps": 0,
            "weights_changed": False,
            "phase": phase,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    arguments = parser.parse_args()
    report = select_candidate(REPO_ROOT / arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
