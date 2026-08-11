# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for a validation-selected OCR V5 candidate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    require_committed_sources,
    sha256_file,
)
from ml.markers.training_budget import CANONICAL_LEDGER_PATH

from .dataset import load_sealed_public_archive
from .pipeline import evaluate_samples
from .protocol import (
    MARKER_EXCLUSION_ACCURACY_MINIMUM,
    PUBLIC_REVISION,
    REVISION,
    ROLE_ACCURACY_MINIMUM,
    SEALED_CER_MAXIMUM,
    SEALED_EXACT_MATCH_MINIMUM,
    STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
    TASK,
    THRESHOLDS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = Path("ml/ocr/component_ensemble_v5/gates/sealed-public-v1.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/component_ensemble_v5/dataset.py"),
    Path("ml/ocr/component_ensemble_v5/pipeline.py"),
    Path("ml/ocr/component_ensemble_v5/protocol.py"),
    Path("ml/ocr/component_ensemble_v5/sealed_gate.py"),
    Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)
GATE_CONFIG = {
    "allowed_confidence_thresholds": list(THRESHOLDS),
    "structural_reject_minimum_height_ratio": STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
    "structural_operator": ">=",
    "structural_position": "before_classifier",
    "sealed_exact_match_minimum": SEALED_EXACT_MATCH_MINIMUM,
    "sealed_cer_maximum": SEALED_CER_MAXIMUM,
    "sealed_role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
    "marker_exclusion_accuracy_minimum": MARKER_EXCLUSION_ACCURACY_MINIMUM,
    "provider": "CPUExecutionProvider",
}


def evaluate_candidate(*, onnx_path: Path, selection_report_path: Path, output_path: Path) -> dict[str, object]:
    require_committed_sources(REPO_ROOT, (CANONICAL_LEDGER_PATH,))
    ledger = json.loads((REPO_ROOT / CANONICAL_LEDGER_PATH).read_text(encoding="utf-8"))
    entry = next(
        (item for item in ledger.get("revisions", []) if item.get("task") == TASK and item.get("revision") == REVISION),
        None,
    )
    report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    report_sha256 = sha256_file(selection_report_path)
    onnx_sha256 = sha256_file(onnx_path)
    if (
        entry is None
        or entry.get("status") != "selection_passed_public_preregistered"
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_authorized_onnx_sha256") != onnx_sha256
        or entry.get("public_gate_authorized_selection_report_sha256") != report_sha256
    ):
        raise RuntimeError("OCR V5 public gate is not authorized by the canonical ledger")
    threshold = float(report.get("selected_threshold", -1.0))
    if (
        report.get("status") != "selected"
        or report.get("selection_gate_passed") is not True
        or report.get("onnx_parity_passed") is not True
        or report.get("revision") != REVISION
        or report.get("onnx_sha256") != onnx_sha256
        or report.get("sealed_public_archive_opened") is not False
        or report.get("public_gate_evaluations") != 0
        or threshold not in THRESHOLDS
    ):
        raise RuntimeError("Only the exact validation-selected OCR V5 candidate may open the public gate")
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / split_config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR V5 public seal differs from the frozen gate configuration")
    seal_data = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal_data["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal_data["private_manifest_path"]
    if sha256_file(archive_path) != seal_data["fixture_archive_sha256"]:
        raise RuntimeError("OCR V5 sealed fixture archive checksum mismatch")
    if sha256_file(private_manifest_path) != seal_data["private_manifest_sha256"]:
        raise RuntimeError("OCR V5 sealed private manifest checksum mismatch")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes={"onnx_sha256": onnx_sha256, "selection_report_sha256": report_sha256},
        dataset_manifest_sha256=seal_data["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    samples = load_sealed_public_archive(archive_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V5 public gate requires CPUExecutionProvider only")
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

    metrics = evaluate_samples(samples, runner, threshold)
    passed = (
        float(metrics["exact_match"]) >= SEALED_EXACT_MATCH_MINIMUM
        and float(metrics["character_error_rate"]) <= SEALED_CER_MAXIMUM
        and float(metrics["role_accuracy"]) >= ROLE_ACCURACY_MINIMUM
        and float(metrics["marker_exclusion_accuracy"]) >= MARKER_EXCLUSION_ACCURACY_MINIMUM
    )
    output: dict[str, object] = {
        "schema": "graphreader.ocr-component-ensemble-public-gate.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "status": "pass" if passed else "fail",
        "production_approval": False,
        "release_eligible": False,
        "evaluation_count": 1,
        "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "onnx_sha256": onnx_sha256,
        "selection_report_path": selection_report_path.relative_to(REPO_ROOT).as_posix(),
        "selection_report_sha256": report_sha256,
        "fixture_archive_sha256": seal_data["fixture_archive_sha256"],
        "private_manifest_sha256": seal_data["private_manifest_sha256"],
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "provider": "CPUExecutionProvider",
        "selected_threshold": threshold,
        "metrics": metrics,
        "direct_execution": {
            "inference_calls": inference_calls,
            "input_tensor_stream_sha256": input_digest.hexdigest(),
            "output_tensor_stream_sha256": output_digest.hexdigest(),
        },
        "gate_requirements": GATE_CONFIG,
        "seal_binding": gate.binding,
        "canonical_seal_key": gate.key,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(output))
    complete_gate_seal(gate, status=str(output["status"]), report_sha256=sha256_file(output_path))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_candidate(
        onnx_path=REPO_ROOT / arguments.onnx,
        selection_report_path=REPO_ROOT / arguments.selection_report,
        output_path=REPO_ROOT / arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
