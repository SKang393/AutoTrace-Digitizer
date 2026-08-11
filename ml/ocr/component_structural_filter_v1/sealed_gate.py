# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for the selected OCR structural-filter candidate."""

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
from ml.ocr.component_geometric_v4.dataset import load_sealed_public_archive

from .pipeline import evaluate_samples
from .protocol import (
    MARKER_EXCLUSION_ACCURACY_MINIMUM,
    PUBLIC_REVISION,
    REVISION,
    ROLE_ACCURACY_MINIMUM,
    SEALED_CER_MAXIMUM,
    SEALED_EXACT_MATCH_MINIMUM,
    SOURCE_CONFIDENCE_THRESHOLD,
    SOURCE_ONNX_SHA256,
    STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
    TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = Path("ml/ocr/component_structural_filter_v1/gates/sealed-public-v1.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/ocr/component_geometric_v4/p3_dataset.py"),
    Path("ml/ocr/component_structural_filter_v1/pipeline.py"),
    Path("ml/ocr/component_structural_filter_v1/protocol.py"),
    Path("ml/ocr/component_structural_filter_v1/sealed_gate.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)
GATE_CONFIG = {
    "confidence_threshold": SOURCE_CONFIDENCE_THRESHOLD,
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
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    selection_report_sha256 = sha256_file(selection_report_path)
    candidate_hash = sha256_file(onnx_path)
    if (
        entry is None
        or entry.get("status") != "selection_passed_public_preregistered"
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_authorized_onnx_sha256") != candidate_hash
        or entry.get("public_gate_authorized_selection_report_sha256") != selection_report_sha256
    ):
        raise RuntimeError("Structural-filter public gate is not authorized by the canonical ledger")
    if (
        selection_report.get("status") != "selected"
        or selection_report.get("selection_gate_passed") is not True
        or selection_report.get("revision") != REVISION
        or selection_report.get("source_onnx_sha256") != candidate_hash
        or selection_report.get("optimizer_steps") != 0
        or selection_report.get("weights_changed") is not False
        or selection_report.get("sealed_public_archive_opened") is not False
        or selection_report.get("public_gate_evaluations") != 0
    ):
        raise RuntimeError("Only the exact validation-selected zero-training candidate may open the public gate")
    if candidate_hash != SOURCE_ONNX_SHA256:
        raise RuntimeError("Structural-filter source ONNX checksum mismatch")
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / split_config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("Structural-filter public seal differs from gate configuration")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"]:
        raise RuntimeError("Structural-filter sealed fixture archive checksum mismatch")
    if sha256_file(private_manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("Structural-filter sealed private manifest checksum mismatch")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes={
            "source_onnx_sha256": candidate_hash,
            "selection_report_sha256": selection_report_sha256,
        },
        dataset_manifest_sha256=seal["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    samples = load_sealed_public_archive(archive_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Structural-filter public gate requires CPUExecutionProvider only")
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

    metrics = evaluate_samples(samples, runner, SOURCE_CONFIDENCE_THRESHOLD)
    passed = (
        float(metrics["exact_match"]) >= SEALED_EXACT_MATCH_MINIMUM
        and float(metrics["character_error_rate"]) <= SEALED_CER_MAXIMUM
        and float(metrics["role_accuracy"]) >= ROLE_ACCURACY_MINIMUM
        and float(metrics["marker_exclusion_accuracy"]) >= MARKER_EXCLUSION_ACCURACY_MINIMUM
    )
    report: dict[str, object] = {
        "schema": "graphreader.ocr-component-structural-filter-public-gate.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "status": "pass" if passed else "fail",
        "production_approval": False,
        "release_eligible": False,
        "evaluation_count": 1,
        "source_onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "source_onnx_sha256": candidate_hash,
        "selection_report_path": selection_report_path.relative_to(REPO_ROOT).as_posix(),
        "selection_report_sha256": selection_report_sha256,
        "fixture_archive_sha256": seal["fixture_archive_sha256"],
        "private_manifest_sha256": seal["private_manifest_sha256"],
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "provider": "CPUExecutionProvider",
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
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(gate, status=str(report["status"]), report_sha256=sha256_file(output_path))
    return report


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
