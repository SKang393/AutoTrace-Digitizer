# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for a selected component-geometric OCR candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    sha256_file,
)

from .dataset import load_sealed_public_archive
from .pipeline import evaluate_samples
from .protocol import (
    MARKER_EXCLUSION_ACCURACY_MINIMUM,
    PUBLIC_REVISION,
    ROLE_ACCURACY_MINIMUM,
    SEALED_CER_MAXIMUM,
    SEALED_EXACT_MATCH_MINIMUM,
    TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = Path("ml/ocr/component_geometric_v4/gates/sealed-public-v1.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/ocr/component_geometric_v4/pipeline.py"),
    Path("ml/ocr/component_geometric_v4/protocol.py"),
    Path("ml/ocr/component_geometric_v4/sealed_gate.py"),
)
GATE_CONFIG = {
    "threshold_source": "selected candidate validation report",
    "sealed_exact_match_minimum": SEALED_EXACT_MATCH_MINIMUM,
    "sealed_cer_maximum": SEALED_CER_MAXIMUM,
    "sealed_role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
    "marker_exclusion_accuracy_minimum": MARKER_EXCLUSION_ACCURACY_MINIMUM,
    "provider": "CPUExecutionProvider",
}


def evaluate_candidate(*, onnx_path: Path, training_report_path: Path, output_path: Path) -> dict[str, object]:
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report.get("status") != "selected" or training_report.get("selection_gate_passed") is not True:
        raise RuntimeError("Only a validation-selected OCR V4 candidate may open the public gate")
    candidate_hash = sha256_file(onnx_path)
    if candidate_hash != training_report.get("onnx_sha256"):
        raise RuntimeError("OCR V4 ONNX differs from its validation report")
    seal_path = REPO_ROOT / split_config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR V4 public seal differs from gate configuration")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"]:
        raise RuntimeError("OCR V4 sealed fixture archive checksum mismatch")
    if sha256_file(private_manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("OCR V4 sealed private manifest checksum mismatch")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes={"onnx_sha256": candidate_hash},
        dataset_manifest_sha256=seal["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    samples = load_sealed_public_archive(archive_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V4 public gate requires CPUExecutionProvider only")

    def runner(value: np.ndarray) -> np.ndarray:
        return np.asarray(session.run(None, {"glyphs": value})[0], dtype=np.float32)

    threshold = float(training_report["selected_threshold"])
    metrics = evaluate_samples(samples, runner, threshold)
    passed = (
        metrics["exact_match"] >= SEALED_EXACT_MATCH_MINIMUM
        and metrics["character_error_rate"] <= SEALED_CER_MAXIMUM
        and metrics["role_accuracy"] >= ROLE_ACCURACY_MINIMUM
        and metrics["marker_exclusion_accuracy"] >= MARKER_EXCLUSION_ACCURACY_MINIMUM
    )
    report: dict[str, object] = {
        "schema": "graphreader.ocr-component-geometric-public-gate.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "status": "pass" if passed else "fail",
        "production_approval": False,
        "release_eligible": False,
        "evaluation_count": 1,
        "candidate_onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "candidate_onnx_sha256": candidate_hash,
        "training_report_path": training_report_path.relative_to(REPO_ROOT).as_posix(),
        "training_report_sha256": sha256_file(training_report_path),
        "fixture_archive_sha256": seal["fixture_archive_sha256"],
        "private_manifest_sha256": seal["private_manifest_sha256"],
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "threshold": threshold,
        "provider": "CPUExecutionProvider",
        "metrics": metrics,
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
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_candidate(
        onnx_path=REPO_ROOT / arguments.onnx,
        training_report_path=REPO_ROOT / arguments.training_report,
        output_path=REPO_ROOT / arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
