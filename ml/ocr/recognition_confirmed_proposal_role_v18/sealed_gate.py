# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for OCR V18."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import acquire_gate_seal, canonical_json_bytes, complete_gate_seal, sha256_file
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from .dataset import load_split_archive, proposal_summary, split_fingerprint
from .pipeline import evaluate_composition, passes_selection
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    PUBLIC_REVISION,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    REVISION,
    TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/recognition_confirmed_proposal_role_v18")
LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")
SPLIT_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "sealed_gate.py",
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"), Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    "evaluation_limit": 1,
    "exact_region_and_role_every_scene": True,
    "false_regions": 0,
    "missed_regions": 0,
    "duplicate_regions": 0,
    "prohibited_structure_hits": 0,
    "recognition_exact_minimum": 0.90,
    "character_error_rate_maximum": 0.05,
    "role_accuracy_minimum": 0.90,
    "per_role_accuracy_minimum": 0.85,
    "provider": "CPUExecutionProvider",
    "direct_fixture_byte_execution_required": True,
    "detector_and_recognizer_tensor_stream_hashes_required": True,
    "case_level_failure_analysis_permitted": False,
}


def _cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V18 public gate requires CPUExecutionProvider only")
    return session


def evaluate_public(*, selection_report_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"OCR V18 public output exists: {output_path}")
    ledger = json.loads((REPO_ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
    entry = next(
        (item for item in ledger["revisions"] if item.get("task") == TASK and item.get("revision") == REVISION),
        None,
    )
    selection = json.loads(selection_report_path.read_text(encoding="utf-8"))
    selection_sha = sha256_file(selection_report_path)
    if (
        entry is None
        or entry.get("status") != "candidate_1_selected_public_gate_pending"
        or entry.get("preregistered_candidate_ids") != []
        or entry.get("consumed_candidate_ids") != ["P1"]
        or entry.get("execution_authorized") is not False
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_authorized_candidate_id") != "P1"
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_archive_opened") is not False
        or entry.get("p1_selection_report_sha256") != selection_sha
    ):
        raise RuntimeError("OCR V18 public gate is not authorized by the canonical ledger")
    if (
        selection.get("status") != "selected"
        or selection.get("selection_gate_passed") is not True
        or selection.get("detector_sha256") != DETECTOR_SHA256
        or selection.get("recognizer_sha256") != RECOGNIZER_SHA256
        or selection.get("public_gate_archive_opened") is not False
        or selection.get("public_gate_evaluations") != 0
    ):
        raise RuntimeError("Only the exact selected OCR V18 pair may open the public gate")
    config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR V18 public seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if (
        sha256_file(archive_path) != seal["fixture_archive_sha256"]
        or sha256_file(private_manifest_path) != seal["private_manifest_sha256"]
    ):
        raise RuntimeError("OCR V18 public fixture bytes changed")
    if sha256_file(REPO_ROOT / DETECTOR_PATH) != DETECTOR_SHA256:
        raise RuntimeError("OCR V18 detector payload changed")
    if sha256_file(REPO_ROOT / RECOGNIZER_PATH) != RECOGNIZER_SHA256:
        raise RuntimeError("OCR V18 recognizer payload changed")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes={
            "detector_onnx_sha256": DETECTOR_SHA256,
            "recognizer_onnx_sha256": RECOGNIZER_SHA256,
            "selection_report_sha256": selection_sha,
        },
        dataset_manifest_sha256=seal["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    scenes = load_split_archive(archive_path, private_manifest_path, expected_split="sealed_public")
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != seal["split_fingerprint"]
        or any(summary[key] != seal[key] for key in summary)
    ):
        raise RuntimeError("OCR V18 public fixtures violate the frozen split")
    detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
    recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
    alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)
    detector_input = detector_session.get_inputs()[0].name
    recognizer_input = recognizer_session.get_inputs()[0].name

    def detector_runner(values: np.ndarray) -> np.ndarray:
        return np.asarray(detector_session.run(None, {detector_input: values})[0], dtype=np.float32)

    def recognizer_runner(values: np.ndarray) -> np.ndarray:
        return np.asarray(recognizer_session.run(None, {recognizer_input: values})[0], dtype=np.float32)

    metrics = evaluate_composition(scenes, detector_runner, recognizer_runner, alphabet)
    passed = passes_selection(metrics)
    report: dict[str, object] = {
        "schema": "graphreader.ocr-recognition-confirmed-public-gate.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "status": "pass" if passed else "fail",
        "evaluation_count": 1,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "selection_report_sha256": selection_sha,
        "fixture_archive_sha256": seal["fixture_archive_sha256"],
        "provider": "CPUExecutionProvider",
        "metrics": metrics,
        "marker_creation_evaluated": False,
        "marker_creation_gate_required_before_production_approval": True,
        "gate_requirements": GATE_CONFIG,
        "seal_binding": gate.binding,
        "canonical_seal_key": gate.key,
        "case_level_details_emitted": False,
        "production_approval": False,
        "release_eligible": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(gate, status=str(report["status"]), report_sha256=sha256_file(output_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_public(
        selection_report_path=REPO_ROOT / args.selection_report,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
