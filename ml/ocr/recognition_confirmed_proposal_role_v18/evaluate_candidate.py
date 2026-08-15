# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer V18 visible-selection evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from .dataset import load_split_archive, proposal_summary, split_fingerprint
from .pipeline import evaluate_composition, passes_selection
from .protocol import (
    CANDIDATE_ID,
    DETECTOR_PATH,
    DETECTOR_RESULT_PATH,
    DETECTOR_RESULT_SHA256,
    DETECTOR_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/recognition_confirmed_proposal_role_v18")
CONFIG_PATH = ROOT / "evaluation/p1.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "evaluate_candidate.py",
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V18 requires CPUExecutionProvider only")
    return session


def evaluate_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V18 P1 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "preflight"
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
            raise RuntimeError("OCR V18 candidate runner sources changed")
        selection_path = REPO_ROOT / SELECTION_PATH
        if config["selection_manifest_sha256"] != sha256_file(selection_path):
            raise RuntimeError("OCR V18 selection manifest changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        exact_files = {
            DETECTOR_PATH: DETECTOR_SHA256,
            DETECTOR_RESULT_PATH: DETECTOR_RESULT_SHA256,
            RECOGNIZER_PATH: RECOGNIZER_SHA256,
            RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        }
        for relative, expected in exact_files.items():
            if sha256_file(REPO_ROOT / relative) != expected:
                raise RuntimeError(f"OCR V18 exact input changed: {relative}")
        validation_archive = REPO_ROOT / selection["validation"]["fixture_archive_path"]
        validation_manifest = REPO_ROOT / selection["validation"]["private_manifest_path"]
        if (
            sha256_file(validation_archive) != selection["validation"]["fixture_archive_sha256"]
            or sha256_file(validation_manifest) != selection["validation"]["private_manifest_sha256"]
        ):
            raise RuntimeError("OCR V18 stored validation fixture bytes changed")
        scenes = load_split_archive(validation_archive, validation_manifest, expected_split="validation")
        summary = proposal_summary(scenes)
        if (
            split_fingerprint(scenes) != selection["validation"]["split_fingerprint"]
            or any(summary[key] != selection["validation"][key] for key in summary)
        ):
            raise RuntimeError("OCR V18 stored validation fixtures violate the frozen split")

        phase = "cpu_execution"
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
            "schema": "graphreader.ocr-recognition-confirmed-selection-result.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "consumed": True,
            "optimizer_steps": 0,
            "provider": "CPUExecutionProvider",
            "detector_path": DETECTOR_PATH,
            "detector_sha256": DETECTOR_SHA256,
            "recognizer_path": RECOGNIZER_PATH,
            "recognizer_sha256": RECOGNIZER_SHA256,
            "selection_fixture_archive_sha256": selection["validation"]["fixture_archive_sha256"],
            "selection_manifest_sha256": sha256_file(selection_path),
            "metrics": metrics,
            "marker_creation_evaluated": False,
            "marker_creation_gate_required_before_production_approval": True,
            "case_level_details_emitted": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "production_approval": False,
            "release_eligible": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(
            authorization, status=str(report["status"]), report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-recognition-confirmed-selection-result.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "selection_gate_passed": False,
            "consumed": True,
            "optimizer_steps": 0,
            "failure_phase": phase,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "case_level_details_emitted": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "production_approval": False,
            "release_eligible": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(
            authorization, status="failed_runner", report_sha256=sha256_file(report_path),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = evaluate_candidate(REPO_ROOT / args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
