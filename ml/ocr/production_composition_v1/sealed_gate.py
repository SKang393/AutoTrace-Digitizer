# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use exact-model OCR production-composition public gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import onnxruntime as ort

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    require_committed_sources,
    sha256_file,
)
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_sealed_archive, proposal_summary, split_fingerprint
from .pipeline import DirectRunner, evaluate_scenes
from .protocol import (
    CHARACTER_ERROR_RATE_MAXIMUM,
    DETECTOR_ONNX_SHA256,
    EXACT_MATCH_MINIMUM,
    NUMERIC_RECOGNIZER_ONNX_SHA256,
    OFFICIAL_INFERENCE_YAML_SHA256,
    OFFICIAL_RECOGNIZER_ONNX_SHA256,
    PUBLIC_REVISION,
    REVISION,
    ROLE_ACCURACY_MINIMUM,
    TASK,
    VALIDATION_REVISION,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = Path("ml/ocr/production_composition_v1/gates/sealed-public-v1.json")
VALIDATION_REPORT_PATH = Path("ml/ocr/production_composition_v1/VALIDATION_REPORT.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/production_composition_v1/pipeline.py"),
    Path("ml/ocr/production_composition_v1/protocol.py"),
    Path("ml/ocr/production_composition_v1/sealed_gate.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_ensemble_v5/dataset.py"),
    Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    "evaluation_limit": 1,
    "exact_region_count_every_fixture": True,
    "false_region_count": 0,
    "missed_region_count": 0,
    "duplicate_region_count": 0,
    "prohibited_structure_hits": 0,
    "recognition_exact_match_minimum": EXACT_MATCH_MINIMUM,
    "character_error_rate_maximum": CHARACTER_ERROR_RATE_MAXIMUM,
    "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
    "forbidden_numeric_route_count": 0,
    "provider": "CPUExecutionProvider",
    "direct_fixture_byte_execution_required": True,
    "marker_creation_evidence_required_for_production_approval": True,
    "validation_gate_pass_required": True,
}


def _cpu_session(path: Path, input_name: str, output_name: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR composition gate requires CPUExecutionProvider only")
    if [value.name for value in session.get_inputs()] != [input_name] or [value.name for value in session.get_outputs()] != [output_name]:
        raise RuntimeError(f"OCR composition ONNX I/O changed for {path}")
    return session


def evaluate_composition(
    *,
    detector_path: Path,
    official_recognizer_path: Path,
    numeric_recognizer_path: Path,
    inference_yaml_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"OCR composition output already exists: {output_path}")
    require_committed_sources(REPO_ROOT, (*EVALUATOR_SOURCE_PATHS, VALIDATION_REPORT_PATH))
    validation_report = json.loads((REPO_ROOT / VALIDATION_REPORT_PATH).read_text(encoding="utf-8"))
    validation_report_sha256 = sha256_file(REPO_ROOT / VALIDATION_REPORT_PATH)
    if (
        validation_report.get("task") != TASK
        or validation_report.get("revision") != VALIDATION_REVISION
        or validation_report.get("status") != "pass"
        or validation_report.get("production_approval") is not False
        or validation_report.get("release_eligible") is not False
        or validation_report.get("payloads", {}).get("detector_onnx_sha256") != DETECTOR_ONNX_SHA256
        or validation_report.get("payloads", {}).get("official_recognizer_onnx_sha256")
        != OFFICIAL_RECOGNIZER_ONNX_SHA256
        or validation_report.get("payloads", {}).get("numeric_recognizer_onnx_sha256")
        != NUMERIC_RECOGNIZER_ONNX_SHA256
    ):
        raise RuntimeError("OCR composition public gate requires the exact passing committed validation report")
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / split_config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR composition public seal differs from the frozen gate configuration")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"]:
        raise RuntimeError("OCR composition fixture archive checksum mismatch")
    if sha256_file(private_manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("OCR composition private manifest checksum mismatch")
    if (
        sha256_file(detector_path) != DETECTOR_ONNX_SHA256
        or sha256_file(official_recognizer_path) != OFFICIAL_RECOGNIZER_ONNX_SHA256
        or sha256_file(numeric_recognizer_path) != NUMERIC_RECOGNIZER_ONNX_SHA256
        or sha256_file(inference_yaml_path) != OFFICIAL_INFERENCE_YAML_SHA256
    ):
        raise RuntimeError("OCR composition exact payload or preprocessing checksum changed")

    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes={
            "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
            "official_recognizer_onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
            "numeric_recognizer_onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
            "validation_report_sha256": validation_report_sha256,
        },
        dataset_manifest_sha256=seal["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    detector_session = _cpu_session(detector_path, "region_proposals", "region_logits")
    official_session = _cpu_session(official_recognizer_path, "x", "fetch_name_0")
    numeric_session = _cpu_session(numeric_recognizer_path, "glyphs", "logits")
    detector_runner = DirectRunner(detector_session, "region_proposals")
    official_runner = DirectRunner(official_session, "x")
    numeric_runner = DirectRunner(numeric_session, "glyphs")
    official_alphabet = read_character_alphabet(inference_yaml_path)
    started = time.perf_counter()
    scenes = load_sealed_archive(archive_path)
    summary = proposal_summary(scenes)
    if (
        summary["scene_count"] != seal["scene_count"]
        or summary["truth_region_count"] != seal["truth_region_count"]
        or summary["proposal_count"] != seal["proposal_count"]
        or summary["positive_proposal_count"] != seal["positive_proposal_count"]
        or summary["negative_proposal_count"] != seal["negative_proposal_count"]
        or split_fingerprint(scenes) != seal["split_fingerprint"]
    ):
        raise RuntimeError("OCR composition fixture bytes do not reproduce the frozen scene contract")
    metrics = evaluate_scenes(
        scenes,
        detector_runner,
        official_runner,
        numeric_runner,
        official_alphabet,
    )
    passed = (
        metrics["exact_detection_scene_count"] == metrics["scene_count"]
        and metrics["true_positives"] == metrics["truth_region_count"]
        and metrics["false_positives"] == 0
        and metrics["false_negatives"] == 0
        and metrics["duplicate_region_count"] == 0
        and metrics["prohibited_structure_hits"] == 0
        and metrics["recognition_exact_match"] >= EXACT_MATCH_MINIMUM
        and metrics["character_error_rate"] <= CHARACTER_ERROR_RATE_MAXIMUM
        and metrics["role_accuracy"] >= ROLE_ACCURACY_MINIMUM
        and metrics["forbidden_numeric_route_count"] == 0
        and metrics["marker_creation_evaluated"] is False
        and detector_runner.calls == len(scenes)
        and official_runner.calls == metrics["truth_region_count"]
        and numeric_runner.calls > 0
    )
    report: dict[str, object] = {
        "schema": "graphreader.ocr-production-composition-public-gate.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "status": "pass" if passed else "fail",
        "production_approval": False,
        "release_eligible": False,
        "evaluation_count": 1,
        "provider": "CPUExecutionProvider",
        "payloads": {
            "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
            "official_recognizer_onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
            "numeric_recognizer_onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
            "official_inference_yaml_sha256": OFFICIAL_INFERENCE_YAML_SHA256,
        },
        "fixture_archive_sha256": seal["fixture_archive_sha256"],
        "private_manifest_sha256": seal["private_manifest_sha256"],
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "validation_report_path": VALIDATION_REPORT_PATH.as_posix(),
        "validation_report_sha256": validation_report_sha256,
        "metrics": metrics,
        "direct_execution": {
            "detector": asdict(detector_runner.evidence()),
            "official_recognizer": asdict(official_runner.evidence()),
            "numeric_recognizer": asdict(numeric_runner.evidence()),
        },
        "gate_requirements": GATE_CONFIG,
        "remaining_mandatory_evidence": [
            "direct C# production-composition execution over the same fixture bytes",
            "independent marker-stage marker_creation_count equals zero for every fixture",
            "approved multi-payload model-store discovery and provider preflight",
            "packaging inventory, notices, forbidden-file scan, and clean-machine offline workflow",
        ],
        "seal_binding": gate.binding,
        "canonical_seal_key": gate.key,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(gate, status=str(report["status"]), report_sha256=sha256_file(output_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--official-recognizer", type=Path, required=True)
    parser.add_argument("--numeric-recognizer", type=Path, required=True)
    parser.add_argument("--inference-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_composition(
        detector_path=REPO_ROOT / arguments.detector,
        official_recognizer_path=REPO_ROOT / arguments.official_recognizer,
        numeric_recognizer_path=REPO_ROOT / arguments.numeric_recognizer,
        inference_yaml_path=REPO_ROOT / arguments.inference_yaml,
        output_path=REPO_ROOT / arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
