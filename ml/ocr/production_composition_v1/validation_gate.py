# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use checksum-bound validation gate for OCR production composition."""

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
    ROLE_ACCURACY_MINIMUM,
    TASK,
    VALIDATION_REVISION,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = Path("ml/ocr/production_composition_v1/gates/validation-v1.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/production_composition_v1/pipeline.py"),
    Path("ml/ocr/production_composition_v1/protocol.py"),
    Path("ml/ocr/production_composition_v1/validation_gate.py"),
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
}


def _cpu_session(path: Path, input_name: str, output_name: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR composition validation requires CPUExecutionProvider only")
    if [value.name for value in session.get_inputs()] != [input_name] or [value.name for value in session.get_outputs()] != [output_name]:
        raise RuntimeError(f"OCR composition ONNX I/O changed for {path}")
    return session


def evaluate_validation(
    *,
    detector_path: Path,
    official_recognizer_path: Path,
    numeric_recognizer_path: Path,
    inference_yaml_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"OCR composition validation output already exists: {output_path}")
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / split_config["validation_seal_path"]
    if sha256_file(seal_path) != split_config["validation_seal_sha256"]:
        raise RuntimeError("OCR composition validation seal differs from frozen gate configuration")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"]:
        raise RuntimeError("OCR composition validation fixture checksum mismatch")
    if sha256_file(private_manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("OCR composition validation private manifest checksum mismatch")
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
        revision=VALIDATION_REVISION,
        candidate_hashes={
            "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
            "official_recognizer_onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
            "numeric_recognizer_onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
        },
        dataset_manifest_sha256=seal["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    detector_runner = DirectRunner(_cpu_session(detector_path, "region_proposals", "region_logits"), "region_proposals")
    official_runner = DirectRunner(_cpu_session(official_recognizer_path, "x", "fetch_name_0"), "x")
    numeric_runner = DirectRunner(_cpu_session(numeric_recognizer_path, "glyphs", "logits"), "glyphs")
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
        raise RuntimeError("OCR composition validation fixture bytes violate the frozen contract")
    metrics = evaluate_scenes(
        scenes,
        detector_runner,
        official_runner,
        numeric_runner,
        read_character_alphabet(inference_yaml_path),
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
        and detector_runner.calls == len(scenes)
        and official_runner.calls == metrics["truth_region_count"]
        and numeric_runner.calls > 0
    )
    report: dict[str, object] = {
        "schema": "graphreader.ocr-production-composition-validation-gate.v1",
        "task": TASK,
        "revision": VALIDATION_REVISION,
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
        "validation_seal_sha256": sha256_file(seal_path),
        "metrics": metrics,
        "direct_execution": {
            "detector": asdict(detector_runner.evidence()),
            "official_recognizer": asdict(official_runner.evidence()),
            "numeric_recognizer": asdict(numeric_runner.evidence()),
        },
        "gate_requirements": GATE_CONFIG,
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
    report = evaluate_validation(
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
