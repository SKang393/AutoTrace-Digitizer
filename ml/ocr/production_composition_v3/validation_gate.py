# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use exact-payload validation gate for composition V3."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import onnxruntime as ort

from ml.markers.gate_seal import acquire_gate_seal, canonical_json_bytes, complete_gate_seal, sha256_file
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_sealed_archive, proposal_summary, split_fingerprint
from .pipeline import DirectRunner, evaluate_scenes
from .protocol import (
    AMBIGUITY_EXACT_MINIMUM, CHARACTER_ERROR_RATE_MAXIMUM, DETECTOR_ONNX_SHA256,
    EXACT_MATCH_MINIMUM, NUMERIC_EXACT_MINIMUM, NUMERIC_RECOGNIZER_ONNX_SHA256,
    OFFICIAL_INFERENCE_YAML_SHA256, OFFICIAL_RECOGNIZER_ONNX_SHA256, ROLE_ACCURACY_MINIMUM,
    SPACING_SOURCE_SHA256, TASK, VALIDATION_REVISION, WORD_EXACT_MINIMUM,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/production_composition_v3")
SPLIT_CONFIG_PATH = ROOT / "gates/validation-v1.json"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "validation_gate.py",
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/production_composition_v2/pipeline.py"),
    Path("ml/ocr/production_composition_v2/protocol.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_spaced_recall_detector_v10/dataset.py"),
    Path("ml/ocr/component_spaced_recall_detector_v10/protocol.py"),
    Path("ml/ocr/component_recall_detector_v9/dataset.py"),
    Path("ml/ocr/component_ensemble_v5/dataset.py"), Path("ml/ocr/component_ensemble_v5/protocol.py"),
    Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/official_recognition_spacing_v2/spacing.py"), Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    "evaluation_limit": 1, "exact_region_count_every_fixture": True,
    "false_region_count": 0, "missed_region_count": 0, "duplicate_region_count": 0,
    "prohibited_structure_hits": 0, "recognition_exact_match_minimum": EXACT_MATCH_MINIMUM,
    "character_error_rate_maximum": CHARACTER_ERROR_RATE_MAXIMUM,
    "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
    "numeric_exact_match_minimum": NUMERIC_EXACT_MINIMUM, "word_exact_match_minimum": WORD_EXACT_MINIMUM,
    "ambiguity_exact_match_minimum": AMBIGUITY_EXACT_MINIMUM,
    "spacing_changed_nonspace_truth_count": 0, "forbidden_numeric_route_count": 0,
    "provider": "CPUExecutionProvider", "direct_fixture_byte_execution_required": True,
}


def _cpu_session(path: Path, input_name: str, output_name: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR composition V3 requires CPUExecutionProvider only")
    if [item.name for item in session.get_inputs()] != [input_name] or [item.name for item in session.get_outputs()] != [output_name]:
        raise RuntimeError(f"OCR composition V3 ONNX I/O changed: {path}")
    return session


def _passes(metrics: dict[str, object]) -> bool:
    return bool(
        metrics["exact_detection_scene_count"] == metrics["scene_count"]
        and metrics["true_positives"] == metrics["truth_region_count"]
        and metrics["false_positives"] == 0 and metrics["false_negatives"] == 0
        and metrics["duplicate_region_count"] == 0 and metrics["prohibited_structure_hits"] == 0
        and metrics["recognition_exact_match"] >= EXACT_MATCH_MINIMUM
        and metrics["character_error_rate"] <= CHARACTER_ERROR_RATE_MAXIMUM
        and metrics["role_accuracy"] >= ROLE_ACCURACY_MINIMUM
        and metrics["numeric_exact_match"] >= NUMERIC_EXACT_MINIMUM
        and metrics["word_exact_match"] >= WORD_EXACT_MINIMUM
        and metrics["ambiguity_exact_match"] >= AMBIGUITY_EXACT_MINIMUM
        and metrics["spacing_changed_nonspace_truth_count"] == 0
        and metrics["forbidden_numeric_route_count"] == 0
    )


def evaluate_validation(
    *, detector_path: Path, official_recognizer_path: Path, numeric_recognizer_path: Path,
    inference_yaml_path: Path, output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"OCR composition V3 validation output exists: {output_path}")
    config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / config["validation_seal_path"]
    if sha256_file(seal_path) != config["validation_seal_sha256"]:
        raise RuntimeError("OCR composition V3 validation seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"] or sha256_file(manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("OCR composition V3 validation fixture bytes changed")
    spacing_path = REPO_ROOT / "ml/ocr/official_recognition_spacing_v2/spacing.py"
    hashes_match = (
        sha256_file(detector_path) == DETECTOR_ONNX_SHA256
        and sha256_file(official_recognizer_path) == OFFICIAL_RECOGNIZER_ONNX_SHA256
        and sha256_file(numeric_recognizer_path) == NUMERIC_RECOGNIZER_ONNX_SHA256
        and sha256_file(inference_yaml_path) == OFFICIAL_INFERENCE_YAML_SHA256
        and sha256_file(spacing_path) == SPACING_SOURCE_SHA256
    )
    if not hashes_match:
        raise RuntimeError("OCR composition V3 exact payload or spacing source changed")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT, task=TASK, revision=VALIDATION_REVISION,
        candidate_hashes={
            "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
            "official_recognizer_onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
            "numeric_recognizer_onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
            "spacing_source_sha256": SPACING_SOURCE_SHA256,
        },
        dataset_manifest_sha256=seal["private_manifest_sha256"], split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS, gate_config=GATE_CONFIG,
        evidence_split="dev",
    )
    detector = DirectRunner(_cpu_session(detector_path, "region_proposals", "region_logits"), "region_proposals")
    official = DirectRunner(_cpu_session(official_recognizer_path, "x", "fetch_name_0"), "x")
    numeric = DirectRunner(_cpu_session(numeric_recognizer_path, "glyphs", "logits"), "glyphs")
    started = time.perf_counter()
    scenes = load_sealed_archive(archive_path)
    summary = proposal_summary(scenes)
    if any(summary[key] != seal[key] for key in summary) or split_fingerprint(scenes) != seal["split_fingerprint"]:
        raise RuntimeError("OCR composition V3 validation fixtures violate frozen contract")
    metrics = evaluate_scenes(scenes, detector, official, numeric, read_character_alphabet(inference_yaml_path))
    passed = (
        _passes(metrics) and detector.calls == len(scenes)
        and official.calls == metrics["truth_region_count"] and numeric.calls > 0
    )
    report: dict[str, object] = {
        "schema": "graphreader.ocr-production-composition-validation-gate.v3", "task": TASK,
        "revision": VALIDATION_REVISION, "status": "pass" if passed else "fail",
        "production_approval": False, "release_eligible": False, "evaluation_count": 1,
        "provider": "CPUExecutionProvider",
        "payloads": {
            "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
            "official_recognizer_onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
            "numeric_recognizer_onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
            "official_inference_yaml_sha256": OFFICIAL_INFERENCE_YAML_SHA256,
            "spacing_source_sha256": SPACING_SOURCE_SHA256,
        },
        "fixture_archive_sha256": seal["fixture_archive_sha256"],
        "private_manifest_sha256": seal["private_manifest_sha256"],
        "validation_seal_sha256": sha256_file(seal_path), "metrics": metrics,
        "direct_execution": {
            "detector": asdict(detector.evidence()), "official_recognizer": asdict(official.evidence()),
            "numeric_recognizer": asdict(numeric.evidence()),
        },
        "gate_requirements": GATE_CONFIG, "seal_binding": gate.binding, "canonical_seal_key": gate.key,
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
    args = parser.parse_args()
    report = evaluate_validation(
        detector_path=REPO_ROOT / args.detector, official_recognizer_path=REPO_ROOT / args.official_recognizer,
        numeric_recognizer_path=REPO_ROOT / args.numeric_recognizer,
        inference_yaml_path=REPO_ROOT / args.inference_yaml, output_path=REPO_ROOT / args.output,
    )
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
