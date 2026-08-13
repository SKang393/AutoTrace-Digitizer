# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for composition V2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from ml.markers.gate_seal import (
    acquire_gate_seal, canonical_json_bytes, complete_gate_seal, require_committed_sources, sha256_file,
)
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_sealed_archive, proposal_summary, split_fingerprint
from .pipeline import DirectRunner, evaluate_scenes
from .protocol import (
    DETECTOR_ONNX_SHA256, NUMERIC_RECOGNIZER_ONNX_SHA256, OFFICIAL_INFERENCE_YAML_SHA256,
    OFFICIAL_RECOGNIZER_ONNX_SHA256, PUBLIC_REVISION, SPACING_SOURCE_SHA256, TASK, VALIDATION_REVISION,
)
from .validation_gate import GATE_CONFIG as METRIC_GATES, _cpu_session, _passes


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/production_composition_v2")
SPLIT_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
VALIDATION_REPORT_PATH = ROOT / "VALIDATION_REPORT.json"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "sealed_gate.py",
    ROOT / "validation_gate.py", Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_recall_detector_v9/dataset.py"),
    Path("ml/ocr/component_ensemble_v5/dataset.py"), Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/official_recognition_spacing_v2/spacing.py"), Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    **METRIC_GATES, "marker_creation_evidence_required_for_production_approval": True,
    "validation_gate_pass_required": True,
}


def evaluate_public(
    *, detector_path: Path, official_recognizer_path: Path, numeric_recognizer_path: Path,
    inference_yaml_path: Path, output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"OCR composition V2 public output exists: {output_path}")
    require_committed_sources(REPO_ROOT, (*EVALUATOR_SOURCE_PATHS, VALIDATION_REPORT_PATH))
    validation = json.loads((REPO_ROOT / VALIDATION_REPORT_PATH).read_text(encoding="utf-8"))
    validation_sha = sha256_file(REPO_ROOT / VALIDATION_REPORT_PATH)
    payloads = validation.get("payloads", {})
    if (
        validation.get("task") != TASK or validation.get("revision") != VALIDATION_REVISION
        or validation.get("status") != "pass" or validation.get("production_approval") is not False
        or validation.get("release_eligible") is not False
        or payloads.get("detector_onnx_sha256") != DETECTOR_ONNX_SHA256
        or payloads.get("official_recognizer_onnx_sha256") != OFFICIAL_RECOGNIZER_ONNX_SHA256
        or payloads.get("numeric_recognizer_onnx_sha256") != NUMERIC_RECOGNIZER_ONNX_SHA256
        or payloads.get("spacing_source_sha256") != SPACING_SOURCE_SHA256
    ):
        raise RuntimeError("OCR composition V2 public gate requires exact committed passing validation")
    config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR composition V2 public seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path, manifest_path = REPO_ROOT / seal["fixture_archive_path"], REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"] or sha256_file(manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("OCR composition V2 public fixture bytes changed")
    spacing_path = REPO_ROOT / "ml/ocr/official_recognition_spacing_v2/spacing.py"
    if not (
        sha256_file(detector_path) == DETECTOR_ONNX_SHA256
        and sha256_file(official_recognizer_path) == OFFICIAL_RECOGNIZER_ONNX_SHA256
        and sha256_file(numeric_recognizer_path) == NUMERIC_RECOGNIZER_ONNX_SHA256
        and sha256_file(inference_yaml_path) == OFFICIAL_INFERENCE_YAML_SHA256
        and sha256_file(spacing_path) == SPACING_SOURCE_SHA256
    ):
        raise RuntimeError("OCR composition V2 public exact payload or spacing source changed")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT, task=TASK, revision=PUBLIC_REVISION,
        candidate_hashes={
            "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
            "official_recognizer_onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
            "numeric_recognizer_onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
            "spacing_source_sha256": SPACING_SOURCE_SHA256, "validation_report_sha256": validation_sha,
        },
        dataset_manifest_sha256=seal["private_manifest_sha256"], split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS, gate_config=GATE_CONFIG,
    )
    detector = DirectRunner(_cpu_session(detector_path, "region_proposals", "region_logits"), "region_proposals")
    official = DirectRunner(_cpu_session(official_recognizer_path, "x", "fetch_name_0"), "x")
    numeric = DirectRunner(_cpu_session(numeric_recognizer_path, "glyphs", "logits"), "glyphs")
    started = time.perf_counter()
    scenes = load_sealed_archive(archive_path)
    summary = proposal_summary(scenes)
    if any(summary[key] != seal[key] for key in summary) or split_fingerprint(scenes) != seal["split_fingerprint"]:
        raise RuntimeError("OCR composition V2 public fixture bytes violate frozen contract")
    metrics = evaluate_scenes(scenes, detector, official, numeric, read_character_alphabet(inference_yaml_path))
    passed = _passes(metrics) and metrics["marker_creation_evaluated"] is False and detector.calls == len(scenes) and official.calls == metrics["truth_region_count"] and numeric.calls > 0
    report: dict[str, object] = {
        "schema": "graphreader.ocr-production-composition-public-gate.v2", "task": TASK,
        "revision": PUBLIC_REVISION, "status": "pass" if passed else "fail",
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
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "validation_report_path": VALIDATION_REPORT_PATH.as_posix(), "validation_report_sha256": validation_sha,
        "metrics": metrics,
        "direct_execution": {"detector": asdict(detector.evidence()), "official_recognizer": asdict(official.evidence()), "numeric_recognizer": asdict(numeric.evidence())},
        "gate_requirements": GATE_CONFIG,
        "remaining_mandatory_evidence": [
            "direct C# production-composition execution over checksum-bound fixture bytes",
            "independent marker-stage zero-marker evidence for OCR text regions",
            "approved multi-payload model-store discovery and provider preflight",
            "private Chandler workflow, packaging inventory, notices, and clean-machine offline evidence",
        ],
        "seal_binding": gate.binding, "canonical_seal_key": gate.key,
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
    report = evaluate_public(
        detector_path=REPO_ROOT / args.detector, official_recognizer_path=REPO_ROOT / args.official_recognizer,
        numeric_recognizer_path=REPO_ROOT / args.numeric_recognizer, inference_yaml_path=REPO_ROOT / args.inference_yaml,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
