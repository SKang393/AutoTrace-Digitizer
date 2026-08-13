# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for composition V8."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

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
from .protocol import *
from .validation_gate import EVALUATOR_SOURCE_PATHS as VALIDATION_SOURCES, GATE_CONFIG as METRIC_GATES, _cpu_session, _passes


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/production_composition_v8")
SPLIT_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
VALIDATION_REPORT_PATH = ROOT / "VALIDATION_REPORT.json"
EVALUATOR_SOURCE_PATHS = tuple(
    path for path in VALIDATION_SOURCES if path != ROOT / "validation_gate.py"
) + (ROOT / "validation_gate.py", ROOT / "sealed_gate.py")
GATE_CONFIG = {
    **METRIC_GATES,
    "validation_gate_pass_required": True,
    "marker_creation_evidence_required_for_production_approval": True,
}


def evaluate_public(*, detector_path: Path, official_path: Path, numeric_path: Path,
                    ambiguity_path: Path, inference_yaml_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"OCR composition V8 public output exists: {output_path}")
    require_committed_sources(REPO_ROOT, (*EVALUATOR_SOURCE_PATHS, VALIDATION_REPORT_PATH))
    validation = json.loads((REPO_ROOT / VALIDATION_REPORT_PATH).read_text(encoding="utf-8"))
    validation_sha = sha256_file(REPO_ROOT / VALIDATION_REPORT_PATH)
    expected = {
        "detector_onnx_sha256": DETECTOR_ONNX_SHA256,
        "official_recognizer_onnx_sha256": OFFICIAL_RECOGNIZER_ONNX_SHA256,
        "numeric_recognizer_onnx_sha256": NUMERIC_RECOGNIZER_ONNX_SHA256,
        "ambiguity_recognizer_onnx_sha256": AMBIGUITY_RECOGNIZER_ONNX_SHA256,
        "spacing_source_sha256": SPACING_SOURCE_SHA256,
    }
    if validation.get("status") != "pass" or validation.get("payloads", {}) != expected | {
        "official_inference_yaml_sha256": OFFICIAL_INFERENCE_YAML_SHA256
    }:
        raise RuntimeError("OCR composition V8 public gate requires exact passing validation")
    config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("V8 public seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive = REPO_ROOT / seal["fixture_archive_path"]
    manifest = REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive) != seal["fixture_archive_sha256"] or sha256_file(manifest) != seal["private_manifest_sha256"]:
        raise RuntimeError("V8 public fixtures changed")
    actual = {
        "detector_onnx_sha256": sha256_file(detector_path),
        "official_recognizer_onnx_sha256": sha256_file(official_path),
        "numeric_recognizer_onnx_sha256": sha256_file(numeric_path),
        "ambiguity_recognizer_onnx_sha256": sha256_file(ambiguity_path),
        "spacing_source_sha256": sha256_file(REPO_ROOT / "ml/ocr/official_recognition_spacing_v3/spacing.py"),
        "validation_report_sha256": validation_sha,
    }
    if {key: actual[key] for key in expected} != expected or sha256_file(inference_yaml_path) != OFFICIAL_INFERENCE_YAML_SHA256:
        raise RuntimeError("V8 public payload changed")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT, task=TASK, revision=PUBLIC_REVISION, candidate_hashes=actual,
        dataset_manifest_sha256=seal["private_manifest_sha256"], split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS, gate_config=GATE_CONFIG,
    )
    runners = (
        DirectRunner(_cpu_session(detector_path, "region_proposals", "region_logits"), "region_proposals"),
        DirectRunner(_cpu_session(official_path, "x", "fetch_name_0"), "x"),
        DirectRunner(_cpu_session(numeric_path, "glyphs", "logits"), "glyphs"),
        DirectRunner(_cpu_session(ambiguity_path, "glyphs", "logits"), "glyphs"),
    )
    started = time.perf_counter()
    scenes = load_sealed_archive(archive)
    summary = proposal_summary(scenes)
    if any(summary[key] != seal[key] for key in summary) or split_fingerprint(scenes) != seal["split_fingerprint"]:
        raise RuntimeError("V8 public split changed")
    metrics = evaluate_scenes(scenes, *runners, read_character_alphabet(inference_yaml_path))
    passed = (
        _passes(metrics)
        and metrics["marker_creation_evaluated"] is False
        and runners[0].calls == len(scenes)
        and runners[1].calls >= metrics["true_positives"]
        and runners[2].calls > 0
        and runners[3].calls > 0
    )
    report = {
        "schema": "graphreader.ocr-production-composition-public-gate.v8",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "status": "pass" if passed else "fail",
        "evaluation_count": 1,
        "production_approval": False,
        "release_eligible": False,
        "provider": "CPUExecutionProvider",
        "payloads": expected | {"official_inference_yaml_sha256": OFFICIAL_INFERENCE_YAML_SHA256},
        "fixture_archive_sha256": seal["fixture_archive_sha256"],
        "private_manifest_sha256": seal["private_manifest_sha256"],
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "validation_report_path": VALIDATION_REPORT_PATH.as_posix(),
        "validation_report_sha256": validation_sha,
        "metrics": metrics,
        "direct_execution": {
            name: asdict(runner.evidence())
            for name, runner in zip(
                ("detector", "official_recognizer", "numeric_recognizer", "ambiguity_recognizer"),
                runners,
                strict=True,
            )
        },
        "gate_requirements": GATE_CONFIG,
        "remaining_mandatory_evidence": [
            "direct C# production composition",
            "marker-stage exclusion evidence",
            "model-store and packaging discovery",
            "private Chandler and clean-machine offline evidence",
        ],
        "seal_binding": gate.binding,
        "canonical_seal_key": gate.key,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(gate, status=report["status"], report_sha256=sha256_file(output_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("detector", "official", "numeric", "ambiguity", "inference-yaml", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_public(
        detector_path=REPO_ROOT / args.detector,
        official_path=REPO_ROOT / args.official,
        numeric_path=REPO_ROOT / args.numeric,
        ambiguity_path=REPO_ROOT / args.ambiguity,
        inference_yaml_path=REPO_ROOT / args.inference_yaml,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
