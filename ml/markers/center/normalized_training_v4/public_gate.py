# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for normalized-input marker-center P1."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.center.background_invariant_v3.pipeline import (
    MINIMUM_CENTER_SEPARATION,
    POSTPROCESS_REVISION,
    PREPROCESS_REVISION,
    evaluate_scenes,
)
from ml.markers.center.normalized_training_v4.dataset import load_sealed_public_archive
from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-normalized-training-public-v4-p1"
SPLIT_CONFIG_PATH = Path("ml/markers/center/normalized_training_v4/gates/sealed-public-p1.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/markers/center/normalized_training_v4/dataset.py"),
    Path("ml/markers/center/normalized_training_v4/public_gate.py"),
    Path("ml/markers/center/background_invariant_v3/pipeline.py"),
    Path("ml/markers/center/runtime_consistency_v2/pipeline_p2.py"),
    Path("ml/markers/center/radial_feature_v1/pipeline_p3.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
)
GATE_CONFIG = {
    "threshold_source": "selected candidate report from frozen threshold list",
    "allowed_thresholds": [0.15, 0.25, 0.35, 0.45, 0.6],
    "minimum_center_separation": MINIMUM_CENTER_SEPARATION,
    "preprocess_revision": PREPROCESS_REVISION,
    "postprocess_revision": POSTPROCESS_REVISION,
    "matching_tolerance_pixels": 5.0,
    "required_exact_scene_fraction": 1.0,
    "required_false_positives": 0,
    "required_false_negatives": 0,
    "required_duplicates": 0,
    "required_prohibited_structure_hits": 0,
    "provider": "CPUExecutionProvider",
}


def evaluate_candidate(*, onnx_path: Path, candidate_report_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"Public output already exists: {output_path}")
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if (
        candidate_report.get("candidate_id") != "P1"
        or candidate_report.get("status") != "selected"
        or candidate_report.get("selection_gate_passed") is not True
        or int(candidate_report.get("optimizer_steps", 0)) <= 0
        or candidate_report.get("weights_changed") is not True
        or candidate_report.get("preprocess_revision") != PREPROCESS_REVISION
        or candidate_report.get("postprocess_revision") != POSTPROCESS_REVISION
        or float(candidate_report.get("minimum_center_separation", -1.0)) != MINIMUM_CENTER_SEPARATION
    ):
        raise RuntimeError("Only the exact selected trained P1 may open this public gate")
    threshold = float(candidate_report.get("selected_threshold", -1.0))
    if threshold not in GATE_CONFIG["allowed_thresholds"]:
        raise RuntimeError("Candidate threshold is not in the frozen selection list")
    candidate_hash = sha256_file(onnx_path)
    report_hash = sha256_file(candidate_report_path)
    if candidate_hash != candidate_report.get("onnx_sha256"):
        raise RuntimeError("Candidate ONNX differs from its selection report")

    seal_path = REPO_ROOT / split_config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("Public-test seal differs from the gate configuration")
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))
    archive = REPO_ROOT / sealed["fixture_archive_path"]
    private_manifest = REPO_ROOT / sealed["private_manifest_path"]
    if sha256_file(archive) != sealed["fixture_archive_sha256"]:
        raise RuntimeError("Fixture archive checksum mismatch")
    if sha256_file(private_manifest) != sealed["private_manifest_sha256"]:
        raise RuntimeError("Private manifest checksum mismatch")

    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_hashes={
            "onnx_sha256": candidate_hash,
            "selection_report_sha256": report_hash,
        },
        dataset_manifest_sha256=sealed["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    try:
        scenes = load_sealed_public_archive(archive)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Public gate requires CPUExecutionProvider only")
        input_name = session.get_inputs()[0].name

        def runner(value: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: value.astype(np.float32, copy=False)})[0]

        metrics = evaluate_scenes(scenes, runner, threshold=threshold)
        passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == 0
            and metrics["false_negatives"] == 0
            and metrics["duplicate_count"] == 0
            and metrics["prohibited_structure_hits"] == 0
        )
        report: dict[str, object] = {
            "schema": "graphreader.marker-center-normalized-training-public-gate-p1.v4",
            "task": TASK,
            "revision": REVISION,
            "status": "pass" if passed else "fail",
            "release_eligible": False,
            "production_approval": False,
            "evaluation_count": 1,
            "candidate_onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "candidate_onnx_sha256": candidate_hash,
            "candidate_report_path": candidate_report_path.relative_to(REPO_ROOT).as_posix(),
            "candidate_report_sha256": report_hash,
            "sealed_public_test_seal_sha256": sha256_file(seal_path),
            "fixture_archive_sha256": sealed["fixture_archive_sha256"],
            "private_manifest_sha256": sealed["private_manifest_sha256"],
            "threshold": threshold,
            "minimum_center_separation": MINIMUM_CENTER_SEPARATION,
            "preprocess_revision": PREPROCESS_REVISION,
            "postprocess_revision": POSTPROCESS_REVISION,
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
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-normalized-training-public-failure-p1.v4",
            "task": TASK,
            "revision": REVISION,
            "status": "failed_runner",
            "release_eligible": False,
            "production_approval": False,
            "evaluation_count": 1,
            "candidate_onnx_sha256": candidate_hash,
            "candidate_report_sha256": report_hash,
            "sealed_public_test_seal_sha256": sha256_file(seal_path),
            "fixture_archive_sha256": sealed["fixture_archive_sha256"],
            "private_manifest_sha256": sealed["private_manifest_sha256"],
            "preprocess_revision": PREPROCESS_REVISION,
            "provider": "CPUExecutionProvider",
            "seal_binding": gate.binding,
            "canonical_seal_key": gate.key,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(canonical_json_bytes(failure))
        complete_gate_seal(gate, status="failed_runner", report_sha256=sha256_file(output_path))
        raise


__all__ = ["EVALUATOR_SOURCE_PATHS", "GATE_CONFIG", "REVISION", "evaluate_candidate"]
