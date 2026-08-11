# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for the runtime-consistent marker-center payload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.center.line_aware_v1.model import LineAwareTensorContract
from ml.markers.center.runtime_consistency_v2.dataset import load_sealed_public_archive
from ml.markers.center.runtime_consistency_v2.pipeline import (
    POSTPROCESS_REVISION,
    evaluate_scenes,
)
from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-runtime-consistency-public-v2"
SPLIT_CONFIG_PATH = Path(
    "ml/markers/center/runtime_consistency_v2/gates/sealed-public-v2.json"
)
EVALUATOR_SOURCE_PATHS = (
    Path("ml/markers/center/runtime_consistency_v2/dataset.py"),
    Path("ml/markers/center/runtime_consistency_v2/pipeline.py"),
    Path("ml/markers/center/runtime_consistency_v2/public_gate.py"),
    Path("ml/markers/center/radial_feature_v1/pipeline_p3.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
)
GATE_CONFIG = {
    "threshold": 0.3,
    "postprocess_revision": POSTPROCESS_REVISION,
    "matching_tolerance_pixels": 5.0,
    "required_exact_scene_fraction": 1.0,
    "required_false_positives": 0,
    "required_false_negatives": 0,
    "required_duplicates": 0,
    "required_prohibited_structure_hits": 0,
    "provider": "CPUExecutionProvider",
}


def evaluate_candidate(
    *,
    onnx_path: Path,
    candidate_report_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"Public output already exists: {output_path}")
    split_config = json.loads(
        (REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8")
    )
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if (
        candidate_report.get("status") != "selected"
        or candidate_report.get("selection_gate_passed") is not True
        or candidate_report.get("optimizer_steps") != 0
        or candidate_report.get("weights_changed") is not False
        or candidate_report.get("postprocess_revision") != POSTPROCESS_REVISION
    ):
        raise RuntimeError(
            "Only the exact zero-training runtime-consistent selection winner may open the public gate"
        )
    candidate_hash = sha256_file(onnx_path)
    if candidate_hash != candidate_report.get("onnx_sha256"):
        raise RuntimeError("Candidate ONNX differs from its selection report")
    if float(candidate_report.get("selected_threshold", -1.0)) != GATE_CONFIG["threshold"]:
        raise RuntimeError("Candidate threshold differs from the frozen public gate")

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
        candidate_hashes={"onnx_sha256": candidate_hash},
        dataset_manifest_sha256=sealed["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    try:
        scenes = load_sealed_public_archive(archive)
        contract = LineAwareTensorContract(
            runtime_revision="marker-center-radial-feature-runtime-v1"
        )
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Public gate requires CPUExecutionProvider only")

        def runner(value: np.ndarray) -> np.ndarray:
            return session.run(None, {contract.input_name: value})[0]

        metrics = evaluate_scenes(scenes, runner, threshold=GATE_CONFIG["threshold"])
        passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == 0
            and metrics["false_negatives"] == 0
            and metrics["duplicate_count"] == 0
            and metrics["prohibited_structure_hits"] == 0
        )
        report: dict[str, object] = {
            "schema": "graphreader.marker-center-runtime-consistency-public-gate.v2",
            "task": TASK,
            "revision": REVISION,
            "status": "pass" if passed else "fail",
            "release_eligible": False,
            "production_approval": False,
            "evaluation_count": 1,
            "candidate_onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "candidate_onnx_sha256": candidate_hash,
            "candidate_report_path": candidate_report_path.relative_to(REPO_ROOT).as_posix(),
            "candidate_report_sha256": sha256_file(candidate_report_path),
            "sealed_public_test_seal_sha256": sha256_file(seal_path),
            "fixture_archive_sha256": sealed["fixture_archive_sha256"],
            "private_manifest_sha256": sealed["private_manifest_sha256"],
            "threshold": GATE_CONFIG["threshold"],
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
        complete_gate_seal(
            gate,
            status=str(report["status"]),
            report_sha256=sha256_file(output_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-runtime-consistency-public-failure.v2",
            "task": TASK,
            "revision": REVISION,
            "status": "failed_runner",
            "release_eligible": False,
            "production_approval": False,
            "evaluation_count": 1,
            "candidate_onnx_sha256": candidate_hash,
            "candidate_report_sha256": sha256_file(candidate_report_path),
            "sealed_public_test_seal_sha256": sha256_file(seal_path),
            "fixture_archive_sha256": sealed["fixture_archive_sha256"],
            "private_manifest_sha256": sealed["private_manifest_sha256"],
            "threshold": GATE_CONFIG["threshold"],
            "postprocess_revision": POSTPROCESS_REVISION,
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
        complete_gate_seal(
            gate,
            status="failed_runner",
            report_sha256=sha256_file(output_path),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_candidate(
        onnx_path=REPO_ROOT / args.onnx,
        candidate_report_path=REPO_ROOT / args.candidate_report,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
