# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Zero-optimizer V23 candidate runner bound to the frozen V21 payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate, void_candidate

from . import protocol
from .diagnose_v23 import DEFAULT_ONNX, summarize


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = Path("ml/markers/center/multiradius_geometry_v23/candidate_config.json")
SOURCE_PATHS = (
    Path("ml/markers/center/multiradius_geometry_v23/protocol.py"),
    Path("ml/markers/center/multiradius_geometry_v23/diagnose_v23.py"),
    Path("ml/markers/center/multiradius_geometry_v23/candidate_runner.py"),
    Path("ml/markers/center/focal_confidence_v21/P1_RESULT.json"),
    Path("ml/markers/center/focal_confidence_v21/diagnostics/V21_DIAGNOSTIC.json"),
    Path("ml/markers/center/veto_override_v22/V22_FEASIBILITY_DIAGNOSTIC.json"),
    Path("ml/markers/center/proposal_geometry_v13/dataset.py"),
    Path("ml/markers/center/proposal_geometry_v13/geometry.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/policy/evidence_policy.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bindings(onnx_path: Path = DEFAULT_ONNX) -> None:
    """Fail closed if any frozen V21 input changed."""
    result_path = REPO_ROOT / protocol.V21_RESULT_PATH
    diagnostic_path = REPO_ROOT / protocol.V21_DIAGNOSTIC_PATH
    for path, expected, label in (
        (result_path, protocol.V21_RESULT_SHA256, "V21 result"),
        (diagnostic_path, protocol.V21_DIAGNOSTIC_SHA256, "V21 diagnostic"),
        (onnx_path, protocol.V21_ONNX_SHA256, "V21 ONNX"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{label} hash changed: expected {expected}, got {actual}")


def evaluate_candidate(onnx_path: Path = DEFAULT_ONNX) -> dict[str, Any]:
    """Evaluate V23 postprocessing without acquiring authorization."""
    validate_bindings(onnx_path)
    result = summarize(onnx_path)
    if result["scope"]["optimizer_steps"] != protocol.OPTIMIZER_STEPS:
        raise AssertionError("V23 runner must perform zero optimizer steps")
    if result["scope"]["training_performed"] is not protocol.TRAINING_PERFORMED:
        raise AssertionError("V23 runner must not train")
    return result


def run_candidate(output_dir: Path, onnx_path: Path = DEFAULT_ONNX) -> dict[str, Any]:
    """Run one authorized zero-optimizer synthetic dev candidate."""
    if output_dir.exists():
        raise RuntimeError(f"V23 output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=protocol.TASK,
        revision=protocol.REVISION,
        candidate_id=protocol.CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    try:
        result = evaluate_candidate(onnx_path)
        bars = config["acceptance_bar"]
        metrics = result["metrics"]
        passed = bool(
            metrics["precision"] >= bars["precision_minimum"]
            and metrics["recall"] >= bars["recall_minimum"]
            and metrics["prohibited_structure_hits"] <= bars["prohibited_hits_maximum"]
        )
        report = {
            **result,
            "schema": "graphreader.marker-center-multiradius-geometry-v23-candidate-result.v1",
            "task": protocol.TASK,
            "revision": protocol.REVISION,
            "candidate_id": protocol.CANDIDATE_ID,
            "status": "dev_passed" if passed else "failed_dev",
            "dev_gate_passed": passed,
            "training_authorization": authorization.binding,
            "v21_onnx_sha256": sha256_file(onnx_path),
            "model_license": config["model_license"],
            "production_approval": False,
            "release_eligible": False,
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(
            authorization,
            status=str(report["status"]),
            report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-multiradius-geometry-v23-candidate-failure.v1",
            "task": protocol.TASK,
            "revision": protocol.REVISION,
            "candidate_id": protocol.CANDIDATE_ID,
            "status": "failed_runner",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "sealed_runs": 0,
            "private_data": False,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        void_candidate(authorization, error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    args = parser.parse_args()
    report = run_candidate(args.output_dir.resolve(), args.onnx.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "dev_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
