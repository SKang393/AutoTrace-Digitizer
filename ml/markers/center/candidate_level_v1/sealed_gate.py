# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use, source-bound sealed public gate for candidate-level marker detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.center.candidate_level_v1.dataset import load_sealed_public_archive
from ml.markers.center.candidate_level_v1.model import CandidateTensorContract
from ml.markers.center.candidate_level_v1.pipeline import evaluate_scenes
from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-candidate-level-public-v1"
SPLIT_CONFIG_PATH = Path(
    "ml/markers/center/candidate_level_v1/gates/sealed-public-v1.json"
)
EVALUATOR_SOURCE_PATHS = (
    Path("ml/markers/center/candidate_level_v1/dataset.py"),
    Path("ml/markers/center/candidate_level_v1/model.py"),
    Path("ml/markers/center/candidate_level_v1/pipeline.py"),
    Path("ml/markers/center/candidate_level_v1/sealed_gate.py"),
)
GATE_CONFIG = {
    "threshold_source": "selected candidate training report",
    "matching_tolerance_pixels": 5.0,
    "required_exact_scene_fraction": 1.0,
    "required_false_positives": 0,
    "required_false_negatives": 0,
    "required_duplicates": 0,
    "required_prohibited_structure_hits": 0,
    "provider": "CPUExecutionProvider",
}


def evaluate_candidate(*, onnx_path: Path, training_report_path: Path, output_path: Path) -> dict[str, object]:
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_report.get("status") != "selected" or training_report.get("selection_gate_passed") is not True:
        raise RuntimeError("Only a selection-passing candidate can open the sealed public gate")
    candidate_sha256 = sha256_file(onnx_path)
    if candidate_sha256 != training_report.get("onnx_sha256"):
        raise RuntimeError("Candidate ONNX differs from the selection report")
    seal_path = REPO_ROOT / split_config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("Sealed public test seal differs from the gate configuration")
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))
    fixture_archive = REPO_ROOT / sealed["fixture_archive_path"]
    if sha256_file(fixture_archive) != sealed["fixture_archive_sha256"]:
        raise RuntimeError("Sealed fixture archive checksum mismatch")
    private_manifest = REPO_ROOT / sealed["private_manifest_path"]
    if sha256_file(private_manifest) != sealed["private_manifest_sha256"]:
        raise RuntimeError("Sealed private fixture manifest checksum mismatch")

    gate_seal = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_hashes={"onnx_sha256": candidate_sha256},
        dataset_manifest_sha256=sealed["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    scenes = load_sealed_public_archive(fixture_archive)
    contract = CandidateTensorContract()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Sealed marker gate requires the CPU execution provider")

    def runner(value: np.ndarray) -> np.ndarray:
        return session.run(None, {contract.input_name: value})[0]

    threshold = float(training_report["selected_threshold"])
    metrics = evaluate_scenes(scenes, runner, threshold=threshold)
    passed = (
        metrics["exact_scene_count"] == metrics["scene_count"]
        and metrics["false_positives"] == 0
        and metrics["false_negatives"] == 0
        and metrics["duplicate_count"] == 0
        and metrics["prohibited_structure_hits"] == 0
    )
    report: dict[str, object] = {
        "schema": "graphreader.marker-center-candidate-sealed-public-gate.v1",
        "task": TASK,
        "revision": REVISION,
        "status": "pass" if passed else "fail",
        "release_eligible": False,
        "production_approval": False,
        "evaluation_count": 1,
        "candidate_onnx_path": onnx_path.as_posix(),
        "candidate_onnx_sha256": candidate_sha256,
        "training_report_path": training_report_path.as_posix(),
        "training_report_sha256": sha256_file(training_report_path),
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "fixture_archive_sha256": sealed["fixture_archive_sha256"],
        "private_manifest_sha256": sealed["private_manifest_sha256"],
        "threshold": threshold,
        "provider": "CPUExecutionProvider",
        "metrics": metrics,
        "gate_requirements": GATE_CONFIG,
        "seal_binding": gate_seal.binding,
        "canonical_seal_key": gate_seal.key,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "remaining_blockers": [
            "production runtime adapter",
            "model manifest and model-store approval",
            "provider compatibility evidence",
            "packaging discovery and clean-machine proof",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(
        gate_seal,
        status=str(report["status"]),
        report_sha256=sha256_file(output_path),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_candidate(
        onnx_path=REPO_ROOT / args.onnx,
        training_report_path=REPO_ROOT / args.training_report,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
