# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Zero-optimizer selection recovery for exact completed V7 P1 bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.center.domain_randomized_v7.dataset import read_archive
from ml.markers.center.dense_contract_v5.train_p1 import (
    PARITY_TOLERANCE,
    THRESHOLDS,
    _evaluate,
    _onnx_outputs,
    _passing_window,
    _torch_outputs,
)
from ml.markers.center.dense_contract_v5.train_p3 import (
    _fuse_inference_model,
    _maximum_output_difference,
)
from ml.markers.center.model import load_checkpoint
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/domain_randomized_v7"
TASK = "marker-center"
REVISION = "marker-center-domain-randomized-v7"
CANDIDATE_ID = "P2"
CONFIG_PATH = Path("ml/markers/center/domain_randomized_v7/training/p2.json")
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/markers/center/model.py"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/markers/center/postprocess.py"),
    Path("ml/markers/center/dense_contract_v5/dataset.py"),
    Path("ml/markers/center/dense_contract_v5/train_p1.py"),
    Path("ml/markers/center/dense_contract_v5/train_p3.py"),
    Path("ml/markers/center/feasible_dense_v6/dataset.py"),
    Path("ml/markers/center/domain_randomized_v7/dataset.py"),
    Path("ml/markers/center/domain_randomized_v7/model.py"),
    Path("ml/markers/center/domain_randomized_v7/P1_RESULT.json"),
    Path("ml/markers/center/domain_randomized_v7/train_p2.py"),
)


def _execute_candidate(
    authorization: TrainingAuthorization,
    progress: dict[str, object],
) -> dict[str, object]:
    started = float(progress["started"])
    progress["phase"] = "input_validation"
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    p1 = json.loads(P1_RESULT_PATH.read_text(encoding="utf-8"))
    if sha256_file(P1_RESULT_PATH) != config["p1_result_sha256"]:
        raise RuntimeError("Marker-center V7 P1 result identity changed")
    expected_p1 = {
        "status": "failed_runner_consumed",
        "optimizer_steps": 2304,
        "failure_phase": "selection",
        "checkpoint_sha256": config["p1_checkpoint_sha256"],
        "onnx_sha256": config["p1_onnx_sha256"],
        "public_gate_archive_opened": False,
        "public_gate_evaluations": 0,
    }
    for key, expected in expected_p1.items():
        if p1.get(key) != expected:
            raise RuntimeError(f"Marker-center V7 P1 recovery identity changed: {key}")
    checkpoint_path = REPO_ROOT / p1["checkpoint_path"]
    onnx_path = REPO_ROOT / p1["onnx_path"]
    if sha256_file(checkpoint_path) != p1["checkpoint_sha256"]:
        raise RuntimeError("Marker-center V7 P1 checkpoint changed")
    if sha256_file(onnx_path) != p1["onnx_sha256"]:
        raise RuntimeError("Marker-center V7 P1 ONNX changed")
    selection = json.loads((ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256_file(ROOT / "SELECTION_MANIFEST.json") != config["selection_manifest_sha256"]:
        raise RuntimeError("Marker-center V7 selection manifest changed")
    validation_path = REPO_ROOT / selection["validation"]["archive_path"]
    if sha256_file(validation_path) != selection["validation"]["archive_sha256"]:
        raise RuntimeError("Frozen marker-center V7 validation archive changed")
    validation = read_archive(validation_path)
    progress["phase"] = "selection"
    model, checkpoint_payload = load_checkpoint(checkpoint_path)
    if checkpoint_payload.get("training_revision") != REVISION:
        raise RuntimeError("Marker-center V7 P1 checkpoint revision changed")
    inference_model = _fuse_inference_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Marker-center V7 P2 selection requires CPUExecutionProvider only")
    onnx_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    parity = 0.0
    fusion_semantic_error = 0.0
    input_stream = hashlib.sha256()
    output_stream = hashlib.sha256()
    for index in range(validation["inputs"].shape[0]):
        value = validation["inputs"][index : index + 1]
        raw_outputs = _torch_outputs(model, value)
        fused_outputs = _torch_outputs(inference_model, value)
        onnx_values = _onnx_outputs(session, value)
        fusion_semantic_error = max(fusion_semantic_error, _maximum_output_difference(raw_outputs, fused_outputs))
        parity = max(parity, _maximum_output_difference(fused_outputs, onnx_values))
        onnx_first, onnx_second_input, onnx_second = onnx_values
        onnx_outputs.append((onnx_first, onnx_second))
        input_stream.update(value.tobytes(order="C"))
        input_stream.update(onnx_second_input.tobytes(order="C"))
        output_stream.update(onnx_first.tobytes(order="C"))
        output_stream.update(onnx_second.tobytes(order="C"))
    comparisons = [_evaluate(validation, onnx_outputs, threshold) for threshold in THRESHOLDS]
    window = _passing_window(comparisons)
    selected = max(
        comparisons,
        key=lambda item: (
            bool(item["passed"]),
            int(item["exact_scene_count"]),
            float(item["artifact_recall"]),
            float(item["artifact_precision"]),
            -abs(float(item["threshold"]) - 0.45),
        ),
    )
    selection_passed = len(window) >= 3 and parity <= PARITY_TOLERANCE
    return {
        "schema": "graphreader.marker-center-domain-randomized-candidate.v7",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "selection_passed" if selection_passed else "failed_selection",
        "selection_gate_passed": selection_passed,
        "isolated_change": config["isolated_change"],
        "private_data": False,
        "chandler_used": False,
        "synthetic_only": True,
        "provider": "CPUExecutionProvider",
        "optimizer_steps": 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "reused_completed_p1_optimizer_steps": p1["optimizer_steps"],
        "checkpoint_path": p1["checkpoint_path"],
        "checkpoint_sha256": p1["checkpoint_sha256"],
        "onnx_path": p1["onnx_path"],
        "onnx_sha256": p1["onnx_sha256"],
        "inference_graph_transform": "fuse_conv_batch_norm_eval_v1",
        "checkpoint_to_inference_graph_maximum_absolute_error": fusion_semantic_error,
        "onnx_parity_maximum_absolute_error": parity,
        "onnx_parity_tolerance": PARITY_TOLERANCE,
        "onnx_parity_passed": parity <= PARITY_TOLERANCE,
        "selected_threshold": selected["threshold"],
        "selection_metrics": selected,
        "passing_threshold_window": window,
        "threshold_aggregates": comparisons,
        "direct_execution_inference_calls": validation["inputs"].shape[0] * 2,
        "input_tensor_stream_sha256": input_stream.hexdigest(),
        "output_tensor_stream_sha256": output_stream.hexdigest(),
        "p1_result_sha256": sha256_file(P1_RESULT_PATH),
        "training_authorization": authorization.binding,
        "public_gate_archive_opened": False,
        "public_gate_evaluations": 0,
        "production_approval": False,
        "release_eligible": False,
    }


def run(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Marker-center V7 P2 output exists: {output_dir}")
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    progress: dict[str, object] = {"started": time.perf_counter(), "phase": "initialization"}
    try:
        report = _execute_candidate(authorization, progress)
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-domain-randomized-failure.v7",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": progress["phase"],
            "optimizer_steps": 0,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "production_approval": False,
            "release_eligible": False,
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/P2")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
