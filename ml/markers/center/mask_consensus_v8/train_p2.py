# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer P2 artifact-threshold calibration for marker-center V8."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import onnxruntime as ort

from ml.markers.center.dense_contract_v5.dataset import KIND_TO_INDEX, PROHIBITED_KINDS
from ml.markers.center.dense_contract_v5.train_p1 import (
    _onnx_outputs,
    _passing_window,
    _torch_outputs,
)
from ml.markers.center.dense_contract_v5.train_p3 import (
    _fuse_inference_model,
    _maximum_output_difference,
)
from ml.markers.center.mask_consensus_v8.dataset import read_archive
from ml.markers.center.mask_consensus_v8.model import load_checkpoint
from ml.markers.center.mask_consensus_v8.protocol import (
    ONNX_PARITY_TOLERANCE,
    REVISION,
    TASK,
    THRESHOLDS,
)
from ml.markers.center.metrics import aggregate_scene_metrics, center_metrics
from ml.markers.center.postprocess import detect_heads
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    TrainingAuthorization,
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/mask_consensus_v8"
CANDIDATE_ID = "P2"
CONFIG_PATH = Path("ml/markers/center/mask_consensus_v8/training/p2.json")
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
    Path("ml/markers/center/mask_consensus_v8/dataset.py"),
    Path("ml/markers/center/mask_consensus_v8/model.py"),
    Path("ml/markers/center/mask_consensus_v8/protocol.py"),
    Path("ml/markers/center/mask_consensus_v8/train_p2.py"),
)
ARTIFACT_THRESHOLD = 0.45
MATCH_TOLERANCE = 5.0
HARD_NEGATIVE_TOLERANCE = 6.0


def _centers(archive: dict[str, np.ndarray], index: int) -> tuple[tuple[float, float], ...]:
    count = int(archive["center_counts"][index])
    return tuple(
        (float(row[0]), float(row[1]))
        for row in archive["centers"][index, :count]
    )


def _hard_negatives(
    archive: dict[str, np.ndarray],
    index: int,
) -> tuple[tuple[str, float, float], ...]:
    count = int(archive["hard_counts"][index])
    reverse = {value: key for key, value in KIND_TO_INDEX.items()}
    return tuple(
        (
            reverse[int(archive["hard_kinds"][index, ordinal])],
            float(archive["hard_points"][index, ordinal, 0]),
            float(archive["hard_points"][index, ordinal, 1]),
        )
        for ordinal in range(count)
    )


def _evaluate(
    archive: dict[str, np.ndarray],
    outputs: list[tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, object]:
    scene_metrics = []
    exact_scene_count = 0
    prohibited = {kind: 0 for kind in PROHIBITED_KINDS}
    artifact_intersection = 0
    artifact_predicted = 0
    artifact_truth = 0
    marker_artifact_hits = 0
    for index, (first_output, second_output) in enumerate(outputs):
        first_input = archive["inputs"][index : index + 1]
        combined_artifact = np.maximum(first_input[0, 2], first_output[0, 2])
        detections = detect_heads(
            second_output,
            text_mask=first_input[0, 1],
            artifact_mask=combined_artifact,
            center_threshold=threshold,
            artifact_threshold=ARTIFACT_THRESHOLD,
        )
        metric = center_metrics(detections, _centers(archive, index), MATCH_TOLERANCE)
        scene_metrics.append(metric)
        if metric.false_positives == 0 and metric.false_negatives == 0 and metric.duplicate_count == 0:
            exact_scene_count += 1
        for kind, x, y in _hard_negatives(archive, index):
            prohibited[kind] += sum(
                math.hypot(item.x - x, item.y - y) <= HARD_NEGATIVE_TOLERANCE
                for item in detections
            )
        predicted_mask = first_output[0, 2] >= ARTIFACT_THRESHOLD
        truth_mask = archive["artifact_targets"][index, 0] >= 0.5
        artifact_intersection += int(np.logical_and(predicted_mask, truth_mask).sum())
        artifact_predicted += int(predicted_mask.sum())
        artifact_truth += int(truth_mask.sum())
        for x, y in _centers(archive, index):
            marker_artifact_hits += int(first_output[0, 2, round(y), round(x)] >= ARTIFACT_THRESHOLD)
    aggregate = aggregate_scene_metrics(scene_metrics, MATCH_TOLERANCE)
    artifact_precision = artifact_intersection / artifact_predicted if artifact_predicted else 0.0
    artifact_recall = artifact_intersection / artifact_truth if artifact_truth else 1.0
    passed = (
        exact_scene_count == len(outputs)
        and aggregate.false_positives == 0
        and aggregate.false_negatives == 0
        and aggregate.duplicate_count == 0
        and not any(prohibited.values())
        and artifact_precision >= 0.90
        and artifact_recall >= 0.95
        and marker_artifact_hits == 0
    )
    return {
        "threshold": threshold,
        "passed": passed,
        "scene_count": len(outputs),
        "exact_scene_count": exact_scene_count,
        "true_positives": aggregate.true_positives,
        "false_positives": aggregate.false_positives,
        "false_negatives": aggregate.false_negatives,
        "duplicate_count": aggregate.duplicate_count,
        "prohibited_structure_hits": prohibited,
        "artifact_precision": artifact_precision,
        "artifact_recall": artifact_recall,
        "marker_artifact_hits": marker_artifact_hits,
    }


def _execute_candidate(
    output_dir: Path,
    authorization: TrainingAuthorization,
    progress: dict[str, object],
) -> dict[str, object]:
    started = float(progress["started"])
    progress["phase"] = "input_validation"
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    expected = {
        "artifact_threshold": ARTIFACT_THRESHOLD,
        "expected_optimizer_steps": 0,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "selection_thresholds": list(THRESHOLDS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V8 P2 configuration changed: {key}")
    p1_result_path = ROOT / "P1_RESULT.json"
    if sha256_file(p1_result_path) != config["p1_result_sha256"]:
        raise RuntimeError("Marker-center V8 P1 result changed")
    p1_result = json.loads(p1_result_path.read_text(encoding="utf-8"))
    if p1_result.get("status") != "failed_selection_consumed":
        raise RuntimeError("Marker-center V8 P1 is not a consumed selection failure")
    report_path = REPO_ROOT / p1_result["candidate_report_path"]
    checkpoint_source = REPO_ROOT / p1_result["checkpoint_path"]
    onnx_source = REPO_ROOT / p1_result["onnx_path"]
    for path, expected_sha256 in (
        (report_path, config["p1_candidate_report_sha256"]),
        (checkpoint_source, config["p1_checkpoint_sha256"]),
        (onnx_source, config["p1_onnx_sha256"]),
    ):
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Marker-center V8 P1 evidence changed: {path.name}")
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    if sha256_file(selection_path) != config["selection_manifest_sha256"]:
        raise RuntimeError("Marker-center V8 selection manifest changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validation_path = REPO_ROOT / selection["validation"]["archive_path"]
    if sha256_file(validation_path) != selection["validation"]["archive_sha256"]:
        raise RuntimeError("Frozen marker-center V8 validation archive changed")
    validation = read_archive(validation_path)
    model, _ = load_checkpoint(checkpoint_source)
    inference_model = _fuse_inference_model(model)
    checkpoint_path = output_dir / "marker-center-mask-consensus-v8-p2.pt"
    onnx_path = output_dir / "marker-center-mask-consensus-v8-p2.onnx"
    shutil.copyfile(checkpoint_source, checkpoint_path)
    shutil.copyfile(onnx_source, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Marker-center V8 P2 selection requires CPUExecutionProvider only")
    progress["phase"] = "selection"
    onnx_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    parity = 0.0
    inference_semantic_error = 0.0
    input_stream = hashlib.sha256()
    output_stream = hashlib.sha256()
    for index in range(validation["inputs"].shape[0]):
        value = validation["inputs"][index : index + 1]
        raw_outputs = _torch_outputs(model, value)
        frozen_outputs = _torch_outputs(inference_model, value)
        onnx_values = _onnx_outputs(session, value)
        inference_semantic_error = max(
            inference_semantic_error,
            _maximum_output_difference(raw_outputs, frozen_outputs),
        )
        parity = max(parity, _maximum_output_difference(frozen_outputs, onnx_values))
        first, second_input, second = onnx_values
        onnx_outputs.append((first, second))
        input_stream.update(value.tobytes(order="C"))
        input_stream.update(second_input.tobytes(order="C"))
        output_stream.update(first.tobytes(order="C"))
        output_stream.update(second.tobytes(order="C"))
    comparisons = [_evaluate(validation, onnx_outputs, threshold) for threshold in THRESHOLDS]
    window = _passing_window(comparisons)
    selected = max(
        comparisons,
        key=lambda item: (
            bool(item["passed"]),
            int(item["exact_scene_count"]),
            float(item["artifact_precision"]),
            float(item["artifact_recall"]),
            -abs(float(item["threshold"]) - 0.45),
        ),
    )
    selection_passed = len(window) >= 3 and parity <= ONNX_PARITY_TOLERANCE
    return {
        "schema": "graphreader.marker-center-mask-consensus-candidate.v8",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "selection_passed" if selection_passed else "failed_selection",
        "selection_gate_passed": selection_passed,
        "isolated_change": config["isolated_change"],
        "aggregate_design_basis": config["aggregate_design_basis"],
        "private_data": False,
        "chandler_used": False,
        "synthetic_only": True,
        "provider": "CPUExecutionProvider",
        "optimizer_steps": 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "inference_graph_transform": "exact_p1_graph_reuse_v1",
        "checkpoint_to_inference_graph_maximum_absolute_error": inference_semantic_error,
        "onnx_parity_maximum_absolute_error": parity,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "onnx_parity_passed": parity <= ONNX_PARITY_TOLERANCE,
        "artifact_threshold": ARTIFACT_THRESHOLD,
        "selected_threshold": selected["threshold"],
        "selection_metrics": selected,
        "passing_threshold_window": window,
        "threshold_aggregates": comparisons,
        "direct_execution_inference_calls": validation["inputs"].shape[0] * 2,
        "input_tensor_stream_sha256": input_stream.hexdigest(),
        "output_tensor_stream_sha256": output_stream.hexdigest(),
        "training_authorization": authorization.binding,
        "public_gate_archive_opened": False,
        "public_gate_evaluations": 0,
        "manifest_created": False,
        "model_store_promoted": False,
        "packaging_discovery": False,
        "private_validation": False,
        "production_approval": False,
        "release_eligible": False,
    }


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Marker-center V8 P2 output exists: {output_dir}")
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
    progress: dict[str, object] = {
        "started": time.perf_counter(),
        "phase": "initialization",
        "optimizer_steps": 0,
    }
    try:
        report = _execute_candidate(output_dir, authorization, progress)
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-mask-consensus-failure.v8",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": progress["phase"],
            "optimizer_steps": progress["optimizer_steps"],
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
        complete_training_candidate(
            authorization,
            status="failed_runner",
            report_sha256=sha256_file(report_path),
        )
        raise
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(
        authorization,
        status=str(report["status"]),
        report_sha256=sha256_file(report_path),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/P2")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
