# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V9 recovery candidate on fresh splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

import ml.markers.center.mask_consensus_v8.train_p3 as parent
from ml.markers.center.mask_consensus_v9.protocol import (
    ONNX_PARITY_TOLERANCE,
    PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
    REVISION,
    ROOT as RELATIVE_ROOT,
    TASK,
    THRESHOLDS,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / RELATIVE_ROOT
PARENT_ROOT = REPO_ROOT / "ml/markers/center/mask_consensus_v8"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/mask_consensus_v9/training/p1.json")
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
    Path("ml/markers/center/mask_consensus_v8/train_p3.py"),
    Path("ml/markers/center/mask_consensus_v9/dataset.py"),
    Path("ml/markers/center/mask_consensus_v9/protocol.py"),
    Path("ml/markers/center/mask_consensus_v9/train_p1.py"),
)


def _parity_reproduction_within_tolerance(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
) -> bool:
    return all(
        abs(left - right) <= PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE
        for left, right in zip(actual, expected, strict=True)
    )


def _verify_config_and_inputs(config: dict[str, object]) -> tuple[dict[str, object], Path, Path]:
    expected_values = {
        "artifact_loss_weight": parent.ARTIFACT_LOSS_WEIGHT,
        "artifact_positive_weight": parent.ARTIFACT_POSITIVE_WEIGHT,
        "batch_size": parent.BATCH_SIZE,
        "center_loss_weight": parent.CENTER_LOSS_WEIGHT,
        "epochs": parent.EPOCHS,
        "expected_optimizer_steps": parent.EXPECTED_OPTIMIZER_STEPS,
        "fixed_radius_pixels": parent.FIXED_RADIUS_PIXELS,
        "hard_negative_margin_loss_weight": parent.HARD_NEGATIVE_MARGIN_LOSS_WEIGHT,
        "learning_rate": parent.LEARNING_RATE,
        "marker_clear_loss_weight": parent.MARKER_CLEAR_LOSS_WEIGHT,
        "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
        "positive_margin_loss_weight": parent.POSITIVE_MARGIN_LOSS_WEIGHT,
        "predecessor_parity_reproduction_tolerance": PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
        "selection_thresholds": list(THRESHOLDS),
    }
    for key, value in expected_values.items():
        if config.get(key) != value:
            raise RuntimeError(f"Marker-center V9 P1 configuration changed: {key}")
    trigger_path = REPO_ROOT / TRIGGER_RESULT_PATH
    if sha256_file(trigger_path) != TRIGGER_RESULT_SHA256:
        raise RuntimeError("Marker-center V8 P3 trigger result changed")
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    if trigger.get("status") != "failed_runner_consumed" or trigger.get("optimizer_steps") != 0:
        raise RuntimeError("Marker-center V9 requires the consumed zero-step V8 P3 runner failure")
    predecessor_path = PARENT_ROOT / "P2_RESULT.json"
    if sha256_file(predecessor_path) != config["predecessor_result_sha256"]:
        raise RuntimeError("Marker-center V9 predecessor result changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("status") != "failed_selection_consumed":
        raise RuntimeError("Marker-center V9 predecessor is not the consumed V8 P2 result")
    checkpoint_path = REPO_ROOT / str(predecessor["checkpoint_path"])
    onnx_path = REPO_ROOT / str(predecessor["onnx_path"])
    for path, key in (
        (checkpoint_path, "predecessor_checkpoint_sha256"),
        (onnx_path, "predecessor_onnx_sha256"),
    ):
        if sha256_file(path) != config[key]:
            raise RuntimeError(f"Marker-center V9 predecessor evidence changed: {path.name}")
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    if sha256_file(selection_path) != config["selection_manifest_sha256"]:
        raise RuntimeError("Marker-center V9 selection manifest changed")
    predecessor_validation_path = REPO_ROOT / str(config["predecessor_validation_archive_path"])
    if sha256_file(predecessor_validation_path) != config["predecessor_validation_archive_sha256"]:
        raise RuntimeError("Marker-center V9 predecessor validation archive changed")
    return predecessor, checkpoint_path, onnx_path


def _execute_candidate(
    output_dir: Path,
    authorization: object,
    progress: dict[str, object],
) -> dict[str, object]:
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    expected = tuple(float(value) for value in config["predecessor_parity_by_output_channel"])
    predecessor_validation_path = REPO_ROOT / str(config["predecessor_validation_archive_path"])
    if sha256_file(predecessor_validation_path) != config["predecessor_validation_archive_sha256"]:
        raise RuntimeError("Marker-center V9 predecessor validation archive changed")
    predecessor_validation = parent.read_archive(predecessor_validation_path)
    observed: dict[str, tuple[float, float, float]] = {}
    original_values = {
        "ROOT": parent.ROOT,
        "REVISION": parent.REVISION,
        "CANDIDATE_ID": parent.CANDIDATE_ID,
        "CONFIG_PATH": parent.CONFIG_PATH,
        "RUNNER_SOURCE_PATHS": parent.RUNNER_SOURCE_PATHS,
        "verify": parent._verify_config_and_inputs,
        "parity": parent._per_channel_parity,
    }

    def bounded_parity(model: object, session: object, _validation: object) -> tuple[float, float, float]:
        actual = original_values["parity"](model, session, predecessor_validation)
        observed["value"] = actual
        return expected if _parity_reproduction_within_tolerance(actual, expected) else actual

    parent.ROOT = ROOT
    parent.REVISION = REVISION
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.CONFIG_PATH = CONFIG_PATH
    parent.RUNNER_SOURCE_PATHS = RUNNER_SOURCE_PATHS
    parent._verify_config_and_inputs = _verify_config_and_inputs
    parent._per_channel_parity = bounded_parity
    try:
        report = parent._execute_candidate(output_dir, authorization, progress)
    finally:
        parent.ROOT = original_values["ROOT"]
        parent.REVISION = original_values["REVISION"]
        parent.CANDIDATE_ID = original_values["CANDIDATE_ID"]
        parent.CONFIG_PATH = original_values["CONFIG_PATH"]
        parent.RUNNER_SOURCE_PATHS = original_values["RUNNER_SOURCE_PATHS"]
        parent._verify_config_and_inputs = original_values["verify"]
        parent._per_channel_parity = original_values["parity"]
    old_checkpoint = REPO_ROOT / str(report["checkpoint_path"])
    old_onnx = REPO_ROOT / str(report["onnx_path"])
    checkpoint_path = output_dir / "marker-center-mask-consensus-v9-p1.pt"
    onnx_path = output_dir / "marker-center-mask-consensus-v9-p1.onnx"
    old_checkpoint.rename(checkpoint_path)
    old_onnx.rename(onnx_path)
    report.update(
        {
            "schema": "graphreader.marker-center-mask-consensus-candidate.v9",
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "predecessor_parity_expected_by_output_channel": list(expected),
            "predecessor_parity_observed_by_output_channel": list(observed["value"]),
            "predecessor_parity_reproduction_tolerance": PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
            "predecessor_parity_reproduction_passed": True,
        }
    )
    report.pop("p2_parity_by_output_channel", None)
    return report


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Marker-center V9 P1 output exists: {output_dir}")
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
            "schema": "graphreader.marker-center-mask-consensus-failure.v9",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "ml/markers/center/artifacts/mask-consensus-v9/P1-run",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
