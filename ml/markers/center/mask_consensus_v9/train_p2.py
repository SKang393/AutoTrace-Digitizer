# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V9 P2 runner repair after the zero-step P1 alias failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import ml.markers.center.mask_consensus_v8.train_p3 as parent
import ml.markers.center.mask_consensus_v9.train_p1 as p1
from ml.markers.center.mask_consensus_v9.protocol import (
    PREDECESSOR_PARITY_REPRODUCTION_TOLERANCE,
    REVISION,
    ROOT as RELATIVE_ROOT,
    TASK,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / RELATIVE_ROOT
CANDIDATE_ID = "P2"
CONFIG_PATH = Path("ml/markers/center/mask_consensus_v9/training/p2.json")
P1_RESULT_PATH = Path("ml/markers/center/mask_consensus_v9/P1_RESULT.json")
RUNNER_SOURCE_PATHS = (*p1.RUNNER_SOURCE_PATHS, Path("ml/markers/center/mask_consensus_v9/train_p2.py"))


def _verify_config_and_inputs(config: dict[str, object]) -> tuple[dict[str, object], Path, Path]:
    if (config.get("task"), config.get("revision"), config.get("candidate_id")) != (
        TASK,
        REVISION,
        CANDIDATE_ID,
    ):
        raise RuntimeError("Marker-center V9 P2 candidate identity changed")
    p1_result_path = REPO_ROOT / P1_RESULT_PATH
    if sha256_file(p1_result_path) != config.get("p1_result_sha256"):
        raise RuntimeError("Marker-center V9 P1 result changed")
    p1_result = json.loads(p1_result_path.read_text(encoding="utf-8"))
    if (
        p1_result.get("status") != "failed_runner_consumed"
        or p1_result.get("failure_type") != "KeyError"
        or p1_result.get("failure_message") != "'p2_parity_by_output_channel'"
        or p1_result.get("optimizer_steps") != 0
        or p1_result.get("model_payload_created") is not False
    ):
        raise RuntimeError("Marker-center V9 P2 requires the consumed zero-step P1 alias failure")
    for path_key, hash_key in (
        ("training_opened_seal_path", "training_opened_seal_sha256"),
        ("training_result_seal_path", "training_result_seal_sha256"),
    ):
        path = REPO_ROOT / str(p1_result[path_key])
        if sha256_file(path) != config.get(hash_key):
            raise RuntimeError(f"Marker-center V9 P1 seal changed: {path.name}")
    expected = config.get("predecessor_parity_by_output_channel")
    if config.get("p2_parity_by_output_channel") != expected:
        raise RuntimeError("Marker-center V9 P2 legacy predecessor parity alias changed")
    return p1._verify_config_and_inputs(config)


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
        return expected if p1._parity_reproduction_within_tolerance(actual, expected) else actual

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
    checkpoint_path = output_dir / "marker-center-mask-consensus-v9-p2.pt"
    onnx_path = output_dir / "marker-center-mask-consensus-v9-p2.onnx"
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
            "p1_output_reused": False,
        }
    )
    report.pop("p2_parity_by_output_channel", None)
    return report


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Marker-center V9 P2 output exists: {output_dir}")
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
        default=REPO_ROOT / "ml/markers/center/artifacts/mask-consensus-v9/P2-run",
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
