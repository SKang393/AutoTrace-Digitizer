# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V31 P2 runner-only correction after consumed P1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import complete_training_candidate

from . import train_p1 as p1
from .dataset import proposal_summary, split_fingerprint
from .prepare_split import SOURCE_PATHS
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    TASK,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    V30_CHECKPOINT_PATH,
    V30_CHECKPOINT_SHA256,
    V30_ONNX_PATH,
    V30_ONNX_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/robust_quorum_recall_v31")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
P1_CONFIG_PATH = ROOT / "training/p1.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v31-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v31-public.zip")
RUNNER_SOURCE_PATHS = (*SOURCE_PATHS, ROOT / "P1_RESULT.json", ROOT / "train_p2.py")

_RAW_CPU_SESSION = p1._cpu_session
_RAW_COMPLETE = complete_training_candidate


def _callable_cpu_session(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt one fixed ORT session to the frozen evidence-pipeline boundary."""

    session = _RAW_CPU_SESSION(path)
    input_name = session.get_inputs()[0].name

    def run(values: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(values)
        return np.asarray(
            session.run(None, {input_name: contiguous})[0],
            dtype=np.float32,
        )

    return run


def _validate_stored_split(
    scenes: tuple[Any, ...], registered: dict[str, Any], name: str,
) -> None:
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != registered["split_fingerprint"]
        or any(summary[key] != registered["proposal_summary"][key] for key in summary)
    ):
        raise RuntimeError(f"OCR V31 P2 {name} stored fixtures violate the seal")


def preflight(*, require_authorized: bool = True) -> dict[str, Any]:
    config = p1._read_json(REPO_ROOT / CONFIG_PATH)
    predecessor_config = p1._read_json(REPO_ROOT / P1_CONFIG_PATH)
    allowed_changes = {
        "candidate_id",
        "candidate_execution_authorized",
        "expected_runner_source_bundle_sha256",
    }
    for key, value in predecessor_config.items():
        if key not in allowed_changes and config.get(key) != value:
            raise RuntimeError(f"OCR V31 P2 changed the frozen P1 decision: {key}")
    expected_correction = {
        "schema": "graphreader.ocr-robust-quorum-recall-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "runner_correction": "contiguous-float32-callable-ort-adapter-v1",
        "isolated_change": (
            "wrap the unchanged detector and recognizer ONNX Runtime sessions as "
            "contiguous float32 callables required by the frozen evidence pipeline"
        ),
        "p1_result_path": P1_RESULT_PATH.as_posix(),
        "p1_result_sha256": sha256_file(REPO_ROOT / P1_RESULT_PATH),
        "expected_optimizer_steps": 0,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected_correction.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V31 P2 config field mismatch: {key}")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V31 P2 runner source bundle changed")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V31 split seal changed before P2")
    seal = p1._read_json(REPO_ROOT / SEAL_PATH)
    required_seal = {
        "schema": "graphreader.ocr-robust-quorum-recall-split-seal.v1",
        "revision": REVISION,
        "optimizer_steps_at_freeze": 0,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "candidate_execution_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "private_data": False,
        "chandler_used": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in required_seal.items():
        if seal.get(key) != value:
            raise RuntimeError(f"OCR V31 split seal field changed: {key}")
    head = p1._repository_head()
    if not p1._is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V31 split source commit is not an ancestor")
    if seal.get("source_bundle_sha256") != config.get("split_source_bundle_sha256"):
        raise RuntimeError("OCR V31 split source bundle changed")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V31 split source inventory is absent")
    for relative, expected_hash in sealed_sources.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V31 frozen split source changed: {relative}")
    for split, (path, key) in {
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(key):
            raise RuntimeError(f"OCR V31 {split} archive changed before P2")
    for relative, expected_hash in {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        V30_CHECKPOINT_PATH: V30_CHECKPOINT_SHA256,
        V30_ONNX_PATH: V30_ONNX_SHA256,
        TRIGGER_RESULT_PATH: TRIGGER_RESULT_SHA256,
    }.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V31 frozen input changed: {relative}")
    if not p1._trigger_is_terminal(p1._read_json(REPO_ROOT / TRIGGER_RESULT_PATH)):
        raise RuntimeError("OCR V31 aggregate-only V30 trigger changed")
    p1_result = p1._read_json(REPO_ROOT / P1_RESULT_PATH)
    if (
        p1_result.get("status") != "failed_runner_consumed"
        or p1_result.get("failure_exception_type") != "TypeError"
        or p1_result.get("optimizer_steps") != 0
        or p1_result.get("selection_archive_read_count") != 1
        or p1_result.get("public_gate_archive_opened") is not False
        or p1_result.get("case_detail_or_pixels_inspected") is not False
    ):
        raise RuntimeError("OCR V31 P1 aggregate failure record changed")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V31 P2 output already exists")
    ledger = p1._read_json(
        REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
    )
    entry = next(
        (
            item for item in ledger.get("revisions", [])
            if item.get("task") == TASK and item.get("revision") == REVISION
        ),
        None,
    )
    if (
        entry is None
        or entry.get("status") != "candidate_2_preregistered"
        or entry.get("preregistered_candidate_ids") != [CANDIDATE_ID]
        or entry.get("consumed_candidate_ids") != ["P1"]
        or entry.get("selection_evaluations") != 1
        or entry.get("public_gate_archive_opened") is not False
        or entry.get("public_gate_evaluations") != 0
    ):
        raise RuntimeError("OCR V31 P2 ledger state is not preregistered")
    if require_authorized and (
        entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or config.get("candidate_execution_authorized") is not True
    ):
        raise RuntimeError("OCR V31 P2 execution is not separately authorized")
    if not require_authorized and config.get("candidate_execution_authorized") not in (
        False, True,
    ):
        raise RuntimeError("OCR V31 P2 authorization field is invalid")
    return {"config": config, "seal": seal, "entry": entry, "head": head}


def _complete_with_p2_payload_names(authorization, *, status: str, report_sha256: str):
    report_path = REPO_ROOT / CANONICAL_OUTPUT / "candidate-report.json"
    if status in {"selected", "failed_selection"}:
        old_checkpoint = REPO_ROOT / CANONICAL_OUTPUT / (
            "graph-text-robust-quorum-recall-v31-p1.pt"
        )
        old_onnx = REPO_ROOT / CANONICAL_OUTPUT / (
            "graph-text-robust-quorum-recall-v31-p1.onnx"
        )
        new_checkpoint = old_checkpoint.with_name(
            "graph-text-robust-quorum-recall-v31-p2.pt"
        )
        new_onnx = old_onnx.with_name("graph-text-robust-quorum-recall-v31-p2.onnx")
        if not old_checkpoint.is_file() or not old_onnx.is_file():
            raise RuntimeError("OCR V31 P2 payload rename source is incomplete")
        if new_checkpoint.exists() or new_onnx.exists():
            raise RuntimeError("OCR V31 P2 payload rename target already exists")
        old_checkpoint.rename(new_checkpoint)
        old_onnx.rename(new_onnx)
        report = p1._read_json(report_path)
        report["checkpoint_path"] = new_checkpoint.relative_to(REPO_ROOT).as_posix()
        report["onnx_path"] = new_onnx.relative_to(REPO_ROOT).as_posix()
        report_path.write_bytes(canonical_json_bytes(report))
        report_sha256 = sha256_file(report_path)
    return _RAW_COMPLETE(
        authorization,
        status=status,
        report_sha256=report_sha256,
    )


def evaluate_candidate(output_dir: Path) -> dict[str, object]:
    originals = {
        "CANDIDATE_ID": p1.CANDIDATE_ID,
        "CONFIG_PATH": p1.CONFIG_PATH,
        "CANONICAL_OUTPUT": p1.CANONICAL_OUTPUT,
        "RUNNER_SOURCE_PATHS": p1.RUNNER_SOURCE_PATHS,
        "preflight": p1.preflight,
        "_cpu_session": p1._cpu_session,
        "complete_training_candidate": p1.complete_training_candidate,
    }
    p1.CANDIDATE_ID = CANDIDATE_ID
    p1.CONFIG_PATH = CONFIG_PATH
    p1.CANONICAL_OUTPUT = CANONICAL_OUTPUT
    p1.RUNNER_SOURCE_PATHS = RUNNER_SOURCE_PATHS
    p1.preflight = preflight
    p1._cpu_session = _callable_cpu_session
    p1.complete_training_candidate = _complete_with_p2_payload_names
    try:
        p1.evaluate_candidate(output_dir)
        return p1._read_json(output_dir / "candidate-report.json")
    finally:
        for name, value in originals.items():
            setattr(p1, name, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        evidence = preflight(require_authorized=True)
        print(json.dumps({
            "head": evidence["head"],
            "runner_source_bundle_sha256": source_bundle_sha256(
                REPO_ROOT, RUNNER_SOURCE_PATHS,
            ),
            "ready": True,
        }, sort_keys=True))
        return 0
    report = evaluate_candidate(REPO_ROOT / CANONICAL_OUTPUT)
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report.get("selected_threshold"),
        "passing_threshold_window": report.get("passing_threshold_window", []),
        "onnx_parity_maximum_absolute_error": report.get(
            "onnx_parity_maximum_absolute_error"
        ),
        "selection_metrics": report.get("selection_metrics"),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT", "CONFIG_PATH", "RUNNER_SOURCE_PATHS",
    "_callable_cpu_session", "evaluate_candidate", "preflight",
]
