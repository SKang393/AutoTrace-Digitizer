# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for a selected OCR V17 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    require_committed_sources,
    sha256_file,
)
from .dataset import load_sealed_public_archive, proposal_summary, split_fingerprint
from .pipeline import evaluate_thresholds
from .protocol import (
    PUBLIC_REVISION,
    REVISION,
    ROBUST_THRESHOLD_RUN_LENGTH,
    ROLE_ACCURACY_MINIMUM,
    ROLE_CLASS_ACCURACY_MINIMUM,
    TASK,
    THRESHOLDS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/structural_veto_proposal_role_v17")
LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")
SPLIT_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "sealed_gate.py",
    Path("ml/ocr/margin_robust_layout_proposal_role_v16/pipeline.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"), Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    "evaluation_limit": 1,
    "minimum_consecutive_thresholds": ROBUST_THRESHOLD_RUN_LENGTH,
    "exact_region_and_role_every_scene_at_each_threshold": True,
    "false_regions": 0,
    "missed_regions": 0,
    "duplicate_regions": 0,
    "prohibited_structure_hits": 0,
    "role_accuracy_minimum": ROLE_ACCURACY_MINIMUM,
    "per_role_accuracy_minimum": ROLE_CLASS_ACCURACY_MINIMUM,
    "provider": "CPUExecutionProvider",
    "direct_fixture_byte_execution_required": True,
    "case_level_failure_analysis_permitted": False,
}


def _passes(metrics: dict[str, object]) -> bool:
    per_role = metrics["per_role_accuracy"]
    assert isinstance(per_role, dict)
    return (
        metrics["exact_scene_count"] == metrics["scene_count"]
        and metrics["true_positives"] == metrics["truth_region_count"]
        and metrics["false_positives"] == metrics["false_negatives"]
        == metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
        and float(metrics["role_accuracy"]) >= ROLE_ACCURACY_MINIMUM
        and min(float(value) for value in per_role.values()) >= ROLE_CLASS_ACCURACY_MINIMUM
    )


def evaluate_candidate(*, onnx_path: Path, selection_report_path: Path, output_path: Path) -> dict[str, object]:
    require_committed_sources(REPO_ROOT, (LEDGER_PATH,))
    if output_path.exists():
        raise RuntimeError(f"OCR V17 public output exists: {output_path}")
    ledger = json.loads((REPO_ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
    entry = next(
        (item for item in ledger["revisions"] if item.get("task") == TASK and item.get("revision") == REVISION),
        None,
    )
    selection = json.loads(selection_report_path.read_text(encoding="utf-8"))
    onnx_sha, selection_sha = sha256_file(onnx_path), sha256_file(selection_report_path)
    threshold = float(selection.get("selected_threshold", -1.0))
    window = tuple(float(value) for value in selection.get("passing_threshold_window", []))
    authorized = entry.get("public_gate_authorized_candidate_id") if entry else None
    ordinal = int(authorized[1:]) if isinstance(authorized, str) and authorized[1:].isdigit() else 0
    expected_consumed = [f"P{index}" for index in range(1, ordinal + 1)]
    if (
        entry is None or entry.get("status") != f"candidate_{ordinal}_selected_public_gate_pending"
        or entry.get("preregistered_candidate_ids") != []
        or entry.get("consumed_candidate_ids") != expected_consumed
        or entry.get("execution_authorized") is not False or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_evaluations") != 0 or entry.get("public_gate_archive_opened") is not False
        or entry.get(f"p{ordinal}_onnx_sha256") != onnx_sha
        or entry.get(f"p{ordinal}_selection_report_sha256") != selection_sha
    ):
        raise RuntimeError("OCR V17 public gate is not authorized by the canonical ledger")
    if (
        selection.get("status") != "selected" or selection.get("selection_gate_passed") is not True
        or selection.get("onnx_parity_passed") is not True
        or selection.get("sealed_public_archive_opened") is not False
        or threshold not in THRESHOLDS or threshold not in window
        or len(window) < ROBUST_THRESHOLD_RUN_LENGTH
        or any(value not in THRESHOLDS for value in window)
    ):
        raise RuntimeError("Only the exact robustly selected V17 candidate may open the public gate")
    selected_index = window.index(threshold)
    lower = max(0, selected_index - 1)
    public_window = window[lower:lower + ROBUST_THRESHOLD_RUN_LENGTH]
    if len(public_window) < ROBUST_THRESHOLD_RUN_LENGTH:
        public_window = window[-ROBUST_THRESHOLD_RUN_LENGTH:]
    config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR V17 public seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if (
        sha256_file(archive_path) != seal["fixture_archive_sha256"]
        or sha256_file(manifest_path) != seal["private_manifest_sha256"]
    ):
        raise RuntimeError("OCR V17 public fixture bytes changed")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V17 public gate requires CPU execution only")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT, task=TASK, revision=PUBLIC_REVISION,
        candidate_hashes={"onnx_sha256": onnx_sha, "selection_report_sha256": selection_sha},
        dataset_manifest_sha256=seal["private_manifest_sha256"], split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS, gate_config=GATE_CONFIG,
    )
    calls = 0

    def runner(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray(
            session.run(None, {"region_proposals": np.ascontiguousarray(values)})[0], dtype=np.float32,
        )

    started = time.perf_counter()
    scenes = load_sealed_public_archive(archive_path, manifest_path)
    summary = proposal_summary(scenes)
    if any(summary[key] != seal[key] for key in summary) or split_fingerprint(scenes) != seal["split_fingerprint"]:
        raise RuntimeError("OCR V17 public fixtures violate the frozen contract")
    comparisons = evaluate_thresholds(scenes, runner, public_window)
    passed = all(_passes(item["metrics"]) for item in comparisons) and calls == len(scenes)
    selected_metrics = next(item["metrics"] for item in comparisons if item["threshold"] == threshold)
    report: dict[str, object] = {
        "schema": "graphreader.ocr-structural-veto-proposal-role-public-gate.v1",
        "task": TASK, "revision": PUBLIC_REVISION, "status": "pass" if passed else "fail",
        "production_approval": False, "release_eligible": False, "evaluation_count": 1,
        "onnx_sha256": onnx_sha, "selection_report_sha256": selection_sha,
        "fixture_archive_sha256": seal["fixture_archive_sha256"], "provider": "CPUExecutionProvider",
        "selected_threshold": threshold, "public_threshold_window": list(public_window),
        "metrics": selected_metrics, "threshold_comparisons": comparisons,
        "direct_execution_inference_calls": calls,
        "gate_requirements": GATE_CONFIG, "seal_binding": gate.binding, "canonical_seal_key": gate.key,
        "case_level_details_emitted": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(gate, status=str(report["status"]), report_sha256=sha256_file(output_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_candidate(
        onnx_path=REPO_ROOT / args.onnx,
        selection_report_path=REPO_ROOT / args.selection_report,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
