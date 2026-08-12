# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for a selected V9 detector candidate."""

from __future__ import annotations

import argparse
from hashlib import sha256
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
from .pipeline import evaluate_scenes
from .protocol import PUBLIC_REVISION, REVISION, TASK, THRESHOLDS


REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")
RESULT_PATH = Path("ml/ocr/component_recall_detector_v9/P1_RESULT.json")
SPLIT_CONFIG_PATH = Path("ml/ocr/component_recall_detector_v9/gates/sealed-public-v1.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/component_recall_detector_v9/dataset.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline.py"),
    Path("ml/ocr/component_recall_detector_v9/protocol.py"),
    Path("ml/ocr/component_recall_detector_v9/sealed_gate.py"),
    Path("ml/ocr/component_fusion_detector_v8/dataset.py"),
    Path("ml/ocr/component_fusion_detector_v8/pipeline.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    "evaluation_limit": 1,
    "exact_region_count_every_fixture": True,
    "false_region_count": 0,
    "missed_region_count": 0,
    "duplicate_region_count": 0,
    "prohibited_structure_hits": 0,
    "onnx_parity_maximum_absolute_error": 1e-5,
    "provider": "CPUExecutionProvider",
}


def evaluate_candidate(*, onnx_path: Path, selection_report_path: Path, output_path: Path) -> dict[str, object]:
    require_committed_sources(REPO_ROOT, (LEDGER_PATH, RESULT_PATH))
    if output_path.exists():
        raise RuntimeError(f"OCR V9 public output already exists: {output_path}")
    ledger = json.loads((REPO_ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
    entry = next(
        (item for item in ledger.get("revisions", []) if item.get("task") == TASK and item.get("revision") == REVISION),
        None,
    )
    result = json.loads((REPO_ROOT / RESULT_PATH).read_text(encoding="utf-8"))
    selection = json.loads(selection_report_path.read_text(encoding="utf-8"))
    onnx_sha256 = sha256_file(onnx_path)
    selection_sha256 = sha256_file(selection_report_path)
    candidate_id = str(result.get("candidate_id"))
    if (
        entry is None
        or entry.get("status") != "candidate_1_selected_public_gate_pending"
        or entry.get("consumed_candidate_ids") != ["P1"]
        or entry.get("execution_authorized") is not False
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_authorized_candidate_id") != "P1"
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_archive_opened") is not False
        or entry.get("candidate_onnx_sha256", {}).get("P1") != onnx_sha256
        or entry.get("p1_training_report_sha256") != selection_sha256
        or entry.get("p1_result_sha256") != sha256_file(REPO_ROOT / RESULT_PATH)
    ):
        raise RuntimeError("OCR V9 public gate is not authorized by the canonical ledger")
    threshold = float(selection.get("selected_threshold", -1.0))
    if (
        result.get("status") != "selected_public_gate_pending"
        or result.get("onnx_sha256") != onnx_sha256
        or result.get("candidate_report_sha256") != selection_sha256
        or selection.get("status") != "selected"
        or selection.get("selection_gate_passed") is not True
        or selection.get("onnx_parity_passed") is not True
        or selection.get("revision") != REVISION
        or selection.get("sealed_public_archive_opened") is not False
        or threshold not in THRESHOLDS
    ):
        raise RuntimeError("Only the exact validation-selected V9 candidate may open the public gate")
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / split_config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR V9 public seal differs from the frozen gate configuration")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"]:
        raise RuntimeError("OCR V9 sealed fixture archive checksum mismatch")
    if sha256_file(private_manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("OCR V9 private manifest checksum mismatch")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V9 public gate requires CPUExecutionProvider only")
    if [item.name for item in session.get_inputs()] != ["region_proposals"]:
        raise RuntimeError("OCR V9 public gate found an unexpected ONNX input contract")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes={"onnx_sha256": onnx_sha256, "selection_report_sha256": selection_sha256},
        dataset_manifest_sha256=seal["private_manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )
    scenes = load_sealed_public_archive(archive_path)
    summary = proposal_summary(scenes)
    if (
        summary["scene_count"] != seal["scene_count"]
        or summary["truth_region_count"] != seal["truth_region_count"]
        or summary["proposal_count"] != seal["proposal_count"]
        or summary["positive_proposal_count"] != seal["positive_proposal_count"]
        or summary["negative_proposal_count"] != seal["negative_proposal_count"]
        or split_fingerprint(scenes) != seal["split_fingerprint"]
    ):
        raise RuntimeError("OCR V9 sealed fixture bytes violate the frozen contract")
    input_digest = sha256()
    output_digest = sha256()
    direct_calls = 0

    def runner(value: np.ndarray) -> np.ndarray:
        nonlocal direct_calls
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        input_digest.update(contiguous.tobytes())
        output = np.asarray(session.run(None, {"region_proposals": contiguous})[0], dtype=np.float32)
        output_digest.update(np.ascontiguousarray(output).tobytes())
        direct_calls += 1
        return output

    started = time.perf_counter()
    metrics = evaluate_scenes(scenes, runner, threshold)
    passed = (
        metrics["exact_scene_count"] == metrics["scene_count"] == seal["scene_count"]
        and metrics["true_positives"] == seal["truth_region_count"]
        and metrics["false_positives"] == metrics["false_negatives"] == 0
        and metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
        and direct_calls == seal["scene_count"]
    )
    report: dict[str, object] = {
        "schema": "graphreader.ocr-component-recall-public-gate.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "status": "pass" if passed else "fail",
        "production_approval": False,
        "release_eligible": False,
        "evaluation_count": 1,
        "candidate_id": candidate_id,
        "onnx_sha256": onnx_sha256,
        "selection_report_sha256": selection_sha256,
        "fixture_archive_sha256": seal["fixture_archive_sha256"],
        "private_manifest_sha256": seal["private_manifest_sha256"],
        "provider": "CPUExecutionProvider",
        "selected_threshold": threshold,
        "metrics": metrics,
        "direct_execution": {
            "inference_calls": direct_calls,
            "input_tensor_stream_sha256": input_digest.hexdigest(),
            "output_tensor_stream_sha256": output_digest.hexdigest(),
        },
        "gate_requirements": GATE_CONFIG,
        "seal_binding": gate.binding,
        "canonical_seal_key": gate.key,
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
    arguments = parser.parse_args()
    report = evaluate_candidate(
        onnx_path=REPO_ROOT / arguments.onnx,
        selection_report_path=REPO_ROOT / arguments.selection_report,
        output_path=REPO_ROOT / arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
