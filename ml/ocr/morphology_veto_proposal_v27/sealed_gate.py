# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for a selected OCR V27 candidate."""

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
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _calibrated_records,
    _feature_groups,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, metrics_pass
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive, proposal_summary, split_fingerprint
from .features import structure_features
from .pipeline import extract_crop_evidence
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    PARENT_ONNX_PATH,
    PARENT_ONNX_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    ROBUST_THRESHOLD_RUN_LENGTH,
    STRUCTURE_FEATURE_COUNT,
    TASK,
    THRESHOLDS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/morphology_veto_proposal_v27")
LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")
SPLIT_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
PUBLIC_REVISION = "graph-text-morphology-veto-proposal-v27-public-v1"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "features.py",
    ROOT / "model.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "sealed_gate.py",
    ROOT / "train_p1.py",
    Path("ml/ocr/scene_topology_proposal_v26/dataset.py"),
    Path("ml/ocr/scene_topology_proposal_v26/model.py"),
    Path("ml/ocr/scene_topology_proposal_v26/model_p2.py"),
    Path("ml/ocr/scene_topology_proposal_v26/model_p3.py"),
    Path("ml/ocr/scene_topology_proposal_v26/pipeline.py"),
    Path("ml/ocr/scene_topology_proposal_v26/protocol.py"),
    Path("ml/ocr/evidence_rescue_v25/model.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/model_p2.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/pipeline.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/protocol.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/train_p1.py"),
    Path("ml/ocr/role_anchor_set_v23/model.py"),
    Path("ml/ocr/role_anchor_set_v23/protocol.py"),
    Path("ml/ocr/margin_calibrator_v20/dataset.py"),
    Path("ml/ocr/margin_calibrator_v20/pipeline.py"),
    Path("ml/ocr/margin_calibrator_v20/protocol.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/dataset.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/protocol.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    "evaluation_limit": 1,
    "minimum_consecutive_thresholds": ROBUST_THRESHOLD_RUN_LENGTH,
    "exact_region_and_role_every_scene_at_each_threshold": True,
    "false_regions": 0,
    "missed_regions": 0,
    "duplicate_regions": 0,
    "prohibited_structure_hits": 0,
    "recognition_exact_minimum": 0.90,
    "character_error_rate_maximum": 0.05,
    "role_accuracy_minimum": 0.90,
    "per_role_accuracy_minimum": 0.85,
    "provider": "CPUExecutionProvider",
    "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
    "parent_role_argmax_preservation_required": True,
    "direct_fixture_byte_execution_required": True,
    "proposal_stream": "production_detector_floor",
    "detector_recognizer_parent_candidate_and_structure_tensor_hashes_required": True,
    "case_level_failure_analysis_permitted": False,
}


def _cpu_session(path: Path, *, disable_optimization: bool = False) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    if disable_optimization:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V27 public gate requires CPUExecutionProvider only")
    return session


def _public_window(selection: dict[str, object]) -> tuple[float, ...]:
    selected = float(selection.get("selected_threshold", -1.0))
    window = tuple(float(value) for value in selection.get("passing_threshold_window", []))
    if (
        selected not in THRESHOLDS
        or selected not in window
        or len(window) < ROBUST_THRESHOLD_RUN_LENGTH
        or any(value not in THRESHOLDS for value in window)
    ):
        raise RuntimeError("OCR V27 selection has no preregistered robust threshold window")
    selected_index = window.index(selected)
    lower = max(0, selected_index - 1)
    result = window[lower:lower + ROBUST_THRESHOLD_RUN_LENGTH]
    if len(result) < ROBUST_THRESHOLD_RUN_LENGTH:
        result = window[-ROBUST_THRESHOLD_RUN_LENGTH:]
    return result


def evaluate_public(*, selection_report_path: Path, output_path: Path) -> dict[str, object]:
    require_committed_sources(REPO_ROOT, (LEDGER_PATH,))
    if output_path.exists():
        raise RuntimeError(f"OCR V27 public output exists: {output_path}")
    selection = json.loads(selection_report_path.read_text(encoding="utf-8"))
    selection_sha = sha256_file(selection_report_path)
    candidate_id = selection.get("candidate_id")
    ordinal = (
        int(candidate_id[1:])
        if isinstance(candidate_id, str)
        and candidate_id.startswith("P")
        and candidate_id[1:].isdigit()
        else 0
    )
    expected_consumed = [f"P{index}" for index in range(1, ordinal + 1)]
    ledger = json.loads((REPO_ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
    entry = next(
        (
            item for item in ledger["revisions"]
            if item.get("task") == TASK and item.get("revision") == REVISION
        ),
        None,
    )
    candidate_path = REPO_ROOT / str(selection.get("onnx_path", ""))
    candidate_sha = sha256_file(candidate_path) if candidate_path.is_file() else ""
    if (
        ordinal < 1
        or entry is None
        or entry.get("status") != f"candidate_{ordinal}_selected_public_gate_pending"
        or entry.get("preregistered_candidate_ids") != []
        or entry.get("consumed_candidate_ids") != expected_consumed
        or entry.get("execution_authorized") is not False
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_authorized_candidate_id") != candidate_id
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_archive_opened") is not False
        or entry.get(f"p{ordinal}_onnx_sha256") != candidate_sha
        or entry.get(f"p{ordinal}_selection_report_sha256") != selection_sha
    ):
        raise RuntimeError("OCR V27 public gate is not authorized by the canonical ledger")
    if (
        selection.get("status") != "selected"
        or selection.get("selection_gate_passed") is not True
        or selection.get("onnx_parity_passed") is not True
        or selection.get("parent_role_argmax_preserved") is not True
        or selection.get("public_gate_archive_opened") is not False
        or selection.get("public_gate_evaluations") != 0
        or selection.get("onnx_sha256") != candidate_sha
    ):
        raise RuntimeError("Only the exact robustly selected OCR V27 candidate may open the public gate")
    public_window = _public_window(selection)

    config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    split_seal_path = REPO_ROOT / config["split_seal_path"]
    if sha256_file(split_seal_path) != config["split_seal_sha256"]:
        raise RuntimeError("OCR V27 split seal changed")
    split_seal = json.loads(split_seal_path.read_text(encoding="utf-8"))
    registered = split_seal["splits"]["sealed_public"]
    archive_path = REPO_ROOT / config["public_fixture_archive_path"]
    if (
        config.get("public_fixture_archive_path") != registered["archive_path"]
        or config.get("public_fixture_archive_sha256") != registered["archive_sha256"]
        or config.get("public_fixture_manifest_sha256") != registered["manifest_sha256"]
        or sha256_file(archive_path) != registered["archive_sha256"]
        or sha256_file(REPO_ROOT / DETECTOR_PATH) != DETECTOR_SHA256
        or sha256_file(REPO_ROOT / RECOGNIZER_PATH) != RECOGNIZER_SHA256
        or sha256_file(REPO_ROOT / RECOGNIZER_YAML_PATH) != RECOGNIZER_YAML_SHA256
        or sha256_file(REPO_ROOT / PARENT_ONNX_PATH) != PARENT_ONNX_SHA256
    ):
        raise RuntimeError("OCR V27 public inputs changed")

    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=PUBLIC_REVISION,
        candidate_hashes={
            "detector_onnx_sha256": DETECTOR_SHA256,
            "recognizer_onnx_sha256": RECOGNIZER_SHA256,
            "v26_parent_onnx_sha256": PARENT_ONNX_SHA256,
            "candidate_onnx_sha256": candidate_sha,
            "selection_report_sha256": selection_sha,
        },
        dataset_manifest_sha256=registered["manifest_sha256"],
        split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIG,
    )

    started = time.perf_counter()
    scenes = load_archive(archive_path)
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != registered["split_fingerprint"]
        or any(summary[key] != registered["proposal_summary"][key] for key in summary)
    ):
        raise RuntimeError("OCR V27 public fixtures violate the frozen contract")
    detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
    recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
    parent_session = _cpu_session(REPO_ROOT / PARENT_ONNX_PATH)
    candidate_session = _cpu_session(candidate_path, disable_optimization=True)
    detector_input = detector_session.get_inputs()[0].name
    recognizer_input = recognizer_session.get_inputs()[0].name
    alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)

    def detector_runner(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            detector_session.run(
                None, {detector_input: np.ascontiguousarray(values)},
            )[0],
            dtype=np.float32,
        )

    def recognizer_runner(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            recognizer_session.run(
                None, {recognizer_input: np.ascontiguousarray(values)},
            )[0],
            dtype=np.float32,
        )

    values, crops, _, records, runtime_evidence = extract_crop_evidence(
        scenes,
        detector_runner,
        recognizer_runner,
        alphabet,
        mode="evaluate",
        recognition_batch_size=64,
    )
    if (
        runtime_evidence["scene_count"] != registered["proposal_summary"]["scene_count"]
        or not 0 < runtime_evidence["proposal_count"] <= registered["proposal_summary"]["proposal_count"]
    ):
        raise RuntimeError("OCR V27 public production proposal stream is invalid")
    structure = structure_features(crops)
    if structure.shape != (len(crops), STRUCTURE_FEATURE_COUNT):
        raise RuntimeError("OCR V27 public structure contract changed")

    groups = _feature_groups(records, len(scenes))
    evidence_inputs = sha256()
    crop_inputs = sha256()
    structure_inputs = sha256()
    parent_outputs = sha256()
    candidate_outputs = sha256()
    output_batches: list[np.ndarray] = []
    parent_role_argmax_mismatches = 0
    for indices in groups:
        evidence_batch = np.ascontiguousarray(values[indices][None, ...])
        crop_batch = np.ascontiguousarray(crops[indices][None, ...])
        structure_batch = np.ascontiguousarray(structure[indices][None, ...])
        parent_output = np.asarray(parent_session.run(None, {
            "proposal_evidence": evidence_batch,
            "proposal_crops": crop_batch,
        })[0], dtype=np.float32)
        candidate_output = np.asarray(candidate_session.run(None, {
            "proposal_evidence": evidence_batch,
            "proposal_crops": crop_batch,
            "structure_features": structure_batch,
        })[0], dtype=np.float32)
        if parent_output.shape != candidate_output.shape or candidate_output.shape[2] != 10:
            raise RuntimeError("OCR V27 public candidate output contract changed")
        parent_role_argmax_mismatches += int(np.count_nonzero(
            np.argmax(candidate_output[:, :, 2:], axis=2)
            != np.argmax(parent_output[:, :, 2:], axis=2)
        ))
        output_batches.append(candidate_output[0])
        evidence_inputs.update(evidence_batch.tobytes(order="C"))
        crop_inputs.update(crop_batch.tobytes(order="C"))
        structure_inputs.update(structure_batch.tobytes(order="C"))
        parent_outputs.update(parent_output.tobytes(order="C"))
        candidate_outputs.update(candidate_output.tobytes(order="C"))
    if len(groups) != len(scenes) or parent_role_argmax_mismatches != 0:
        raise RuntimeError("OCR V27 public candidate did not preserve parent role argmax")

    flat_output = np.concatenate(output_batches)
    calibrated_records = _calibrated_records(records, flat_output)
    runtime_evidence.update({
        "candidate_inference_calls": len(groups),
        "parent_inference_calls": len(groups),
        "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
        "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
        "candidate_structure_input_tensor_stream_sha256": structure_inputs.hexdigest(),
        "parent_output_tensor_stream_sha256": parent_outputs.hexdigest(),
        "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
        "parent_role_argmax_mismatch_count": parent_role_argmax_mismatches,
        "candidate_onnx_sha256": candidate_sha,
        "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
    })
    comparisons = evaluate_thresholds(
        scenes,
        calibrated_records,
        flat_output[:, :2],
        public_window,
        runtime_evidence,
    )
    passed = (
        len(comparisons) == ROBUST_THRESHOLD_RUN_LENGTH
        and all(metrics_pass(item["metrics"]) for item in comparisons)
    )
    selected_threshold = float(selection["selected_threshold"])
    selected_metrics = next(
        item["metrics"] for item in comparisons if item["threshold"] == selected_threshold
    )
    report: dict[str, object] = {
        "schema": "graphreader.ocr-morphology-veto-public-gate.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "candidate_id": candidate_id,
        "status": "pass" if passed else "fail",
        "evaluation_count": 1,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "v26_parent_sha256": PARENT_ONNX_SHA256,
        "candidate_sha256": candidate_sha,
        "selection_report_sha256": selection_sha,
        "fixture_archive_sha256": registered["archive_sha256"],
        "fixture_manifest_sha256": registered["manifest_sha256"],
        "provider": "CPUExecutionProvider",
        "selected_threshold": selected_threshold,
        "public_threshold_window": list(public_window),
        "metrics": selected_metrics,
        "threshold_comparisons": comparisons,
        "gate_requirements": GATE_CONFIG,
        "seal_binding": gate.binding,
        "canonical_seal_key": gate.key,
        "case_level_details_emitted": False,
        "marker_creation_evaluated": False,
        "marker_creation_gate_required_before_production_approval": True,
        "production_approval": False,
        "release_eligible": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(
        gate, status=str(report["status"]), report_sha256=sha256_file(output_path),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_public(
        selection_report_path=REPO_ROOT / args.selection_report,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATOR_SOURCE_PATHS", "GATE_CONFIG", "PUBLIC_REVISION",
    "SPLIT_CONFIG_PATH", "_public_window", "evaluate_public",
]
