# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Preregistered unchanged-weight candidate for probability-packed runtime v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import FILL_NAMES, SHAPE_NAMES, build_fixed_dataset
from .metrics import binary_metrics, classification_metrics
from .runtime_v2 import PARITY_TOLERANCE, export_probability_onnx, run_probability_runtime


TASK = "marker-classifier"
REVISION = "marker-classifier-production-runtime-repair-v2"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/classifier/training/production-runtime-repair-v2-p1.json")
SOURCE_CHECKPOINT_PATH = Path("ml/markers/classifier/artifacts/production-repair-v1/P2/marker-classifier.pt")
SELECTION_MANIFEST_PATH = Path("ml/markers/classifier/manifests/runtime-repair-v2-selection.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/classifier/runtime_repair_v2.py"),
    Path("ml/markers/classifier/runtime_v2.py"),
    Path("ml/markers/classifier/dataset.py"),
    Path("ml/markers/classifier/metrics.py"),
    Path("ml/markers/classifier/model.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/markers/gate_seal.py"),
    SELECTION_MANIFEST_PATH,
)
SHAPE_MACRO_F1_GATE = 0.90
FILL_MACRO_F1_GATE = 0.90
ARTIFACT_F1_GATE = 1.0
MINORITY_CLASS_F1_GATE = 0.90


def selection_manifest() -> dict[str, object]:
    samples = build_fixed_dataset("validation")
    return {
        "manifest_version": 1,
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "selection_use": True,
        "public_gate_use": False,
        "private_data": False,
        "source_dataset_revision": "marker-classifier-procedural-v1",
        "selected_split": "validation",
        "selected_case_count": len(samples),
        "cases": [
            {
                "sample_id": sample.sample_id,
                "family": sample.family,
                "template": sample.template,
                "scenario": sample.scenario,
                "shape": SHAPE_NAMES[sample.shape_index],
                "fill": FILL_NAMES[sample.fill_index],
                "artifact": sample.artifact,
                "artifact_kind": sample.artifact_kind,
                "tensor_sha256": hashlib.sha256(sample.tensor.numpy().tobytes(order="C")).hexdigest(),
            }
            for sample in samples
        ],
    }


def _selection_metrics(actual: np.ndarray) -> tuple[dict[str, object], dict[str, bool]]:
    samples = build_fixed_dataset("validation")
    marker_indices = np.array([index for index, sample in enumerate(samples) if sample.artifact < 0.5])
    shape_targets = np.array([sample.shape_index for sample in samples], dtype=np.int64)[marker_indices]
    fill_targets = np.array([sample.fill_index for sample in samples], dtype=np.int64)[marker_indices]
    artifact_targets = np.array([sample.artifact for sample in samples], dtype=np.float32)
    shape = classification_metrics(actual[marker_indices, 0:9], shape_targets, len(SHAPE_NAMES))
    fill = classification_metrics(actual[marker_indices, 9:12], fill_targets, len(FILL_NAMES))
    artifact = binary_metrics(actual[:, 12], artifact_targets)
    minority = {
        name: shape.per_class_f1[SHAPE_NAMES.index(name)]
        for name in ("star", "asterisk", "cross")
    }
    metrics = {
        "shape": shape.to_dict(),
        "fill": fill.to_dict(),
        "artifact": artifact,
        "minority_shape_f1": minority,
    }
    gates = {
        "shape_macro_f1": shape.macro_f1 >= SHAPE_MACRO_F1_GATE,
        "fill_macro_f1": fill.macro_f1 >= FILL_MACRO_F1_GATE,
        "artifact_f1": float(artifact["f1"]) == ARTIFACT_F1_GATE,
        "minority_shape_preservation": min(minority.values()) >= MINORITY_CLASS_F1_GATE,
    }
    return metrics, gates


def run_candidate(output_dir: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    config = json.loads((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    checkpoint = repo_root / SOURCE_CHECKPOINT_PATH
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Exact local source checkpoint is missing: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != config["source_checkpoint_sha256"]:
        raise RuntimeError("Local source checkpoint does not match the preregistered checksum")
    manifest_bytes = canonical_json_bytes(selection_manifest())
    committed_manifest = repo_root / SELECTION_MANIFEST_PATH
    if committed_manifest.read_bytes() != manifest_bytes:
        raise RuntimeError("Generated selection manifest does not match the committed frozen manifest")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != config["selection_dataset_manifest_sha256"]:
        raise RuntimeError("Selection manifest does not match the preregistered checksum")
    authorization = acquire_training_candidate(
        repo_root,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "marker-classifier-probability-packed.onnx"
    parity_path = output_dir / "selection-onnx-parity.json"
    samples = build_fixed_dataset("validation")
    parity_report = export_probability_onnx(checkpoint, onnx_path, parity_path, samples)
    _, actual, parity_max, inference_ms, provider = run_probability_runtime(checkpoint, onnx_path, samples)
    metrics, gate_results = _selection_metrics(actual)
    gate_results["probability_packed_onnx_parity"] = parity_max <= PARITY_TOLERANCE
    unchanged_checkpoint = sha256_file(checkpoint) == config["source_checkpoint_sha256"]
    gate_results["weights_unchanged"] = unchanged_checkpoint
    report: dict[str, object] = {
        "status": "selected" if all(gate_results.values()) else "fail",
        "release_eligible": False,
        "release_blocker": "Once-only public and disjoint confirmation gates have not run; production discovery and packaging also remain blocked.",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "experiment_budget": 3,
        "experiment_ordinal": 1,
        "optimizer_steps": 0,
        "weights_changed": False,
        "source_checkpoint": SOURCE_CHECKPOINT_PATH.as_posix(),
        "source_checkpoint_sha256": checkpoint_sha256,
        "selection_dataset_manifest": SELECTION_MANIFEST_PATH.as_posix(),
        "selection_dataset_manifest_sha256": manifest_sha256,
        "selection_sample_count": len(samples),
        "public_gate_evaluations": 0,
        "confirmation_gate_evaluations": 0,
        "probability_packed_onnx": str(onnx_path),
        "probability_packed_onnx_sha256": sha256_file(onnx_path),
        "parity_report": str(parity_path),
        "parity_report_sha256": sha256_file(parity_path),
        "provider": provider,
        "selection_metrics": metrics,
        "selection_gate_results": gate_results,
        "probability_packed_onnx_maximum_absolute_error": parity_max,
        "probability_packed_onnx_parity_tolerance": PARITY_TOLERANCE,
        "inference_total_ms": round(inference_ms, 3),
        "parity_export_status": parity_report["status"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    report_path = output_dir / "candidate-report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(
        authorization,
        status=str(report["status"]),
        report_sha256=sha256_file(report_path),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_candidate(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ID",
    "CONFIG_PATH",
    "REVISION",
    "RUNNER_SOURCE_PATHS",
    "SELECTION_MANIFEST_PATH",
    "SOURCE_CHECKPOINT_PATH",
    "TASK",
    "run_candidate",
    "selection_manifest",
]
