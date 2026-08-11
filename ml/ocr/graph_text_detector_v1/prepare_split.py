# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze V1 selection metadata and a truth-hidden sealed public split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256

from .dataset import build_validation_split, split_fingerprint, training_split_fingerprint
from .protocol import (
    BATCH_SIZE,
    CANDIDATE_ID,
    EPOCHS,
    EXPERIMENT_BUDGET,
    LEARNING_RATE,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    REVISION,
    SEED,
    TASK,
    WEIGHT_DECAY,
    protocol_configuration,
)
from .sealed_dataset import build_sealed_public_split, save_sealed_public_archive
from .train_p1 import RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIVATE_ROOT = Path("ml/ocr/graph_text_detector_v1/artifacts/split-freeze")
DEFAULT_PROTOCOL = Path("ml/ocr/graph_text_detector_v1/PROTOCOL.json")
DEFAULT_SELECTION = Path("ml/ocr/graph_text_detector_v1/SELECTION_MANIFEST.json")
DEFAULT_SEAL = Path("ml/ocr/graph_text_detector_v1/SEALED_PUBLIC_TEST_SEAL.json")
DEFAULT_TRAINING = Path("ml/ocr/graph_text_detector_v1/training/p1.json")
DEFAULT_DIAGNOSTIC_RESULT = Path("ml/ocr/combined_component_v5/DETECTOR_DIAGNOSTIC_RESULT.json")
DEFAULT_DIAGNOSTIC_REPORT = Path("ml/ocr/combined_component_v5/runs/detector-diagnostic-v1/report.json")
SPLIT_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_detector_v1/dataset.py"),
    Path("ml/ocr/graph_text_detector_v1/sealed_dataset.py"),
    Path("ml/ocr/graph_text_detector_v1/prepare_split.py"),
    Path("ml/ocr/graph_text_detector_v1/protocol.py"),
)


def _validate_diagnostic(path: Path) -> tuple[dict[str, object], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "graphreader.ocr-combined-v5-detector-diagnostic.v1",
        "profile": "graphreader-ocr-combined-v5-detector-diagnostic-v1",
        "status": "diagnostic_complete",
        "purpose": "non_approval_detector_defect_characterization",
        "production_approval": False,
        "release_eligible": False,
        "private_data": False,
        "chandler_used": False,
        "detector_onnx_sha256": "d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb",
    }
    for key, value in required.items():
        if report.get(key) != value:
            raise RuntimeError(f"Detector diagnostic field changed: {key}")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("case_count") != 72:
        raise RuntimeError("Detector diagnostic metrics are invalid")
    expected_metrics = {
        "strict_probability_violation_count": 0,
        "nonfinite_output_count": 0,
        "shape_failure_count": 0,
        "composition_exact_rate": 0.4861111111111111,
        "text_detection_exact_rate": 0.22916666666666666,
        "exclusion_exact_rate": 1.0,
        "false_region_count": 9,
    }
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"Detector diagnostic metric changed: {key}")
    return report, sha256_file(path)


def freeze_split(
    *,
    private_root: Path,
    protocol_path: Path,
    selection_path: Path,
    seal_path: Path,
    training_path: Path,
    diagnostic_result_path: Path,
    diagnostic_report_path: Path,
) -> dict[str, object]:
    private = REPO_ROOT / private_root
    tracked_targets = tuple(
        REPO_ROOT / path
        for path in (protocol_path, selection_path, seal_path, training_path, diagnostic_result_path)
    )
    archive_path = private / "sealed-public-fixtures.npz"
    private_manifest_path = private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in (*tracked_targets, private, archive_path, private_manifest_path) if path.exists()]
    if existing:
        raise RuntimeError("Graph text detector split freeze refuses to overwrite evidence: " + ", ".join(existing))
    diagnostic_report, diagnostic_sha256 = _validate_diagnostic(REPO_ROOT / diagnostic_report_path)
    private.mkdir(parents=True, exist_ok=False)
    for target in tracked_targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    diagnostic_summary = {
        "schema": "graphreader.ocr-combined-v5-detector-diagnostic-result.v1",
        "status": "official_detector_rejected_for_combined_component_composition",
        "diagnostic_report_path": diagnostic_report_path.as_posix(),
        "diagnostic_report_sha256": diagnostic_sha256,
        "detector_onnx_sha256": diagnostic_report["detector_onnx_sha256"],
        "provider": diagnostic_report["provider"],
        "metrics": diagnostic_report["metrics"],
        "decision": (
            "Do not open a combined production gate with the official detector. "
            "Preregister a distinct project-trained graph text-region detector without changing DB thresholds."
        ),
        "production_approval": False,
        "release_eligible": False,
        "model_manifest_created": False,
        "model_store_promoted": False,
        "private_data": False,
        "chandler_used": False,
    }
    (REPO_ROOT / diagnostic_result_path).write_bytes(canonical_json_bytes(diagnostic_summary))

    protocol = protocol_configuration()
    protocol["trigger"]["diagnostic_report_sha256"] = diagnostic_sha256
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SPLIT_SOURCE_PATHS]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    (REPO_ROOT / protocol_path).write_bytes(canonical_json_bytes(protocol))

    validation = build_validation_split()
    train_fingerprint = training_split_fingerprint()
    validation_fingerprint = split_fingerprint(validation)
    selection = {
        "schema": "graphreader.ocr-graph-text-detector-selection.v1",
        "task": TASK,
        "revision": REVISION,
        "train_sample_count": 512,
        "validation_sample_count": len(validation),
        "train_split_fingerprint": train_fingerprint,
        "validation_split_fingerprint": validation_fingerprint,
        "diagnostic_report_sha256": diagnostic_sha256,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "diagnostic_fixture_reused": False,
        "sealed_public_truth_available_to_training": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / selection_path).write_bytes(canonical_json_bytes(selection))

    sealed = build_sealed_public_split()
    private_manifest = save_sealed_public_archive(sealed, archive_path)
    private_manifest_path.write_bytes(canonical_json_bytes(private_manifest))
    seal = {
        "schema": "graphreader.ocr-graph-text-detector-sealed-test-seal.v1",
        "task": TASK,
        "revision": REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(sealed),
        "text_count": private_manifest["text_count"],
        "exclusion_count": private_manifest["exclusion_count"],
        "split_fingerprint": private_manifest["split_fingerprint"],
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "diagnostic_fixture_reused": False,
        "truth_hidden_from_training_runner": True,
        "fixture_archive_path": private_root.joinpath("sealed-public-fixtures.npz").as_posix(),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_path": private_root.joinpath("sealed-public-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "selection_manifest_path": selection_path.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "public_release_eligible": False,
    }
    (REPO_ROOT / seal_path).write_bytes(canonical_json_bytes(seal))

    training = {
        "schema": "graphreader.ocr-graph-text-detector-training.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "experiment_ordinal": 1,
        "experiment_budget": EXPERIMENT_BUDGET,
        "architecture": "tiny-strided-encoder-decoder-probability-map-v1",
        "trigger": (
            "The exact official PP-OCRv5 detector produced strict probabilities but only 0.22916666666666666 "
            "text detection exact and 0.4861111111111111 composition exact on the single frozen diagnostic."
        ),
        "isolated_change": (
            "replace detector weights and architecture with an export-safe project-trained strided encoder-decoder; "
            "retain exact BGR normalization and DB postprocessing thresholds"
        ),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "onnx_parity_tolerance": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "selection_manifest_path": selection_path.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "sealed_public_test_seal_path": seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "expected_runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
        "public_gate_evaluations": 0,
        "diagnostic_fixture_reused": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / training_path).write_bytes(canonical_json_bytes(training))
    return {
        "diagnostic_result_sha256": sha256_file(REPO_ROOT / diagnostic_result_path),
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "training_config_sha256": sha256_file(REPO_ROOT / training_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--diagnostic-result", type=Path, default=DEFAULT_DIAGNOSTIC_RESULT)
    parser.add_argument("--diagnostic-report", type=Path, default=DEFAULT_DIAGNOSTIC_REPORT)
    arguments = parser.parse_args()
    result = freeze_split(
        private_root=arguments.private_root,
        protocol_path=arguments.protocol,
        selection_path=arguments.selection,
        seal_path=arguments.seal,
        training_path=arguments.training,
        diagnostic_result_path=arguments.diagnostic_result,
        diagnostic_report_path=arguments.diagnostic_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
