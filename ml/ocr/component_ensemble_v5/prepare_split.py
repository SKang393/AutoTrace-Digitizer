# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze OCR V5 selection metadata and a new truth-hidden public split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256

from .dataset import build_split, save_sealed_public_archive, split_fingerprint
from .protocol import (
    ALPHABET,
    BATCH_SIZE,
    EPOCHS,
    EXPERIMENT_BUDGET,
    LEARNING_RATE,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    PUBLIC_REVISION,
    REVISION,
    SEED,
    TASK,
    THRESHOLDS,
    WEIGHT_DECAY,
    protocol_configuration,
)
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from .train_p1 import RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIVATE_ROOT = Path("ml/ocr/component_ensemble_v5/artifacts/split-freeze")
DEFAULT_PROTOCOL = Path("ml/ocr/component_ensemble_v5/PROTOCOL.json")
DEFAULT_SELECTION = Path("ml/ocr/component_ensemble_v5/SELECTION_MANIFEST.json")
DEFAULT_SEAL = Path("ml/ocr/component_ensemble_v5/SEALED_PUBLIC_TEST_SEAL.json")
DEFAULT_GATE = Path("ml/ocr/component_ensemble_v5/gates/sealed-public-v1.json")
DEFAULT_TRAINING = Path("ml/ocr/component_ensemble_v5/training/p1.json")
SPLIT_SOURCE_PATHS = (
    Path("ml/ocr/component_ensemble_v5/dataset.py"),
    Path("ml/ocr/component_ensemble_v5/prepare_split.py"),
    Path("ml/ocr/component_ensemble_v5/protocol.py"),
    Path("ml/ocr/component_geometric_v4/dataset.py"),
)


def freeze_split(
    *,
    private_root: Path,
    protocol_path: Path,
    selection_path: Path,
    seal_path: Path,
    gate_path: Path,
    training_path: Path,
) -> dict[str, object]:
    private = REPO_ROOT / private_root
    targets = tuple(REPO_ROOT / path for path in (protocol_path, selection_path, seal_path, gate_path, training_path))
    archive_path = private / "sealed-public-fixtures.npz"
    private_manifest_path = private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in (*targets, private, archive_path, private_manifest_path) if path.exists()]
    if existing:
        raise RuntimeError("OCR V5 split freeze refuses to overwrite evidence: " + ", ".join(existing))
    private.mkdir(parents=True, exist_ok=False)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    protocol = protocol_configuration()
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SPLIT_SOURCE_PATHS]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    (REPO_ROOT / protocol_path).write_bytes(canonical_json_bytes(protocol))

    training_samples = build_split("train")
    validation_samples = build_split("validation")
    sealed_samples = build_split("sealed_public")
    selection = {
        "schema": "graphreader.ocr-component-ensemble-selection.v1",
        "task": TASK,
        "revision": REVISION,
        "train_sample_count": len(training_samples),
        "validation_sample_count": len(validation_samples),
        "train_split_fingerprint": split_fingerprint(training_samples),
        "validation_split_fingerprint": split_fingerprint(validation_samples),
        "alphabet": ALPHABET,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "exposed_predecessor_cases_used_for_selection": False,
        "sealed_public_truth_available_to_training": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / selection_path).write_bytes(canonical_json_bytes(selection))
    private_manifest = save_sealed_public_archive(sealed_samples, archive_path)
    private_manifest_path.write_bytes(canonical_json_bytes(private_manifest))
    seal = {
        "schema": "graphreader.ocr-component-ensemble-sealed-test-seal.v1",
        "task": TASK,
        "revision": REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(sealed_samples),
        "positive_count": private_manifest["positive_count"],
        "exclusion_count": private_manifest["exclusion_count"],
        "split_fingerprint": private_manifest["split_fingerprint"],
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "predecessor_public_archive_reused": False,
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
    gate = {
        "schema": "graphreader.ocr-component-ensemble-gate-config.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": ["onnx_sha256", "selection_report_sha256"],
        "sealed_public_test_seal_path": seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "expected_dataset_manifest_sha256": sha256_file(private_manifest_path),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / gate_path).write_bytes(canonical_json_bytes(gate))
    training = {
        "schema": "graphreader.ocr-component-ensemble-training.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": "P1",
        "experiment_ordinal": 1,
        "experiment_budget": EXPERIMENT_BUDGET,
        "architecture": "multi-renderer-fixed-feature-ensemble-mlp-v1",
        "trigger": (
            "The exhausted structural-filter revision passed public role and exclusion gates but failed exact match "
            "and CER across independently rendered decimal-bearing families."
        ),
        "isolated_change": (
            "train one per-glyph fixed-feature ensemble across all reviewed Noto weights and new renderer families; "
            "add multiscale pooling, edge profiles, darkness, and aspect evidence while retaining the preregistered "
            "structural rejection rule"
        ),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "selection_thresholds": list(THRESHOLDS),
        "onnx_parity_tolerance": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "selection_manifest_path": selection_path.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "sealed_public_test_seal_path": seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "expected_runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
        "public_gate_evaluations": 0,
        "exposed_predecessor_cases_used_for_selection": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / training_path).write_bytes(canonical_json_bytes(training))
    return {
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "gate_config_sha256": sha256_file(REPO_ROOT / gate_path),
        "training_config_sha256": sha256_file(REPO_ROOT / training_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    arguments = parser.parse_args()
    result = freeze_split(
        private_root=arguments.private_root,
        protocol_path=arguments.protocol,
        selection_path=arguments.selection,
        seal_path=arguments.seal,
        gate_path=arguments.gate,
        training_path=arguments.training,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
