# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze ignore-band selection metadata and a truth-hidden public split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256

from .dataset import build_validation_split, split_fingerprint, training_split_fingerprint
from .protocol import (
    BATCH_SIZE,
    CANDIDATE_ID,
    DB_SHRINK_RATIO,
    DICE_LOSS_WEIGHT,
    EMPTY_TARGET_NEGATIVE_PIXELS,
    EPOCHS,
    EXPERIMENT_BUDGET,
    IGNORE_BAND_EXPANSION_PIXELS,
    LEARNING_RATE,
    MINIMUM_NEGATIVE_PIXELS,
    NEGATIVE_TO_POSITIVE_RATIO,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    REVISION,
    SEED,
    TASK,
    TRAIN_SAMPLE_COUNT,
    WEIGHT_DECAY,
    protocol_configuration,
)
from .sealed_dataset import build_sealed_public_split, save_sealed_public_archive
from .train_p1 import RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIVATE_ROOT = Path("ml/ocr/graph_text_ignore_band_v3/artifacts/split-freeze")
DEFAULT_PROTOCOL = Path("ml/ocr/graph_text_ignore_band_v3/PROTOCOL.json")
DEFAULT_SELECTION = Path("ml/ocr/graph_text_ignore_band_v3/SELECTION_MANIFEST.json")
DEFAULT_SEAL = Path("ml/ocr/graph_text_ignore_band_v3/SEALED_PUBLIC_TEST_SEAL.json")
DEFAULT_PREREGISTRATION = Path("ml/ocr/graph_text_ignore_band_v3/P1_PREREGISTRATION.json")
DEFAULT_TRAINING = Path("ml/ocr/graph_text_ignore_band_v3/training/p1.json")
DEFAULT_TRIGGER = Path("ml/ocr/graph_text_balanced_v2/P3_RESULT.json")
SPLIT_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_ignore_band_v3/dataset.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/prepare_split.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/protocol.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/sealed_dataset.py"),
)


def _validate_trigger(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "graphreader.ocr-graph-text-balanced-recall-result.v1",
        "task": TASK,
        "revision": "graph-text-balanced-recall-v2",
        "candidate_id": "P3",
        "status": "failed_selection",
        "production_approval": False,
        "release_eligible": False,
        "sealed_public_archive_opened": False,
        "public_gate_evaluations": 0,
        "selection_gate_passed": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(f"Ignore-band trigger field changed: {key}")
    metrics = value.get("selection_metrics")
    expected_metrics = {
        "fixture_count": 96,
        "exact_fixture_count": 30,
        "text_fixture_count": 72,
        "text_exact_fixture_count": 24,
        "text_missed_fixture_count": 13,
        "false_region_count": 82,
        "exclusion_false_region_count": 20,
        "duplicate_region_count": 0,
    }
    if not isinstance(metrics, dict):
        raise RuntimeError("Ignore-band trigger metrics are missing")
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise RuntimeError(f"Ignore-band trigger metric changed: {key}")
    return sha256_file(path)


def freeze_split(
    *,
    private_root: Path,
    protocol_path: Path,
    selection_path: Path,
    seal_path: Path,
    preregistration_path: Path,
    training_path: Path,
    trigger_path: Path,
) -> dict[str, object]:
    private = REPO_ROOT / private_root
    tracked_targets = tuple(
        REPO_ROOT / path
        for path in (protocol_path, selection_path, seal_path, preregistration_path, training_path)
    )
    archive_path = private / "sealed-public-fixtures.npz"
    private_manifest_path = private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in (*tracked_targets, private) if path.exists()]
    if existing:
        raise RuntimeError("Ignore-band split freeze refuses to overwrite evidence: " + ", ".join(existing))
    trigger_sha256 = _validate_trigger(REPO_ROOT / trigger_path)
    private.mkdir(parents=True, exist_ok=False)
    for target in tracked_targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    protocol = protocol_configuration()
    protocol["trigger"]["prior_result_path"] = trigger_path.as_posix()
    protocol["trigger"]["prior_result_sha256"] = trigger_sha256
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SPLIT_SOURCE_PATHS]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    (REPO_ROOT / protocol_path).write_bytes(canonical_json_bytes(protocol))

    validation = build_validation_split()
    train_fingerprint = training_split_fingerprint()
    validation_fingerprint = split_fingerprint(validation)
    selection = {
        "schema": "graphreader.ocr-graph-text-ignore-band-selection.v1",
        "task": TASK,
        "revision": REVISION,
        "train_sample_count": TRAIN_SAMPLE_COUNT,
        "validation_sample_count": len(validation),
        "train_split_fingerprint": train_fingerprint,
        "validation_split_fingerprint": validation_fingerprint,
        "trigger_result_path": trigger_path.as_posix(),
        "trigger_result_sha256": trigger_sha256,
        "prior_selection_fixture_reused": False,
        "prior_sealed_fixture_reused": False,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "sealed_public_truth_available_to_training": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / selection_path).write_bytes(canonical_json_bytes(selection))

    sealed = build_sealed_public_split()
    private_manifest = save_sealed_public_archive(sealed, archive_path)
    private_manifest_path.write_bytes(canonical_json_bytes(private_manifest))
    seal = {
        "schema": "graphreader.ocr-graph-text-ignore-band-sealed-test-seal.v1",
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
        "prior_selection_fixture_reused": False,
        "prior_sealed_fixture_reused": False,
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

    isolated_change = (
        "retain the V2 export-safe topology and production DB thresholds, but replace contradictory full-background "
        "supervision with a one-source-pixel dilated ignore band around each text polygon, masked Dice, and frozen "
        "3:1 online hard-negative mining on new disjoint renderer and degradation families"
    )
    training = {
        "schema": "graphreader.ocr-graph-text-ignore-band-training.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "experiment_ordinal": 1,
        "experiment_budget": EXPERIMENT_BUDGET,
        "architecture": "skip-connected-probability-map-v2",
        "trigger": (
            "Balanced-recall V2 exhausted three candidates. Final P3 passed probability and parity but produced "
            "82 false regions, 20 exclusion false regions, and 13 text misses across 96 selection fixtures."
        ),
        "isolated_change": isolated_change,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dice_loss_weight": DICE_LOSS_WEIGHT,
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "minimum_negative_pixels": MINIMUM_NEGATIVE_PIXELS,
        "empty_target_negative_pixels": EMPTY_TARGET_NEGATIVE_PIXELS,
        "ignore_band_expansion_pixels": IGNORE_BAND_EXPANSION_PIXELS,
        "db_shrink_ratio": DB_SHRINK_RATIO,
        "onnx_parity_tolerance": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "trigger_result_path": trigger_path.as_posix(),
        "trigger_result_sha256": trigger_sha256,
        "training_split_fingerprint": train_fingerprint,
        "selection_manifest_path": selection_path.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "sealed_public_test_seal_path": seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "expected_runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
        "public_gate_evaluations": 0,
        "prior_selection_fixture_reused": False,
        "prior_sealed_fixture_reused": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / training_path).write_bytes(canonical_json_bytes(training))
    preregistration = {
        "schema": "graphreader.ocr-graph-text-ignore-band-preregistration.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "experiment_ordinal": 1,
        "experiment_budget": EXPERIMENT_BUDGET,
        "trigger_result_path": trigger_path.as_posix(),
        "trigger_result_sha256": trigger_sha256,
        "candidate_config_path": training_path.as_posix(),
        "candidate_config_sha256": sha256_file(REPO_ROOT / training_path),
        "expected_runner_source_bundle_sha256": training["expected_runner_source_bundle_sha256"],
        "training_split_fingerprint": train_fingerprint,
        "validation_split_fingerprint": validation_fingerprint,
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "isolated_change": isolated_change,
        "public_gate_authorized": False,
        "public_gate_evaluations": 0,
        "sealed_public_archive_opened": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / preregistration_path).write_bytes(canonical_json_bytes(preregistration))
    return {
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "training_config_sha256": sha256_file(REPO_ROOT / training_path),
        "preregistration_sha256": sha256_file(REPO_ROOT / preregistration_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--trigger", type=Path, default=DEFAULT_TRIGGER)
    arguments = parser.parse_args()
    result = freeze_split(
        private_root=arguments.private_root,
        protocol_path=arguments.protocol,
        selection_path=arguments.selection,
        seal_path=arguments.seal,
        preregistration_path=arguments.preregistration,
        training_path=arguments.training,
        trigger_path=arguments.trigger,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
