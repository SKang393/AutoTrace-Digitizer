# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Materialize and seal the fresh mutually feasible V6 splits exactly once."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.center.feasible_dense_v6.dataset import (
    PUBLIC_DATASET_SEED,
    SPLITS,
    feasibility_summary,
    render_split,
    write_archive,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/feasible_dense_v6"
OUTPUT_ROOT = ROOT / "artifacts/splits"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
PUBLIC_DATASET_PATH = ROOT / "PUBLIC_DATASET_MANIFEST.json"
PUBLIC_SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
FREEZE_REPORT_PATH = ROOT / "SPLIT_FREEZE_REPORT.json"
SOURCE_PATHS = (
    Path("ml/markers/center/feasible_dense_v6/dataset.py"),
    Path("ml/markers/center/feasible_dense_v6/prepare_split.py"),
)
REVISION = "marker-center-feasible-dense-v6"


def _record(scene) -> dict[str, object]:
    return {
        "fixture_id": scene.scene_id,
        "renderer_family": scene.renderer_family,
        "degradation_family": scene.degradation_family,
        "source_sha256": scene.source_sha256,
        "ground_truth_sha256": scene.ground_truth_sha256,
    }


def prepare() -> dict[str, object]:
    tracked_outputs = (SELECTION_PATH, PUBLIC_DATASET_PATH, PUBLIC_SEAL_PATH, FREEZE_REPORT_PATH)
    if OUTPUT_ROOT.exists() or any(path.exists() for path in tracked_outputs):
        raise RuntimeError("Feasible dense V6 splits or seals already exist")
    train = render_split("train")
    validation = render_split("validation")
    sealed_public = render_split("sealed_public")
    train_summary = feasibility_summary(train)
    validation_summary = feasibility_summary(validation)
    public_summary = feasibility_summary(sealed_public)
    train_path = OUTPUT_ROOT / "train.npz"
    validation_path = OUTPUT_ROOT / "validation.npz"
    public_path = OUTPUT_ROOT / "sealed-public.npz"
    train_sha256 = write_archive(train_path, train)
    validation_sha256 = write_archive(validation_path, validation)
    public_sha256 = write_archive(public_path, sealed_public)
    public_dataset = {
        "schema": "graphreader.marker-center-feasible-public-dataset.v6",
        "revision": REVISION,
        "scope": "public_synthetic_truth_hidden",
        "seed": PUBLIC_DATASET_SEED,
        "fixture_count": len(sealed_public),
        "fixtures": [_record(scene) for scene in sealed_public],
        "feasibility": public_summary,
        "case_truth_values_emitted": False,
        "fixture_pixels_emitted": False,
        "private_data": False,
        "chandler_used": False,
    }
    PUBLIC_DATASET_PATH.write_bytes(canonical_json_bytes(public_dataset))
    public_seal = {
        "schema": "graphreader.marker-center-feasible-public-seal.v6",
        "revision": REVISION,
        "fixture_archive_path": public_path.relative_to(REPO_ROOT).as_posix(),
        "fixture_archive_sha256": public_sha256,
        "fixture_count": len(sealed_public),
        "fixture_ids": [scene.scene_id for scene in sealed_public],
        "dataset_manifest_path": PUBLIC_DATASET_PATH.relative_to(REPO_ROOT).as_posix(),
        "dataset_manifest_sha256": sha256_file(PUBLIC_DATASET_PATH),
        "feasibility": public_summary,
        "public_gate_archive_opened": False,
        "public_gate_evaluations": 0,
        "production_approval": False,
        "release_eligible": False,
    }
    PUBLIC_SEAL_PATH.write_bytes(canonical_json_bytes(public_seal))
    selection = {
        "schema": "graphreader.marker-center-feasible-selection.v6",
        "revision": REVISION,
        "train": {
            **SPLITS["train"],
            "archive_path": train_path.relative_to(REPO_ROOT).as_posix(),
            "archive_sha256": train_sha256,
            "feasibility": train_summary,
        },
        "validation": {
            **SPLITS["validation"],
            "archive_path": validation_path.relative_to(REPO_ROOT).as_posix(),
            "archive_sha256": validation_sha256,
            "feasibility": validation_summary,
        },
        "sealed_public": {
            **SPLITS["sealed_public"],
            "archive_path": public_path.relative_to(REPO_ROOT).as_posix(),
            "archive_sha256": public_sha256,
            "feasibility": public_summary,
        },
    }
    SELECTION_PATH.write_bytes(canonical_json_bytes(selection))
    report = {
        "schema": "graphreader.marker-center-feasible-split-freeze.v6",
        "revision": REVISION,
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
        "generator_source_paths": [path.as_posix() for path in SOURCE_PATHS],
        "generator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, SOURCE_PATHS),
        "selection_manifest_sha256": sha256_file(SELECTION_PATH),
        "public_dataset_manifest_sha256": sha256_file(PUBLIC_DATASET_PATH),
        "sealed_public_test_seal_sha256": sha256_file(PUBLIC_SEAL_PATH),
        "train_archive_sha256": train_sha256,
        "validation_archive_sha256": validation_sha256,
        "sealed_public_fixture_archive_sha256": public_sha256,
        "train_feasibility": train_summary,
        "validation_feasibility": validation_summary,
        "sealed_public_feasibility": public_summary,
        "public_gate_archive_opened": False,
        "public_gate_evaluations": 0,
        "private_data": False,
        "chandler_used": False,
        "production_approval": False,
        "release_eligible": False,
    }
    FREEZE_REPORT_PATH.write_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    print(json.dumps(prepare(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
