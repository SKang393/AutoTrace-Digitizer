# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Materialize one fresh V8 split before any optimizer or model execution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.center.mask_consensus_v8.dataset import (
    PUBLIC_DATASET_SEED,
    SPLITS,
    feasibility_summary,
    render_split,
    write_archive,
)
from ml.markers.center.mask_consensus_v8.protocol import (
    DESIGN_SOURCE_PATHS,
    REVISION,
    ROOT,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPO_ROOT / "ml/markers/center/artifacts/mask-consensus-v8/split-freeze"
SELECTION_PATH = REPO_ROOT / ROOT / "SELECTION_MANIFEST.json"
PUBLIC_DATASET_PATH = REPO_ROOT / ROOT / "PUBLIC_DATASET_MANIFEST.json"
PUBLIC_SEAL_PATH = REPO_ROOT / ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
FREEZE_REPORT_PATH = REPO_ROOT / ROOT / "SPLIT_FREEZE_REPORT.json"


def _record(scene: object) -> dict[str, object]:
    return {
        "fixture_id": scene.scene_id,
        "image_sha256": scene.source_sha256,
        "ground_truth_sha256": scene.ground_truth_sha256,
        "renderer_family": scene.renderer_family,
        "degradation_family": scene.degradation_family,
    }


def prepare() -> dict[str, object]:
    tracked_outputs = (
        SELECTION_PATH,
        PUBLIC_DATASET_PATH,
        PUBLIC_SEAL_PATH,
        FREEZE_REPORT_PATH,
    )
    if any(path.exists() for path in tracked_outputs) or OUTPUT_ROOT.exists():
        raise RuntimeError("Marker-center V8 splits or seals already exist")
    train = render_split("train")
    validation = render_split("validation")
    sealed_public = render_split("sealed_public")
    all_hashes = [scene.source_sha256 for scenes in (train, validation, sealed_public) for scene in scenes]
    if len(all_hashes) != len(set(all_hashes)):
        raise RuntimeError("Marker-center V8 source bytes overlap across splits")
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
        "schema": "graphreader.marker-center-mask-consensus-public-dataset.v8",
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
        "schema": "graphreader.marker-center-mask-consensus-public-seal.v8",
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
        "schema": "graphreader.marker-center-mask-consensus-selection.v8",
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
        "schema": "graphreader.marker-center-mask-consensus-split-freeze.v8",
        "revision": REVISION,
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
        "generator_source_paths": [path.as_posix() for path in DESIGN_SOURCE_PATHS],
        "generator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, DESIGN_SOURCE_PATHS),
        "selection_manifest_sha256": sha256_file(SELECTION_PATH),
        "public_dataset_manifest_sha256": sha256_file(PUBLIC_DATASET_PATH),
        "sealed_public_test_seal_sha256": sha256_file(PUBLIC_SEAL_PATH),
        "train_archive_sha256": train_sha256,
        "validation_archive_sha256": validation_sha256,
        "sealed_public_fixture_archive_sha256": public_sha256,
        "cross_split_source_overlap_count": 0,
        "train_feasibility": train_summary,
        "validation_feasibility": validation_summary,
        "sealed_public_feasibility": public_summary,
        "model_execution_count_at_freeze": 0,
        "optimizer_step_count_at_freeze": 0,
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
