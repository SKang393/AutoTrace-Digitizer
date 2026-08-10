# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze selection metadata and a truth-hidden line-aware public split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets

from ml.markers.center.line_aware_v1.dataset import (
    DATASET_REVISION,
    DEGRADATIONS,
    SEALED_PUBLIC_FAMILIES,
    build_sealed_public_scenes,
    save_sealed_public_archive,
    selection_manifest,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PRIVATE_ROOT = Path("ml/markers/center/artifacts/line-aware-v1/split-freeze")
DEFAULT_SELECTION_MANIFEST = Path("ml/markers/center/line_aware_v1/SELECTION_MANIFEST.json")
DEFAULT_SEALED_TEST_SEAL = Path("ml/markers/center/line_aware_v1/SEALED_PUBLIC_TEST_SEAL.json")
SOURCE_PATHS = (
    Path("ml/markers/center/line_aware_v1/dataset.py"),
    Path("ml/markers/center/line_aware_v1/prepare_split.py"),
)


def freeze_split(
    *, private_root: Path, selection_manifest_path: Path, sealed_test_seal_path: Path
) -> dict[str, object]:
    private = REPO_ROOT / private_root
    selection_path = REPO_ROOT / selection_manifest_path
    seal_path = REPO_ROOT / sealed_test_seal_path
    seed_path = private / "private-seed.json"
    archive_path = private / "sealed-public-fixtures.npz"
    private_manifest_path = private / "sealed-public-private-manifest.json"
    targets = (seed_path, archive_path, private_manifest_path, selection_path, seal_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError("Split freeze refuses to overwrite existing evidence: " + ", ".join(existing))
    private.mkdir(parents=True, exist_ok=False)
    secret_seed = secrets.randbelow(2_000_000_000) + 100_000
    seed_path.write_bytes(canonical_json_bytes({
        "schema": "graphreader.marker-center-private-split-seed.v1",
        "dataset_revision": DATASET_REVISION,
        "seed": secret_seed,
    }))
    selection_path.write_bytes(canonical_json_bytes(selection_manifest()))
    scenes = build_sealed_public_scenes(secret_seed)
    private_manifest = save_sealed_public_archive(scenes, archive_path)
    private_manifest_path.write_bytes(canonical_json_bytes(private_manifest))
    seal = {
        "schema": "graphreader.marker-center-line-aware-sealed-test-seal.v1",
        "task": "marker-center",
        "revision": "marker-center-line-aware-v1",
        "dataset_revision": DATASET_REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(scenes),
        "family_ids": list(SEALED_PUBLIC_FAMILIES),
        "degradation_ids": list(DEGRADATIONS["sealed_public"]),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "truth_hidden_from_training_runner": True,
        "fixture_archive_path": private_root.joinpath("sealed-public-fixtures.npz").as_posix(),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_path": private_root.joinpath("sealed-public-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "private_seed_sha256": sha256_file(seed_path),
        "selection_manifest_path": selection_manifest_path.as_posix(),
        "selection_manifest_sha256": sha256_file(selection_path),
        "split_generator_source_paths": [path.as_posix() for path in SOURCE_PATHS],
        "split_generator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, SOURCE_PATHS),
        "public_release_eligible": False,
    }
    seal_path.write_bytes(canonical_json_bytes(seal))
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--sealed-test-seal", type=Path, default=DEFAULT_SEALED_TEST_SEAL)
    args = parser.parse_args()
    result = freeze_split(
        private_root=args.private_root,
        selection_manifest_path=args.selection_manifest,
        sealed_test_seal_path=args.sealed_test_seal,
    )
    print(json.dumps({key: result[key] for key in (
        "scene_count", "fixture_archive_sha256", "private_manifest_sha256", "selection_manifest_sha256"
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
