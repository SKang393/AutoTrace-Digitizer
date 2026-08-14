# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Materialize fresh visible and truth-hidden dense-contract splits once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.center.dense_contract_v5.dataset import (
    PUBLIC_DATASET_SEED,
    SPLITS,
    render_split,
    tensor_stream_sha256,
    write_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/dense_contract_v5"
EVALUATOR_PATH = REPO_ROOT / "ml/markers/center/artifact_mask_public_gate.py"
PROFILE = "marker-center-artifact-mask-public-gate-v1"


def _record(scene) -> dict[str, object]:
    return {
        "fixture_id": scene.scene_id,
        "image_sha256": scene.source_sha256,
        "ground_truth_sha256": scene.ground_truth_sha256,
        "family": scene.renderer_family,
    }


def materialize() -> dict[str, object]:
    output_root = ROOT / "artifacts/splits"
    train = render_split("train")
    validation = render_split("validation")
    public = render_split("sealed_public")
    train_path = output_root / "train.npz"
    validation_path = output_root / "validation.npz"
    public_path = output_root / "sealed-public.npz"
    train_sha256 = write_archive(train_path, train)
    validation_sha256 = write_archive(validation_path, validation)
    public_sha256 = write_archive(public_path, public)
    dataset_manifest = {
        "schema": "graphreader.marker-artifact-mask-dataset.v1",
        "scope": "public_synthetic",
        "private_data": False,
        "chandler_used": False,
        "seed": PUBLIC_DATASET_SEED,
        "fixtures": [_record(scene) for scene in public],
    }
    dataset_path = ROOT / "PUBLIC_DATASET_MANIFEST.json"
    dataset_path.write_bytes(canonical_json_bytes(dataset_manifest))
    evaluator_sha256 = sha256_file(EVALUATOR_PATH)
    split_seal = {
        "schema": "graphreader.marker-artifact-mask-split-seal.v1",
        "profile": PROFILE,
        "sealed": True,
        "selection_locked_before_inference": True,
        "private_data": False,
        "chandler_used": False,
        "dataset_manifest_sha256": sha256_file(dataset_path),
        "evaluator_source_sha256": evaluator_sha256,
        "fixture_count": len(public),
        "fixture_ids": [scene.scene_id for scene in public],
        "fixture_archive_path": "ml/markers/center/dense_contract_v5/artifacts/splits/sealed-public.npz",
        "fixture_archive_sha256": public_sha256,
        "public_gate_evaluations": 0,
        "public_gate_archive_opened": False,
    }
    seal_path = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    seal_path.write_bytes(canonical_json_bytes(split_seal))
    selection = {
        "schema": "graphreader.marker-center-dense-contract-selection.v5",
        "task": "marker-center",
        "revision": "marker-center-dense-contract-v5",
        "private_data": False,
        "chandler_used": False,
        "train": {
            **SPLITS["train"],
            "archive_path": "ml/markers/center/dense_contract_v5/artifacts/splits/train.npz",
            "archive_sha256": train_sha256,
            "tensor_stream_sha256": tensor_stream_sha256(train),
        },
        "validation": {
            **SPLITS["validation"],
            "archive_path": "ml/markers/center/dense_contract_v5/artifacts/splits/validation.npz",
            "archive_sha256": validation_sha256,
            "tensor_stream_sha256": tensor_stream_sha256(validation),
        },
        "sealed_public": {
            **SPLITS["sealed_public"],
            "archive_sha256": public_sha256,
            "dataset_manifest_sha256": sha256_file(dataset_path),
            "split_seal_sha256": sha256_file(seal_path),
            "truth_hidden_from_candidate_runner": True,
        },
    }
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    selection_path.write_bytes(canonical_json_bytes(selection))
    return {
        "selection_manifest_sha256": sha256_file(selection_path),
        "public_dataset_manifest_sha256": sha256_file(dataset_path),
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "sealed_public_fixture_archive_sha256": public_sha256,
        "train_archive_sha256": train_sha256,
        "validation_archive_sha256": validation_sha256,
        "evaluator_source_sha256": evaluator_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(materialize(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
