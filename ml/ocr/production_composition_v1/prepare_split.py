# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze OCR production-composition validation and truth-hidden public fixtures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256

from .dataset import build_split, proposal_summary, save_sealed_archive, split_fingerprint
from .protocol import PUBLIC_REVISION, REVISION, TASK, VALIDATION_REVISION, protocol_configuration
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from .validation_gate import (
    EVALUATOR_SOURCE_PATHS as VALIDATION_EVALUATOR_SOURCE_PATHS,
    GATE_CONFIG as VALIDATION_GATE_CONFIG,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIVATE_ROOT = Path("ml/ocr/production_composition_v1/artifacts/composition-v1-final-freeze")
DEFAULT_PROTOCOL = Path("ml/ocr/production_composition_v1/PROTOCOL.json")
DEFAULT_SELECTION = Path("ml/ocr/production_composition_v1/VALIDATION_SEAL.json")
DEFAULT_SEAL = Path("ml/ocr/production_composition_v1/SEALED_PUBLIC_TEST_SEAL.json")
DEFAULT_GATE = Path("ml/ocr/production_composition_v1/gates/sealed-public-v1.json")
DEFAULT_VALIDATION_GATE = Path("ml/ocr/production_composition_v1/gates/validation-v1.json")
SPLIT_SOURCE_PATHS = (
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/production_composition_v1/prepare_split.py"),
    Path("ml/ocr/production_composition_v1/protocol.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
)


def freeze_split(
    *,
    private_root: Path,
    protocol_path: Path,
    validation_seal_path: Path,
    seal_path: Path,
    gate_path: Path,
    validation_gate_path: Path,
) -> dict[str, str]:
    private = REPO_ROOT / private_root
    targets = tuple(
        REPO_ROOT / path
        for path in (protocol_path, validation_seal_path, seal_path, gate_path, validation_gate_path)
    )
    validation_archive_path = private / "validation-fixtures.zip"
    validation_private_manifest_path = private / "validation-private-manifest.json"
    public_archive_path = private / "sealed-public-fixtures.zip"
    public_private_manifest_path = private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in (*targets, private) if path.exists()]
    if existing:
        raise RuntimeError("OCR composition split freeze refuses to overwrite evidence: " + ", ".join(existing))
    private.mkdir(parents=True, exist_ok=False)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    protocol = protocol_configuration()
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SPLIT_SOURCE_PATHS]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    (REPO_ROOT / protocol_path).write_bytes(canonical_json_bytes(protocol))

    validation = build_split("validation")
    sealed_public = build_split("sealed_public")
    validation_summary = proposal_summary(validation)
    public_summary = proposal_summary(sealed_public)
    validation_private = save_sealed_archive(validation, validation_archive_path)
    validation_private_manifest_path.write_bytes(canonical_json_bytes(validation_private))
    public_private = save_sealed_archive(sealed_public, public_archive_path)
    public_private_manifest_path.write_bytes(canonical_json_bytes(public_private))
    validation_seal = {
        "schema": "graphreader.ocr-production-composition-validation-seal.v1",
        "task": TASK,
        "revision": REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        **validation_summary,
        "split_fingerprint": split_fingerprint(validation),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_fixture_bytes_reused": False,
        "validation_model_execution_count": 0,
        "fixture_archive_path": private_root.joinpath("validation-fixtures.zip").as_posix(),
        "fixture_archive_sha256": sha256_file(validation_archive_path),
        "private_manifest_path": private_root.joinpath("validation-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(validation_private_manifest_path),
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / validation_seal_path).write_bytes(canonical_json_bytes(validation_seal))
    validation_gate = {
        "schema": "graphreader.ocr-production-composition-validation-gate-config.v1",
        "task": TASK,
        "revision": VALIDATION_REVISION,
        "expected_candidate_hash_keys": [
            "detector_onnx_sha256",
            "official_recognizer_onnx_sha256",
            "numeric_recognizer_onnx_sha256",
        ],
        "validation_seal_path": validation_seal_path.as_posix(),
        "validation_seal_sha256": sha256_file(REPO_ROOT / validation_seal_path),
        "expected_dataset_manifest_sha256": sha256_file(validation_private_manifest_path),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(
            REPO_ROOT, VALIDATION_EVALUATOR_SOURCE_PATHS
        ),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(VALIDATION_GATE_CONFIG)),
        "evaluation_limit": 1,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / validation_gate_path).write_bytes(canonical_json_bytes(validation_gate))
    seal = {
        "schema": "graphreader.ocr-production-composition-sealed-test-seal.v1",
        "task": TASK,
        "revision": REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        **public_summary,
        "split_fingerprint": split_fingerprint(sealed_public),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_public_archive_reused": False,
        "prior_public_sample_or_pixel_inspection_used": False,
        "truth_hidden_from_model_execution_until_gate": True,
        "fixture_archive_path": private_root.joinpath("sealed-public-fixtures.zip").as_posix(),
        "fixture_archive_sha256": sha256_file(public_archive_path),
        "private_manifest_path": private_root.joinpath("sealed-public-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(public_private_manifest_path),
        "validation_seal_path": validation_seal_path.as_posix(),
        "validation_seal_sha256": sha256_file(REPO_ROOT / validation_seal_path),
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / seal_path).write_bytes(canonical_json_bytes(seal))
    gate = {
        "schema": "graphreader.ocr-production-composition-gate-config.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": [
            "detector_onnx_sha256",
            "official_recognizer_onnx_sha256",
            "numeric_recognizer_onnx_sha256",
            "validation_report_sha256",
        ],
        "sealed_public_test_seal_path": seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "expected_dataset_manifest_sha256": sha256_file(public_private_manifest_path),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / gate_path).write_bytes(canonical_json_bytes(gate))
    return {
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "validation_seal_sha256": sha256_file(REPO_ROOT / validation_seal_path),
        "validation_fixture_archive_sha256": sha256_file(validation_archive_path),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "sealed_public_fixture_archive_sha256": sha256_file(public_archive_path),
        "gate_config_sha256": sha256_file(REPO_ROOT / gate_path),
        "validation_gate_config_sha256": sha256_file(REPO_ROOT / validation_gate_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--validation-seal", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--validation-gate", type=Path, default=DEFAULT_VALIDATION_GATE)
    arguments = parser.parse_args()
    result = freeze_split(
        private_root=arguments.private_root,
        protocol_path=arguments.protocol,
        validation_seal_path=arguments.validation_seal,
        seal_path=arguments.seal,
        gate_path=arguments.gate,
        validation_gate_path=arguments.validation_gate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
