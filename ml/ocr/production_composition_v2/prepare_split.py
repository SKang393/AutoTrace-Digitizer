# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze fresh validation and truth-hidden public composition V2 fixtures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256

from .dataset import build_split, proposal_summary, save_sealed_archive, split_fingerprint
from .protocol import PUBLIC_REVISION, REVISION, TASK, VALIDATION_REVISION, protocol_configuration
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from .validation_gate import EVALUATOR_SOURCE_PATHS as VALIDATION_SOURCES, GATE_CONFIG as VALIDATION_GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/production_composition_v2")
DEFAULT_PRIVATE_ROOT = ROOT / "artifacts/composition-v2-exact-execution-freeze"
DEFAULT_PROTOCOL = ROOT / "PROTOCOL.json"
DEFAULT_VALIDATION_SEAL = ROOT / "VALIDATION_SEAL.json"
DEFAULT_PUBLIC_SEAL = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
DEFAULT_GATE = ROOT / "gates/sealed-public-v1.json"
DEFAULT_VALIDATION_GATE = ROOT / "gates/validation-v1.json"
SPLIT_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "prepare_split.py", ROOT / "protocol.py",
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
)


def freeze_split(
    *, private_root: Path = DEFAULT_PRIVATE_ROOT, protocol_path: Path = DEFAULT_PROTOCOL,
    validation_seal_path: Path = DEFAULT_VALIDATION_SEAL, public_seal_path: Path = DEFAULT_PUBLIC_SEAL,
    gate_path: Path = DEFAULT_GATE, validation_gate_path: Path = DEFAULT_VALIDATION_GATE,
) -> dict[str, str]:
    private = REPO_ROOT / private_root
    targets = tuple(REPO_ROOT / path for path in (protocol_path, validation_seal_path, public_seal_path, gate_path, validation_gate_path))
    validation_archive = private / "validation-fixtures.zip"
    validation_manifest = private / "validation-private-manifest.json"
    public_archive = private / "sealed-public-fixtures.zip"
    public_manifest = private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in (*targets, private) if path.exists()]
    if existing:
        raise RuntimeError("OCR composition V2 freeze refuses to overwrite evidence: " + ", ".join(existing))
    private.mkdir(parents=True, exist_ok=False)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    protocol = protocol_configuration()
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SPLIT_SOURCE_PATHS]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    (REPO_ROOT / protocol_path).write_bytes(canonical_json_bytes(protocol))
    validation = build_split("validation")
    public = build_split("sealed_public")
    validation_private = save_sealed_archive(validation, validation_archive)
    validation_manifest.write_bytes(canonical_json_bytes(validation_private))
    public_private = save_sealed_archive(public, public_archive)
    public_manifest.write_bytes(canonical_json_bytes(public_private))
    common = {
        "task": TASK, "revision": REVISION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "predecessor_fixture_bytes_reused": False,
        "protocol_path": protocol_path.as_posix(), "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "production_approval": False, "release_eligible": False,
    }
    validation_seal = {
        "schema": "graphreader.ocr-production-composition-validation-seal.v2", **common,
        **proposal_summary(validation), "split_fingerprint": split_fingerprint(validation),
        "validation_model_execution_count": 0,
        "fixture_archive_path": private_root.joinpath("validation-fixtures.zip").as_posix(),
        "fixture_archive_sha256": sha256_file(validation_archive),
        "private_manifest_path": private_root.joinpath("validation-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(validation_manifest),
    }
    (REPO_ROOT / validation_seal_path).write_bytes(canonical_json_bytes(validation_seal))
    validation_gate = {
        "schema": "graphreader.ocr-production-composition-validation-gate-config.v2",
        "task": TASK, "revision": VALIDATION_REVISION,
        "expected_candidate_hash_keys": [
            "detector_onnx_sha256", "official_recognizer_onnx_sha256",
            "numeric_recognizer_onnx_sha256", "spacing_source_sha256",
        ],
        "validation_seal_path": validation_seal_path.as_posix(),
        "validation_seal_sha256": sha256_file(REPO_ROOT / validation_seal_path),
        "expected_dataset_manifest_sha256": sha256_file(validation_manifest),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, VALIDATION_SOURCES),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(VALIDATION_GATE_CONFIG)),
        "evaluation_limit": 1, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / validation_gate_path).write_bytes(canonical_json_bytes(validation_gate))
    public_seal = {
        "schema": "graphreader.ocr-production-composition-sealed-test-seal.v2", **common,
        **proposal_summary(public), "split_fingerprint": split_fingerprint(public),
        "prior_public_sample_or_pixel_inspection_used": False,
        "truth_hidden_from_model_execution_until_gate": True,
        "fixture_archive_path": private_root.joinpath("sealed-public-fixtures.zip").as_posix(),
        "fixture_archive_sha256": sha256_file(public_archive),
        "private_manifest_path": private_root.joinpath("sealed-public-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(public_manifest),
        "validation_seal_path": validation_seal_path.as_posix(),
        "validation_seal_sha256": sha256_file(REPO_ROOT / validation_seal_path),
    }
    (REPO_ROOT / public_seal_path).write_bytes(canonical_json_bytes(public_seal))
    public_gate = {
        "schema": "graphreader.ocr-production-composition-gate-config.v2",
        "task": TASK, "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": [
            "detector_onnx_sha256", "official_recognizer_onnx_sha256",
            "numeric_recognizer_onnx_sha256", "spacing_source_sha256", "validation_report_sha256",
        ],
        "sealed_public_test_seal_path": public_seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / public_seal_path),
        "expected_dataset_manifest_sha256": sha256_file(public_manifest),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / gate_path).write_bytes(canonical_json_bytes(public_gate))
    return {
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "validation_seal_sha256": sha256_file(REPO_ROOT / validation_seal_path),
        "validation_fixture_archive_sha256": sha256_file(validation_archive),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / public_seal_path),
        "sealed_public_fixture_archive_sha256": sha256_file(public_archive),
        "validation_gate_config_sha256": sha256_file(REPO_ROOT / validation_gate_path),
        "public_gate_config_sha256": sha256_file(REPO_ROOT / gate_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(freeze_split(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
