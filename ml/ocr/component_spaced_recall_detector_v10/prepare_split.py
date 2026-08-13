# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze V10 train, selection, and hidden-public evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from .dataset import build_split, proposal_summary, save_sealed_public_archive, split_fingerprint
from .protocol import PUBLIC_REVISION, REVISION, TASK, protocol_configuration
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/component_spaced_recall_detector_v10")
PRIVATE_ROOT = ROOT / "artifacts/split-freeze"
PROTOCOL = ROOT / "PROTOCOL.json"
SELECTION = ROOT / "SELECTION_MANIFEST.json"
SEAL = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
GATE = ROOT / "gates/sealed-public-v1.json"
SOURCES = (
    ROOT / "dataset.py", ROOT / "prepare_split.py", ROOT / "protocol.py",
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
)


def freeze_split() -> dict[str, str]:
    private = REPO_ROOT / PRIVATE_ROOT
    targets = tuple(REPO_ROOT / path for path in (PROTOCOL, SELECTION, SEAL, GATE))
    archive, private_manifest = private / "sealed-public-fixtures.zip", private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in (*targets, private) if path.exists()]
    if existing:
        raise RuntimeError("OCR V10 split freeze refuses overwrite: " + ", ".join(existing))
    private.mkdir(parents=True, exist_ok=False)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    protocol = protocol_configuration()
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SOURCES]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SOURCES)
    (REPO_ROOT / PROTOCOL).write_bytes(canonical_json_bytes(protocol))
    train, validation, public = build_split("train"), build_split("validation"), build_split("sealed_public")
    selection = {
        "schema": "graphreader.ocr-spaced-component-recall-selection.v1", "task": TASK, "revision": REVISION,
        "train": {**proposal_summary(train), "split_fingerprint": split_fingerprint(train)},
        "validation": {**proposal_summary(validation), "split_fingerprint": split_fingerprint(validation)},
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "predecessor_fixture_bytes_reused": False,
        "prior_validation_pixels_used": False, "sealed_public_truth_available_to_candidate": False,
        "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / SELECTION).write_bytes(canonical_json_bytes(selection))
    private_value = save_sealed_public_archive(public, archive)
    private_manifest.write_bytes(canonical_json_bytes(private_value))
    seal = {
        "schema": "graphreader.ocr-spaced-component-recall-sealed-test-seal.v1", "task": TASK,
        "revision": REVISION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        **proposal_summary(public), "split_fingerprint": split_fingerprint(public),
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "predecessor_fixture_bytes_reused": False,
        "truth_hidden_from_candidate_runner": True,
        "fixture_archive_path": PRIVATE_ROOT.joinpath("sealed-public-fixtures.zip").as_posix(),
        "fixture_archive_sha256": sha256_file(archive),
        "private_manifest_path": PRIVATE_ROOT.joinpath("sealed-public-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(private_manifest),
        "selection_manifest_path": SELECTION.as_posix(), "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION),
        "protocol_path": PROTOCOL.as_posix(), "protocol_sha256": sha256_file(REPO_ROOT / PROTOCOL),
        "public_gate_evaluations": 0, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / SEAL).write_bytes(canonical_json_bytes(seal))
    gate = {
        "schema": "graphreader.ocr-spaced-component-recall-gate-config.v1", "task": TASK,
        "revision": PUBLIC_REVISION, "expected_candidate_hash_keys": ["onnx_sha256", "selection_report_sha256"],
        "sealed_public_test_seal_path": SEAL.as_posix(), "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL),
        "expected_dataset_manifest_sha256": sha256_file(private_manifest),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / GATE).write_bytes(canonical_json_bytes(gate))
    return {
        "protocol_sha256": sha256_file(REPO_ROOT / PROTOCOL),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL),
        "fixture_archive_sha256": sha256_file(archive), "private_manifest_sha256": sha256_file(private_manifest),
        "gate_config_sha256": sha256_file(REPO_ROOT / GATE),
    }


if __name__ == "__main__":
    print(json.dumps(freeze_split(), indent=2, sort_keys=True))
