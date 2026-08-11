# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze OCR V7 selection metadata and a truth-hidden public archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256

from .dataset import build_split, proposal_summary, save_sealed_public_archive, split_fingerprint
from .protocol import PUBLIC_REVISION, REVISION, TASK, protocol_configuration
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIVATE_ROOT = Path("ml/ocr/component_context_detector_v7/artifacts/split-freeze")
DEFAULT_PROTOCOL = Path("ml/ocr/component_context_detector_v7/PROTOCOL.json")
DEFAULT_SELECTION = Path("ml/ocr/component_context_detector_v7/SELECTION_MANIFEST.json")
DEFAULT_SEAL = Path("ml/ocr/component_context_detector_v7/SEALED_PUBLIC_TEST_SEAL.json")
DEFAULT_GATE = Path("ml/ocr/component_context_detector_v7/gates/sealed-public-v1.json")
SPLIT_SOURCE_PATHS = (
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/prepare_split.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
)


def freeze_split(*, private_root: Path, protocol_path: Path, selection_path: Path, seal_path: Path, gate_path: Path) -> dict[str, str]:
    private = REPO_ROOT / private_root
    targets = tuple(REPO_ROOT / path for path in (protocol_path, selection_path, seal_path, gate_path))
    archive_path = private / "sealed-public-fixtures.zip"
    private_manifest_path = private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in targets if path.exists()]
    if private.exists() and any(private.iterdir()):
        existing.append(str(private))
    if existing:
        raise RuntimeError("OCR V7 split freeze refuses to overwrite evidence: " + ", ".join(existing))
    private.mkdir(parents=True, exist_ok=True)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    protocol = protocol_configuration()
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SPLIT_SOURCE_PATHS]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS)
    (REPO_ROOT / protocol_path).write_bytes(canonical_json_bytes(protocol))

    training = build_split("train")
    validation = build_split("validation")
    sealed = build_split("sealed_public")
    training_summary = proposal_summary(training)
    validation_summary = proposal_summary(validation)
    sealed_summary = proposal_summary(sealed)
    selection = {
        "schema": "graphreader.ocr-component-context-selection.v1",
        "task": TASK,
        "revision": REVISION,
        "train": {**training_summary, "split_fingerprint": split_fingerprint(training)},
        "validation": {**validation_summary, "split_fingerprint": split_fingerprint(validation)},
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_public_cases_used_for_selection": False,
        "prior_public_sample_or_pixel_inspection_used": False,
        "sealed_public_truth_available_to_training": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / selection_path).write_bytes(canonical_json_bytes(selection))
    private_manifest = save_sealed_public_archive(sealed, archive_path)
    private_manifest_path.write_bytes(canonical_json_bytes(private_manifest))
    seal = {
        "schema": "graphreader.ocr-component-context-sealed-test-seal.v1",
        "task": TASK,
        "revision": REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        **sealed_summary,
        "split_fingerprint": split_fingerprint(sealed),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_public_archive_reused": False,
        "predecessor_public_sample_or_pixel_inspection_used": False,
        "truth_hidden_from_training_runner": True,
        "fixture_archive_path": private_root.joinpath("sealed-public-fixtures.zip").as_posix(),
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
        "schema": "graphreader.ocr-component-context-gate-config.v1",
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
    return {
        "protocol_sha256": sha256_file(REPO_ROOT / protocol_path),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / selection_path),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / seal_path),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "gate_config_sha256": sha256_file(REPO_ROOT / gate_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    arguments = parser.parse_args()
    result = freeze_split(
        private_root=arguments.private_root,
        protocol_path=arguments.protocol,
        selection_path=arguments.selection,
        seal_path=arguments.seal,
        gate_path=arguments.gate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

