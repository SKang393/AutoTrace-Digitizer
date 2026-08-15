# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze V18 validation and truth-hidden public fixture bytes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from .dataset import build_split, proposal_summary, save_split_archive, split_fingerprint
from .evaluate_candidate import RUNNER_SOURCE_PATHS
from .protocol import (
    CANDIDATE_ID,
    DETECTOR_PATH,
    DETECTOR_RESULT_PATH,
    DETECTOR_RESULT_SHA256,
    DETECTOR_SHA256,
    FEASIBILITY_PATH,
    FEASIBILITY_SHA256,
    LICENSE_PATH,
    LICENSE_SHA256,
    NOTICE_PATH,
    NOTICE_SHA256,
    PUBLIC_REVISION,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    TASK,
    protocol_configuration,
)
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/recognition_confirmed_proposal_role_v18")
PRIVATE_ROOT = ROOT / "artifacts/split-freeze"
PROTOCOL_PATH = ROOT / "PROTOCOL.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
GATE_PATH = ROOT / "gates/sealed-public-v1.json"
CONFIG_PATH = ROOT / "evaluation/p1.json"
SPLIT_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "prepare_split.py", ROOT / "protocol.py",
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
)


def freeze_split() -> dict[str, str]:
    private = REPO_ROOT / PRIVATE_ROOT
    generated = tuple(REPO_ROOT / path for path in (SELECTION_PATH, SEAL_PATH, GATE_PATH, CONFIG_PATH))
    existing = [str(path) for path in (*generated, private) if path.exists()]
    if existing:
        raise RuntimeError("OCR V18 split freeze refuses overwrite: " + ", ".join(existing))
    protocol_path = REPO_ROOT / PROTOCOL_PATH
    if protocol_path.read_bytes() != canonical_json_bytes(protocol_configuration()):
        raise RuntimeError("OCR V18 committed protocol is not canonical")
    exact_files = {
        DETECTOR_PATH: DETECTOR_SHA256,
        DETECTOR_RESULT_PATH: DETECTOR_RESULT_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        FEASIBILITY_PATH: FEASIBILITY_SHA256,
        LICENSE_PATH: LICENSE_SHA256,
        NOTICE_PATH: NOTICE_SHA256,
    }
    for relative, expected in exact_files.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"OCR V18 preregistered input changed: {relative}")
    private.mkdir(parents=True, exist_ok=False)
    for target in generated:
        target.parent.mkdir(parents=True, exist_ok=True)

    validation = build_split("validation")
    public = build_split("sealed_public")
    validation_archive = private / "validation-fixtures.zip"
    validation_manifest = private / "validation-private-manifest.json"
    public_archive = private / "sealed-public-fixtures.zip"
    public_manifest = private / "sealed-public-private-manifest.json"
    validation_private = save_split_archive(validation, validation_archive)
    public_private = save_split_archive(public, public_archive)
    validation_manifest.write_bytes(canonical_json_bytes(validation_private))
    public_manifest.write_bytes(canonical_json_bytes(public_private))

    validation_summary = proposal_summary(validation)
    public_summary = proposal_summary(public)
    selection = {
        "schema": "graphreader.ocr-recognition-confirmed-selection.v1",
        "task": TASK,
        "revision": REVISION,
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "split_generator_source_paths": [path.as_posix() for path in SPLIT_SOURCE_PATHS],
        "split_generator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS),
        "fixed_artifacts": {
            relative: expected for relative, expected in exact_files.items()
        },
        "validation": {
            **validation_summary,
            "split_fingerprint": split_fingerprint(validation),
            "fixture_archive_path": PRIVATE_ROOT.joinpath("validation-fixtures.zip").as_posix(),
            "fixture_archive_sha256": sha256_file(validation_archive),
            "private_manifest_path": PRIVATE_ROOT.joinpath("validation-private-manifest.json").as_posix(),
            "private_manifest_sha256": sha256_file(validation_manifest),
        },
        "sealed_public": {
            **public_summary,
            "split_fingerprint": split_fingerprint(public),
            "fixture_archive_path": PRIVATE_ROOT.joinpath("sealed-public-fixtures.zip").as_posix(),
            "fixture_archive_sha256": sha256_file(public_archive),
            "private_manifest_path": PRIVATE_ROOT.joinpath("sealed-public-private-manifest.json").as_posix(),
            "private_manifest_sha256": sha256_file(public_manifest),
        },
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_fixture_bytes_reused": False,
        "v17_fixture_bytes_scene_truth_or_case_identity_reused": False,
        "optimizer_steps": 0,
        "sealed_public_truth_available_to_candidate": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / SELECTION_PATH).write_bytes(canonical_json_bytes(selection))

    seal = {
        "schema": "graphreader.ocr-recognition-confirmed-sealed-test-seal.v1",
        "task": TASK,
        "revision": REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        **public_summary,
        "split_fingerprint": split_fingerprint(public),
        "fixture_archive_path": PRIVATE_ROOT.joinpath("sealed-public-fixtures.zip").as_posix(),
        "fixture_archive_sha256": sha256_file(public_archive),
        "private_manifest_path": PRIVATE_ROOT.joinpath("sealed-public-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(public_manifest),
        "selection_manifest_path": SELECTION_PATH.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION_PATH),
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_fixture_bytes_reused": False,
        "truth_hidden_from_candidate_runner": True,
        "public_gate_evaluations": 0,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / SEAL_PATH).write_bytes(canonical_json_bytes(seal))

    gate = {
        "schema": "graphreader.ocr-recognition-confirmed-gate-config.v1",
        "task": TASK,
        "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": [
            "detector_onnx_sha256", "recognizer_onnx_sha256", "selection_report_sha256",
        ],
        "sealed_public_test_seal_path": SEAL_PATH.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL_PATH),
        "expected_dataset_manifest_sha256": sha256_file(public_manifest),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / GATE_PATH).write_bytes(canonical_json_bytes(gate))

    config = {
        "schema": "graphreader.ocr-recognition-confirmed-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "experiment_ordinal": 1,
        "experiment_budget": 1,
        "architecture": protocol_configuration()["architecture"],
        "isolated_change": protocol_configuration()["isolated_change"],
        "optimizer_steps": 0,
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "selection_manifest_path": SELECTION_PATH.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION_PATH),
        "sealed_public_test_seal_path": SEAL_PATH.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL_PATH),
        "expected_runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
        "validation_fixture_archive_sha256": sha256_file(validation_archive),
        "validation_private_manifest_sha256": sha256_file(validation_manifest),
        "visible_feasibility_aggregate_only_used_for_design": True,
        "v17_validation_case_detail_or_pixels_used_for_design": False,
        "predecessor_fixture_bytes_reused": False,
        "v17_fixture_bytes_scene_truth_or_case_identity_reused": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "public_gate_evaluations": 0,
        "public_gate_archive_opened": False,
        "marker_creation_evaluated": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / CONFIG_PATH).write_bytes(canonical_json_bytes(config))
    return {
        "protocol_sha256": sha256_file(protocol_path),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION_PATH),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL_PATH),
        "validation_fixture_archive_sha256": sha256_file(validation_archive),
        "validation_private_manifest_sha256": sha256_file(validation_manifest),
        "sealed_public_fixture_archive_sha256": sha256_file(public_archive),
        "sealed_public_private_manifest_sha256": sha256_file(public_manifest),
        "gate_config_sha256": sha256_file(REPO_ROOT / GATE_PATH),
        "candidate_config_sha256": sha256_file(REPO_ROOT / CONFIG_PATH),
    }


if __name__ == "__main__":
    print(json.dumps(freeze_split(), indent=2, sort_keys=True))
