# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze V20 train, visible validation, and truth-hidden public fixture bytes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from .dataset import build_split, proposal_summary, save_split_archive, split_fingerprint
from .protocol import (
    DETECTOR_FLOOR, DETECTOR_PATH, DETECTOR_SHA256, EXPERIMENT_BUDGET, LICENSE_PATH,
    LICENSE_SHA256, MARGIN_LOSS_WEIGHT, NEGATIVE_LOGIT_MARGIN, NOTICE_PATH,
    NOTICE_SHA256, POSITIVE_LOGIT_MARGIN, PUBLIC_REVISION, RECOGNIZER_PATH,
    RECOGNIZER_SHA256, RECOGNIZER_YAML_PATH, RECOGNIZER_YAML_SHA256, REVISION,
    SEED, TASK, THRESHOLDS, TRAINING_NEGATIVE_CAP_PER_SCENE, TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256, protocol_configuration,
)
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from .train_p1 import RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/margin_calibrator_v20")
PRIVATE_ROOT = ROOT / "artifacts/split-freeze"
PROTOCOL_PATH = ROOT / "PROTOCOL.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
GATE_PATH = ROOT / "gates/sealed-public-v1.json"
CONFIG_PATH = ROOT / "training/p1.json"
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
    existing = [str(path) for path in generated if path.exists()]
    if private.exists() and any(private.iterdir()):
        existing.append(str(private))
    if existing:
        raise RuntimeError("OCR V20 split freeze refuses overwrite: " + ", ".join(existing))
    protocol_path = REPO_ROOT / PROTOCOL_PATH
    if protocol_path.read_bytes() != canonical_json_bytes(protocol_configuration()):
        raise RuntimeError("OCR V20 committed protocol is not canonical")
    exact_files = {
        DETECTOR_PATH: DETECTOR_SHA256, RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256, TRIGGER_RESULT_PATH: TRIGGER_RESULT_SHA256,
        LICENSE_PATH: LICENSE_SHA256, NOTICE_PATH: NOTICE_SHA256,
    }
    for relative, expected in exact_files.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"OCR V20 preregistered input changed: {relative}")
    private.mkdir(parents=True, exist_ok=True)
    for target in generated:
        target.parent.mkdir(parents=True, exist_ok=True)

    splits = {name: build_split(name) for name in ("train", "validation", "sealed_public")}
    registered: dict[str, dict[str, object]] = {}
    for split, scenes in splits.items():
        archive = private / f"{split}-fixtures.zip"
        manifest = private / f"{split}-private-manifest.json"
        manifest.write_bytes(canonical_json_bytes(save_split_archive(scenes, archive)))
        registered[split] = {
            **proposal_summary(scenes), "split_fingerprint": split_fingerprint(scenes),
            "fixture_archive_path": PRIVATE_ROOT.joinpath(f"{split}-fixtures.zip").as_posix(),
            "fixture_archive_sha256": sha256_file(archive),
            "private_manifest_path": PRIVATE_ROOT.joinpath(f"{split}-private-manifest.json").as_posix(),
            "private_manifest_sha256": sha256_file(manifest),
        }

    selection = {
        "schema": "graphreader.ocr-margin-calibrator-selection.v1",
        "task": TASK, "revision": REVISION,
        "protocol_path": PROTOCOL_PATH.as_posix(), "protocol_sha256": sha256_file(protocol_path),
        "split_generator_source_paths": [path.as_posix() for path in SPLIT_SOURCE_PATHS],
        "split_generator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, SPLIT_SOURCE_PATHS),
        "fixed_artifacts": exact_files, **registered,
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "predecessor_fixture_bytes_reused": False,
        "v19_fixture_bytes_scene_truth_or_case_identity_reused": False,
        "sealed_public_truth_available_to_candidate": False,
        "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / SELECTION_PATH).write_bytes(canonical_json_bytes(selection))

    public = registered["sealed_public"]
    seal = {
        "schema": "graphreader.ocr-margin-calibrator-sealed-test-seal.v1",
        "task": TASK, "revision": REVISION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        **public, "selection_manifest_path": SELECTION_PATH.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION_PATH),
        "protocol_path": PROTOCOL_PATH.as_posix(), "protocol_sha256": sha256_file(protocol_path),
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "predecessor_fixture_bytes_reused": False, "truth_hidden_from_candidate_runner": True,
        "public_gate_evaluations": 0, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / SEAL_PATH).write_bytes(canonical_json_bytes(seal))

    gate = {
        "schema": "graphreader.ocr-margin-calibrator-gate-config.v1",
        "task": TASK, "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": [
            "calibrator_onnx_sha256", "detector_onnx_sha256", "recognizer_onnx_sha256",
            "selection_report_sha256",
        ],
        "sealed_public_test_seal_path": SEAL_PATH.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL_PATH),
        "expected_dataset_manifest_sha256": public["private_manifest_sha256"],
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / GATE_PATH).write_bytes(canonical_json_bytes(gate))

    train = registered["train"]
    train_count = int(train["positive_proposal_count"]) + TRAINING_NEGATIVE_CAP_PER_SCENE * int(train["scene_count"])
    expected_steps = ((train_count + 255) // 256) * 24
    config = {
        "schema": "graphreader.ocr-margin-calibrator-candidate.v1",
        "task": TASK, "revision": REVISION, "candidate_id": "P1",
        "experiment_ordinal": 1, "experiment_budget": EXPERIMENT_BUDGET,
        "architecture": protocol_configuration()["architecture"],
        "isolated_change": protocol_configuration()["isolated_change"],
        "seed": SEED, "epochs": 24, "batch_size": 256, "learning_rate": 0.0008,
        "weight_decay": 0.0001, "negative_class_weight": 4.0,
        "positive_logit_margin": POSITIVE_LOGIT_MARGIN,
        "negative_logit_margin": NEGATIVE_LOGIT_MARGIN,
        "margin_loss_weight": MARGIN_LOSS_WEIGHT,
        "negative_cap_per_scene": TRAINING_NEGATIVE_CAP_PER_SCENE,
        "recognition_batch_size": 64, "selection_thresholds": list(THRESHOLDS),
        "expected_optimizer_steps": expected_steps,
        "scene_count": train["scene_count"], "proposal_count": train_count,
        "positive_proposal_count": train["positive_proposal_count"],
        "negative_proposal_count": TRAINING_NEGATIVE_CAP_PER_SCENE * int(train["scene_count"]),
        "detector_floor": DETECTOR_FLOOR, "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256, "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "selection_manifest_path": SELECTION_PATH.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION_PATH),
        "sealed_public_test_seal_path": SEAL_PATH.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL_PATH),
        "expected_runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
        "trigger_result_path": TRIGGER_RESULT_PATH,
        "trigger_result_sha256": TRIGGER_RESULT_SHA256,
        "train_fixture_archive_sha256": train["fixture_archive_sha256"],
        "validation_fixture_archive_sha256": registered["validation"]["fixture_archive_sha256"],
        "v19_aggregate_metrics_only_used_for_design": True,
        "v19_case_details_fixture_bytes_scene_truth_or_case_identity_used": False,
        "private_or_article_images": False, "chandler_included": False,
        "public_gate_evaluations": 0, "public_gate_archive_opened": False,
        "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / CONFIG_PATH).write_bytes(canonical_json_bytes(config))
    return {
        "protocol_sha256": sha256_file(protocol_path),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION_PATH),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL_PATH),
        "train_fixture_archive_sha256": str(train["fixture_archive_sha256"]),
        "validation_fixture_archive_sha256": str(registered["validation"]["fixture_archive_sha256"]),
        "sealed_public_fixture_archive_sha256": str(public["fixture_archive_sha256"]),
        "gate_config_sha256": sha256_file(REPO_ROOT / GATE_PATH),
        "candidate_config_sha256": sha256_file(REPO_ROOT / CONFIG_PATH),
    }


if __name__ == "__main__":
    print(json.dumps(freeze_split(), indent=2, sort_keys=True))
