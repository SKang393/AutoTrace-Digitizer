# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze V17 train, selection, and truth-hidden public evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import ceil
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from .dataset import build_split, proposal_summary, save_sealed_public_archive, split_fingerprint, training_examples
from .protocol import BASE_CHECKPOINT_SHA256, PUBLIC_REVISION, REVISION, TASK, protocol_configuration
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from .train_p1 import RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/structural_veto_proposal_role_v17")
PRIVATE_ROOT = ROOT / "artifacts/split-freeze"
PROTOCOL = ROOT / "PROTOCOL.json"
SELECTION = ROOT / "SELECTION_MANIFEST.json"
SEAL = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
GATE = ROOT / "gates/sealed-public-v1.json"
CONFIG = ROOT / "training/p1.json"
SOURCES = (
    ROOT / "dataset.py", ROOT / "prepare_split.py", ROOT / "protocol.py",
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
)


def freeze_split() -> dict[str, str]:
    private = REPO_ROOT / PRIVATE_ROOT
    generated = tuple(REPO_ROOT / path for path in (SELECTION, SEAL, GATE, CONFIG))
    archive = private / "sealed-public-fixtures.zip"
    private_manifest = private / "sealed-public-private-manifest.json"
    existing = [str(path) for path in (*generated, private) if path.exists()]
    if existing:
        raise RuntimeError("OCR V17 split freeze refuses overwrite: " + ", ".join(existing))
    protocol_path = REPO_ROOT / PROTOCOL
    if protocol_path.read_bytes() != canonical_json_bytes(protocol_configuration()):
        raise RuntimeError("OCR V17 committed preregistration changed before split freeze")
    base_path = REPO_ROOT / protocol_configuration()["base_checkpoint"]["path"]
    if sha256_file(base_path) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("OCR V17 base checkpoint changed before split freeze")
    private.mkdir(parents=True, exist_ok=False)
    for target in generated:
        target.parent.mkdir(parents=True, exist_ok=True)
    train, validation, public = build_split("train"), build_split("validation"), build_split("sealed_public")
    training = protocol_configuration()["training"]
    values, labels, role_labels, training_evidence = training_examples(
        train, negative_cap_per_scene=int(training["negative_cap_per_scene"]),
    )
    if any(training_evidence[key] is not False for key in (
        "validation_or_public_pixels_used", "predecessor_fixture_bytes_used",
        "v16_fixture_bytes_scene_truth_or_case_identity_used",
    )):
        raise RuntimeError("OCR V17 training examples violated the frozen scope")
    selection = {
        "schema": "graphreader.ocr-structural-veto-proposal-role-selection.v1",
        "task": TASK, "revision": REVISION,
        "split_generator_source_paths": [path.as_posix() for path in SOURCES],
        "split_generator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, SOURCES),
        "protocol_path": PROTOCOL.as_posix(), "protocol_sha256": sha256_file(protocol_path),
        "base_checkpoint_path": protocol_configuration()["base_checkpoint"]["path"],
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "train": {**proposal_summary(train), "split_fingerprint": split_fingerprint(train)},
        "validation": {**proposal_summary(validation), "split_fingerprint": split_fingerprint(validation)},
        "training_evidence": training_evidence, "training_tensor_shape": list(values.shape),
        "training_proposal_label_count": int(len(labels)),
        "training_role_label_count": int((role_labels >= 0).sum()),
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False,
        "v16_fixture_bytes_scene_truth_or_case_identity_used": False,
        "predecessor_fixture_bytes_reused": False,
        "validation_or_public_pixels_used_for_training": False,
        "sealed_public_truth_available_to_candidate": False,
        "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / SELECTION).write_bytes(canonical_json_bytes(selection))
    private_value = save_sealed_public_archive(public, archive)
    private_manifest.write_bytes(canonical_json_bytes(private_value))
    seal = {
        "schema": "graphreader.ocr-structural-veto-proposal-role-sealed-test-seal.v1",
        "task": TASK, "revision": REVISION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        **proposal_summary(public), "split_fingerprint": split_fingerprint(public), "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False,
        "v16_fixture_bytes_scene_truth_or_case_identity_used": False,
        "predecessor_fixture_bytes_reused": False, "truth_hidden_from_candidate_runner": True,
        "fixture_archive_path": PRIVATE_ROOT.joinpath("sealed-public-fixtures.zip").as_posix(),
        "fixture_archive_sha256": sha256_file(archive),
        "private_manifest_path": PRIVATE_ROOT.joinpath("sealed-public-private-manifest.json").as_posix(),
        "private_manifest_sha256": sha256_file(private_manifest),
        "selection_manifest_path": SELECTION.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION),
        "protocol_path": PROTOCOL.as_posix(), "protocol_sha256": sha256_file(protocol_path),
        "public_gate_evaluations": 0, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / SEAL).write_bytes(canonical_json_bytes(seal))
    gate = {
        "schema": "graphreader.ocr-structural-veto-proposal-role-gate-config.v1",
        "task": TASK, "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": ["onnx_sha256", "selection_report_sha256"],
        "sealed_public_test_seal_path": SEAL.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL),
        "expected_dataset_manifest_sha256": sha256_file(private_manifest),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / GATE).write_bytes(canonical_json_bytes(gate))
    expected_optimizer_steps = ceil(len(values) / int(training["batch_size"])) * int(training["epochs"])
    config = {
        "schema": "graphreader.ocr-structural-veto-proposal-role-training.v1",
        "task": TASK, "revision": REVISION, "candidate_id": "P1",
        "experiment_ordinal": 1, "experiment_budget": 3,
        "architecture": protocol_configuration()["architecture"],
        "isolated_change": protocol_configuration()["isolated_change"],
        "seed": training["seed"], "epochs": training["epochs"], "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"], "weight_decay": training["weight_decay"],
        "negative_cap_per_scene": training["negative_cap_per_scene"],
        "negative_sampling": training["negative_sampling"],
        "negative_class_weight": training["negative_class_weight"],
        "positive_margin": training["positive_margin"], "negative_margin": training["negative_margin"],
        "margin_loss_weight": training["margin_loss_weight"],
        "base_checkpoint_path": protocol_configuration()["base_checkpoint"]["path"],
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "base_output_scale": protocol_configuration()["base_checkpoint"]["base_output_scale"],
        "expected_optimizer_steps": expected_optimizer_steps,
        **training_evidence,
        "selection_thresholds": protocol_configuration()["selection_thresholds"],
        "minimum_consecutive_passing_thresholds": protocol_configuration()["selection_gates"]["minimum_consecutive_passing_thresholds"],
        "selection_manifest_path": SELECTION.as_posix(),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION),
        "sealed_public_test_seal_path": SEAL.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL),
        "expected_runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
        "onnx_parity_tolerance": protocol_configuration()["selection_gates"]["onnx_parity_maximum_absolute_error"],
        "v16_aggregate_metrics_only_used_for_design": True,
        "v16_validation_case_detail_or_pixels_used_for_design": False,
        "predecessor_fixture_bytes_reused": False,
        "v16_fixture_bytes_scene_truth_or_case_identity_used": False,
        "validation_or_public_pixels_used_for_training": False,
        "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "public_gate_evaluations": 0,
        "public_gate_archive_opened": False, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / CONFIG).write_bytes(canonical_json_bytes(config))
    return {
        "protocol_sha256": sha256_file(protocol_path),
        "selection_manifest_sha256": sha256_file(REPO_ROOT / SELECTION),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / SEAL),
        "fixture_archive_sha256": sha256_file(archive),
        "private_manifest_sha256": sha256_file(private_manifest),
        "gate_config_sha256": sha256_file(REPO_ROOT / GATE),
        "candidate_config_sha256": sha256_file(REPO_ROOT / CONFIG),
    }


if __name__ == "__main__":
    print(json.dumps(freeze_split(), indent=2, sort_keys=True))
