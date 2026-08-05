# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from ml.markers.gate_seal import canonical_json_bytes, source_bundle_sha256
from ml.markers.classifier.confirmation_gate import (
    build_confirmation_split,
    confirmation_manifest,
    evaluate_confirmation_gate,
)
from ml.markers.classifier.dataset import SPLIT_FAMILIES, SPLIT_TEMPLATES, build_fixed_dataset
from ml.markers.classifier.production_train import CANDIDATE_SEEDS, EXPERIMENTS, train_candidates
from ml.markers.classifier.public_gate import (
    FILL_MACRO_F1_GATE,
    MINORITY_CLASS_F1_GATE,
    PARITY_TOLERANCE,
    SHAPE_MACRO_F1_GATE,
    build_public_gate_split,
    classifier_gate_results,
    public_gate_manifest,
    evaluate_public_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _tensor_hash(sample) -> str:
    return hashlib.sha256(sample.tensor.numpy().tobytes(order="C")).hexdigest()


def _contains_approval(value: object) -> bool:
    if isinstance(value, dict):
        return value.get("production_approval") is True or any(_contains_approval(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_approval(item) for item in value)
    return False


def test_frozen_manifest_hashes_and_confirmation_families_are_disjoint() -> None:
    public_payload = public_gate_manifest()
    confirmation_samples = build_confirmation_split()
    confirmation_payload = confirmation_manifest(confirmation_samples)
    assert hashlib.sha256(canonical_json_bytes(public_payload)).hexdigest() == "cdd0db99a0da47e5d6d56ecf73f37157331d2a41e3817e09728559f7f646db7d"
    assert hashlib.sha256(canonical_json_bytes(confirmation_payload)).hexdigest() == "d90cb4d8777a71cdc8cd103184624961dcc6929faf1619fd0ee864d739a8ff3a"
    selection_samples = tuple(
        sample
        for split in ("train", "validation", "test")
        for sample in build_fixed_dataset(split)
    )
    public_samples = build_public_gate_split()
    selection_hashes = {_tensor_hash(sample) for sample in selection_samples}
    public_hashes = {_tensor_hash(sample) for sample in public_samples}
    confirmation_hashes = {_tensor_hash(sample) for sample in confirmation_samples}
    assert selection_hashes.isdisjoint(public_hashes)
    assert selection_hashes.isdisjoint(confirmation_hashes)
    assert public_hashes.isdisjoint(confirmation_hashes)
    selection_families = {family for families in SPLIT_FAMILIES.values() for family in families}
    selection_templates = {template for templates in SPLIT_TEMPLATES.values() for template in templates}
    public_families = {sample.family for sample in public_samples}
    public_templates = {sample.template for sample in public_samples}
    confirmation_families = {sample.family for sample in confirmation_samples}
    confirmation_templates = {sample.template for sample in confirmation_samples}
    assert confirmation_families.isdisjoint(selection_families | public_families)
    assert confirmation_templates.isdisjoint(selection_templates | public_templates)
    assert "revision" not in inspect.signature(evaluate_public_gate).parameters
    assert "manifest_payload" not in inspect.signature(evaluate_public_gate).parameters
    assert "revision" not in inspect.signature(evaluate_confirmation_gate).parameters


def test_frozen_evaluator_source_bundles_match_configs() -> None:
    public_paths = (
        Path("ml/markers/classifier/public_gate.py"),
        Path("ml/markers/gate_seal.py"),
        Path("ml/markers/classifier/dataset.py"),
        Path("ml/markers/classifier/metrics.py"),
        Path("ml/markers/classifier/export.py"),
        Path("ml/markers/classifier/model.py"),
    )
    confirmation_paths = public_paths[:1] + (Path("ml/markers/classifier/confirmation_gate.py"),) + public_paths[1:]
    public_config = json.loads((REPO_ROOT / "ml/markers/classifier/gates/public-v1.json").read_text(encoding="utf-8"))
    confirmation_config = json.loads((REPO_ROOT / "ml/markers/classifier/gates/confirmation-v2.json").read_text(encoding="utf-8"))
    assert public_config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, public_paths)
    assert confirmation_config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, confirmation_paths)


def test_classifier_gate_exact_boundaries_pass_and_just_below_fail() -> None:
    minority = {"star": MINORITY_CLASS_F1_GATE, "asterisk": MINORITY_CLASS_F1_GATE, "cross": MINORITY_CLASS_F1_GATE}
    exact = classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=1.0,
        minority_shape_f1=minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE,
    )
    assert all(exact.values())
    assert not classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE - 1e-9,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=1.0,
        minority_shape_f1=minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE,
    )["shape_macro_f1"]
    assert not classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE - 1e-9,
        artifact_f1=1.0,
        minority_shape_f1=minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE,
    )["fill_macro_f1"]
    assert not classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=0.999,
        minority_shape_f1=minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE,
    )["artifact_f1"]
    low_minority = dict(minority)
    low_minority["star"] -= 1e-9
    assert not classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=1.0,
        minority_shape_f1=low_minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE,
    )["minority_shape_preservation"]
    assert not classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=1.0,
        minority_shape_f1=minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE + 1e-12,
    )["packed_onnx_parity"]


def test_exhausted_historical_revision_remains_fail_closed_after_runtime_approval() -> None:
    assert len(EXPERIMENTS) == 3
    assert len({item["id"] for item in EXPERIMENTS}) == 3
    assert len(CANDIDATE_SEEDS) == len(EXPERIMENTS)
    assert len(set(CANDIDATE_SEEDS)) == len(CANDIDATE_SEEDS)
    manifest = json.loads((REPO_ROOT / "models/manifest/markers/graph-marker-classifier-0.1.0.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["benchmarks"] if item["profile"] == "production-public-and-confirmation-gates-20260804")
    assert entry["experiment_count"] == len(EXPERIMENTS)
    assert entry["status"] == "fail"
    assert entry["release_eligible"] is False
    assert entry["generalization_evidence_valid"] is False
    assert entry["disjoint_confirmation_v2_status"] == "evaluated exactly once; fail"
    assert entry["disjoint_confirmation_v2_evidence_sha256"] == "2bb119c5ece6167177e225486c01441e84363c60c5afc4d3e6872aaae99d46b4"
    assert entry["disjoint_confirmation_v2_manifest_sha256"] == "d90cb4d8777a71cdc8cd103184624961dcc6929faf1619fd0ee864d739a8ff3a"
    assert entry["disjoint_confirmation_v2_seal_sha256"] == "793eb5fcaf226fceee283476729d5fae6974fb2cc7e9ed7520ed3c18f328e7f8"
    assert entry["disjoint_confirmation_v2_shape_macro_f1"] == 1.0
    assert entry["disjoint_confirmation_v2_fill_macro_f1"] >= FILL_MACRO_F1_GATE
    assert entry["disjoint_confirmation_v2_artifact_f1"] == 1.0
    assert entry["disjoint_confirmation_v2_minority_shape_f1"] == 1.0
    assert entry["disjoint_confirmation_v2_packed_onnx_maximum_absolute_error"] > PARITY_TOLERANCE
    assert entry["historical_seed_provenance_status"] == "incomplete"
    assert entry["historical_actual_batch_order_seed"] == 20260803
    assert entry["current_source_reproduces_historical_pipeline"] is False
    assert entry["failed_public_gate_evidence_sha256"] == "8ce39a252ac9ce37105625f1833a53485a4ea8738ff06de6267f7ce1d44feb74"
    assert entry["failed_confirmation_evidence_sha256"] == "59d0f37a95ae089cd8cf7815c7f5f4b526bb0dc6efc787b1eb9074ccaab54bd9"
    assert not _contains_approval(entry)
    approvals = [item for item in manifest["benchmarks"] if item.get("production_approval") is True]
    assert [item["profile"] for item in approvals] == [
        "production-runtime-repair-v2-p1-integrated-candidate-20260805"
    ]


def test_exhausted_classifier_revision_refuses_before_output(tmp_path: Path) -> None:
    output = tmp_path / "no-fourth-classifier-candidate"
    with pytest.raises(RuntimeError, match="committed before use|committed revision|budget is exhausted"):
        train_candidates(output)
    assert not output.exists()
