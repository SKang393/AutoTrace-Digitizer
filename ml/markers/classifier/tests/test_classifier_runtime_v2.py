# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, source_bundle_sha256
from ml.markers.classifier.confirmation_gate import build_confirmation_split
from ml.markers.classifier.dataset import (
    FILL_NAMES,
    SHAPE_NAMES,
    SPLIT_FAMILIES,
    SPLIT_TEMPLATES,
    build_fixed_dataset,
)
from ml.markers.classifier.model import CompactMarkerClassifier
from ml.markers.classifier.public_gate import build_public_gate_split
from ml.markers.classifier.runtime_gate_v2 import (
    ARTIFACT_F1_GATE,
    CONFIRMATION_GATE,
    FILL_MACRO_F1_GATE,
    MINORITY_CLASS_F1_GATE,
    PUBLIC_GATE,
    RUNTIME_GATE_CONFIG,
    SHAPE_MACRO_F1_GATE,
    build_gate_split,
    classifier_gate_results,
    gate_manifest,
)
from ml.markers.classifier.runtime_repair_v2 import (
    CANDIDATE_ID,
    CONFIG_PATH,
    REVISION,
    RUNNER_SOURCE_PATHS,
    SELECTION_MANIFEST_PATH,
    TASK,
    selection_manifest,
)
from ml.markers.classifier.runtime_v2 import PARITY_TOLERANCE, ProbabilityPackedRuntimeClassifier


REPO_ROOT = Path(__file__).resolve().parents[4]


def _tensor_hashes(samples) -> set[str]:
    return {hashlib.sha256(sample.tensor.numpy().tobytes(order="C")).hexdigest() for sample in samples}


def test_probability_runtime_contract_preserves_decisions_and_bounds_probabilities() -> None:
    model = CompactMarkerClassifier().eval()
    runtime = ProbabilityPackedRuntimeClassifier(model, 1.35, 0.70).eval()
    tensor = torch.stack([sample.tensor for sample in build_fixed_dataset("validation")[:8]])
    with torch.inference_mode():
        shape, fill, artifact, embedding = model(tensor)
        packed = runtime(tensor)
    assert packed.shape == (8, 25)
    assert torch.equal(packed[:, 0:9].argmax(dim=1), shape.argmax(dim=1))
    assert torch.equal(packed[:, 9:12].argmax(dim=1), fill.argmax(dim=1))
    assert torch.equal(packed[:, 12].ge(0.5), artifact[:, 0].ge(0.0))
    assert torch.allclose(packed[:, 0:9].sum(dim=1), torch.ones(8), atol=1e-6)
    assert torch.allclose(packed[:, 9:12].sum(dim=1), torch.ones(8), atol=1e-6)
    assert torch.equal(packed[:, 13:25], embedding)
    assert bool(torch.all((packed[:, 0:13] >= 0.0) & (packed[:, 0:13] <= 1.0)))


def test_runtime_v2_selection_and_gate_manifests_are_frozen_and_disjoint() -> None:
    expected_selection = canonical_json_bytes(selection_manifest())
    assert (REPO_ROOT / SELECTION_MANIFEST_PATH).read_bytes() == expected_selection
    selection_config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert selection_config["selection_dataset_manifest_sha256"] == sha256_bytes(expected_selection)

    public_manifest = canonical_json_bytes(gate_manifest(PUBLIC_GATE))
    confirmation_manifest = canonical_json_bytes(gate_manifest(CONFIRMATION_GATE))
    public_path = REPO_ROOT / "ml/markers/classifier/manifests/public-v3.json"
    confirmation_path = REPO_ROOT / "ml/markers/classifier/manifests/confirmation-v3.json"
    assert public_path.read_bytes() == public_manifest
    assert confirmation_path.read_bytes() == confirmation_manifest
    public_config = json.loads((REPO_ROOT / PUBLIC_GATE.split_config_path).read_text(encoding="utf-8"))
    confirmation_config = json.loads((REPO_ROOT / CONFIRMATION_GATE.split_config_path).read_text(encoding="utf-8"))
    assert public_config["expected_dataset_manifest_sha256"] == sha256_bytes(public_manifest)
    assert confirmation_config["expected_dataset_manifest_sha256"] == sha256_bytes(confirmation_manifest)

    selection_samples = tuple(
        sample for split in ("train", "validation", "test") for sample in build_fixed_dataset(split)
    )
    historical_public = build_public_gate_split()
    historical_confirmation = build_confirmation_split()
    public_samples = build_gate_split(PUBLIC_GATE)
    confirmation_samples = build_gate_split(CONFIRMATION_GATE)
    groups = [
        _tensor_hashes(selection_samples),
        _tensor_hashes(historical_public),
        _tensor_hashes(historical_confirmation),
        _tensor_hashes(public_samples),
        _tensor_hashes(confirmation_samples),
    ]
    assert all(left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1 :])
    historical_families = {family for values in SPLIT_FAMILIES.values() for family in values}
    historical_templates = {template for values in SPLIT_TEMPLATES.values() for template in values}
    historical_families |= {sample.family for sample in historical_public + historical_confirmation}
    historical_templates |= {sample.template for sample in historical_public + historical_confirmation}
    public_families = {sample.family for sample in public_samples}
    confirmation_families = {sample.family for sample in confirmation_samples}
    public_templates = {sample.template for sample in public_samples}
    confirmation_templates = {sample.template for sample in confirmation_samples}
    assert public_families.isdisjoint(historical_families | confirmation_families)
    assert confirmation_families.isdisjoint(historical_families | public_families)
    assert public_templates.isdisjoint(historical_templates | confirmation_templates)
    assert confirmation_templates.isdisjoint(historical_templates | public_templates)


def test_runtime_v2_source_bundles_and_budget_binding_match_preregistration() -> None:
    candidate_config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert candidate_config["task"] == TASK
    assert candidate_config["revision"] == REVISION
    assert candidate_config["candidate_id"] == CANDIDATE_ID
    assert candidate_config["optimizer_steps"] == 0
    assert candidate_config["weights_changed"] is False
    assert candidate_config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT,
        RUNNER_SOURCE_PATHS,
    )
    gate_config_hash = sha256_bytes(canonical_json_bytes(RUNTIME_GATE_CONFIG))
    for definition in (PUBLIC_GATE, CONFIRMATION_GATE):
        config = json.loads((REPO_ROOT / definition.split_config_path).read_text(encoding="utf-8"))
        assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
            REPO_ROOT,
            definition.evaluator_source_paths,
        )
        assert config["expected_gate_config_sha256"] == gate_config_hash
    ledger = json.loads(
        (REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert entry["status"] == "candidate_1_public_gate_passed_confirmation_authorized"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == [CANDIDATE_ID]
    assert entry["consumed_candidate_ids"] == [CANDIDATE_ID]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 1
    assert entry["public_gate_report_sha256"] == "32eed939875a3f6a3465fe8cf42a7f9f1ab9c33a4e5b225dfb7c396de2741757"
    assert entry["public_gate_probability_packed_onnx_maximum_absolute_error"] <= PARITY_TOLERANCE
    assert entry["confirmation_gate_authorized"] is True
    assert entry["confirmation_gate_evaluations"] == 0
    assert entry["p1_optimizer_steps"] == 0
    assert entry["p1_weights_changed"] is False
    assert entry["p1_probability_packed_onnx_maximum_absolute_error"] <= PARITY_TOLERANCE
    assert entry["p1_candidate_report_sha256"] == "3947e5a7f9f35e3684caa49f1fa1cbdc763d632a253ba07ded1720f5a9f6d7d8"
    assert entry["p1_probability_packed_onnx_sha256"] == "26f9304f1689053a0b94aa896a1e239f6ade1e5c1920736a3535c1b32f803b8a"
    assert entry["candidate_config_sha256"][CANDIDATE_ID] == hashlib.sha256(
        (REPO_ROOT / CONFIG_PATH).read_bytes()
    ).hexdigest()
    model_manifest = json.loads(
        (REPO_ROOT / "models/manifest/markers/graph-marker-classifier-0.1.0.json").read_text(encoding="utf-8")
    )
    benchmark = next(
        item
        for item in model_manifest["benchmarks"]
        if item["profile"] == "production-runtime-repair-v2-p1-selection-20260804"
    )
    assert benchmark["status"] == "pass"
    assert benchmark["release_eligible"] is False
    assert benchmark["production_approval"] is False
    assert benchmark["public_v3_evaluation_count"] == 0
    assert benchmark["confirmation_v3_evaluation_count"] == 0
    public_benchmark = next(
        item
        for item in model_manifest["benchmarks"]
        if item["profile"] == "production-runtime-repair-v2-p1-public-v3-20260804"
    )
    assert public_benchmark["status"] == "pass"
    assert public_benchmark["release_eligible"] is False
    assert public_benchmark["production_approval"] is False
    assert public_benchmark["evaluation_count"] == 1
    assert public_benchmark["confirmation_v3_evaluation_count"] == 0


def test_runtime_v2_gate_boundaries_remain_strict() -> None:
    minority = {name: MINORITY_CLASS_F1_GATE for name in ("star", "asterisk", "cross")}
    exact = classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=ARTIFACT_F1_GATE,
        minority_shape_f1=minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE,
    )
    assert all(exact.values())
    assert not classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=ARTIFACT_F1_GATE,
        minority_shape_f1=minority,
        parity_maximum_absolute_error=np.nextafter(PARITY_TOLERANCE, np.inf),
    )["probability_packed_onnx_parity"]
    low_minority = dict(minority)
    low_minority["star"] = np.nextafter(MINORITY_CLASS_F1_GATE, 0.0)
    assert not classifier_gate_results(
        shape_macro_f1=SHAPE_MACRO_F1_GATE,
        fill_macro_f1=FILL_MACRO_F1_GATE,
        artifact_f1=ARTIFACT_F1_GATE,
        minority_shape_f1=low_minority,
        parity_maximum_absolute_error=PARITY_TOLERANCE,
    )["minority_shape_preservation"]


def test_runtime_v2_manifests_cover_every_class_and_artifact() -> None:
    for definition in (PUBLIC_GATE, CONFIRMATION_GATE):
        samples = build_gate_split(definition)
        markers = [sample for sample in samples if sample.artifact < 0.5]
        artifacts = [sample for sample in samples if sample.artifact >= 0.5]
        assert len(samples) == 140
        assert {SHAPE_NAMES[sample.shape_index] for sample in markers} == set(SHAPE_NAMES)
        assert {FILL_NAMES[sample.fill_index] for sample in markers} == set(FILL_NAMES)
        assert {sample.artifact_kind for sample in artifacts} == {
            "text",
            "axis",
            "tick",
            "divider",
            "arrow",
            "bracket",
            "intersection",
            "legend",
        }
