# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from ml.markers.center.radial_feature_v1 import dataset as radial_dataset
from ml.markers.center.radial_feature_v1 import pipeline_p3
from ml.markers.center.runtime_consistency_v2 import dataset, pipeline, pipeline_p2
from ml.markers.center.runtime_consistency_v2.candidate_runner import (
    REPO_ROOT,
    RUNNER_SOURCE_PATHS,
)
from ml.markers.center.runtime_consistency_v2.candidate_runner_p2 import (
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
)
from ml.markers.center.runtime_consistency_v2.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    GATE_CONFIG,
)
from ml.markers.center.runtime_consistency_v2.public_gate_p2 import (
    EVALUATOR_SOURCE_PATHS as P2_EVALUATOR_SOURCE_PATHS,
    GATE_CONFIG as P2_GATE_CONFIG,
)
from ml.markers.gate_seal import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    source_bundle_sha256,
)


REVISION_ROOT = REPO_ROOT / "ml/markers/center/runtime_consistency_v2"
REVISION = "marker-center-runtime-consistency-v2"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_trigger_binds_the_selection_public_postprocess_mismatch() -> None:
    selection_source = (
        REPO_ROOT / "ml/markers/center/radial_feature_v1/train_p3.py"
    ).read_text(encoding="utf-8")
    public_source = (
        REPO_ROOT / "ml/markers/center/radial_feature_v1/sealed_gate.py"
    ).read_text(encoding="utf-8")
    assert (
        "from ml.markers.center.radial_feature_v1.pipeline_p3 import evaluate_scenes"
        in selection_source
    )
    assert (
        "from ml.markers.center.line_aware_v1.pipeline import evaluate_scenes"
        in public_source
    )
    protocol = _json(REVISION_ROOT / "PROTOCOL.json")
    assert protocol["trigger_evidence"]["selection_runner_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/center/radial_feature_v1/train_p3.py"
    )
    assert protocol["trigger_evidence"]["public_evaluator_sha256"] == sha256_file(
        REPO_ROOT / "ml/markers/center/radial_feature_v1/sealed_gate.py"
    )


def test_runtime_pipeline_is_the_exact_selected_p3_postprocessor() -> None:
    assert pipeline.POSTPROCESS_REVISION == "radial-local-consensus-refinement-v1"
    assert pipeline.evaluate_scenes is pipeline_p3.evaluate_scenes
    assert pipeline.infer_scene is pipeline_p3.infer_scene
    assert pipeline.postprocess_predictions is pipeline_p3.postprocess_predictions


def test_new_families_and_degradations_are_disjoint_from_radial_public_evidence() -> None:
    current_families = {
        item
        for values in dataset.SELECTION_FAMILIES.values()
        for item in values
    } | set(dataset.SEALED_PUBLIC_FAMILIES)
    prior_families = {
        item
        for values in radial_dataset.SELECTION_FAMILIES.values()
        for item in values
    } | set(radial_dataset.SEALED_PUBLIC_FAMILIES)
    current_degradations = {
        item for values in dataset.DEGRADATIONS.values() for item in values
    }
    prior_degradations = {
        item for values in radial_dataset.DEGRADATIONS.values() for item in values
    }
    assert current_families.isdisjoint(prior_families)
    assert current_degradations.isdisjoint(prior_degradations)


def test_selection_manifest_reproduces_without_private_or_chandler_data() -> None:
    manifest_path = REVISION_ROOT / "SELECTION_MANIFEST.json"
    manifest = _json(manifest_path)
    assert sha256_bytes(canonical_json_bytes(dataset.selection_manifest())) == sha256_file(
        manifest_path
    )
    assert manifest["synthetic_only"] is True
    assert manifest["private_or_article_images"] is False
    assert manifest["chandler_included"] is False
    assert len(manifest["cases"]) == 42
    validation = [case for case in manifest["cases"] if case["split"] == "validation"]
    assert len(validation) == 12
    assert sum(case["center_count"] for case in validation) == 96
    required = {
        "text",
        "axis",
        "tick",
        "divider",
        "legend",
        "bracket",
        "arrow_shaft",
        "arrowhead",
    }
    assert all(required.issubset(case["prohibited_kinds"]) for case in validation)


def test_frozen_public_archive_is_hash_bound_and_untracked() -> None:
    seal = _json(REVISION_ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    if not archive_path.is_file() or not private_manifest_path.is_file():
        pytest.skip("Ignored truth-hidden public evidence is not present in this checkout")
    assert sha256_file(archive_path) == seal["fixture_archive_sha256"]
    assert sha256_file(private_manifest_path) == seal["private_manifest_sha256"]
    assert seal["scene_count"] == 20
    assert seal["truth_hidden_from_candidate_until_selection_pass"] is True
    tracked = subprocess.run(
        ["git", "ls-files", "--", archive_path.relative_to(REPO_ROOT).as_posix()],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert tracked.stdout.strip() == ""


def test_candidate_config_binds_exact_zero_training_payload_and_sources() -> None:
    config_path = REVISION_ROOT / "training/p1.json"
    config = _json(config_path)
    assert config["optimizer_steps"] == 0
    assert config["weights_changed"] is False
    assert config["selected_threshold"] == 0.3
    assert config["postprocess_revision"] == pipeline.POSTPROCESS_REVISION
    assert (
        config["source_training_report_sha256"]
        == "67b5ea3b28973f0bd24ae0f755713af1c70b6fe6a9b2437268be5975b9f14af3"
    )
    assert (
        config["source_checkpoint_sha256"]
        == "6b670a6f29454d7f63527f57210aa918540a817fca156a71b96872ff09aa2787"
    )
    assert (
        config["source_onnx_sha256"]
        == "924c555e2f27955c644143125d7abd3b05859ea9928ab9d1e741e0544fa19e8b"
    )
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS
    )


def test_public_gate_binds_runtime_consistent_sources_and_single_execution() -> None:
    config = _json(REVISION_ROOT / "gates/sealed-public-v2.json")
    seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
    assert config["evaluation_limit"] == 1
    assert config["expected_candidate_hash_keys"] == ["onnx_sha256"]
    assert config["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert config["expected_gate_config_sha256"] == sha256_bytes(
        canonical_json_bytes(GATE_CONFIG)
    )
    assert GATE_CONFIG["postprocess_revision"] == pipeline.POSTPROCESS_REVISION
    assert GATE_CONFIG["required_exact_scene_fraction"] == 1.0
    assert GATE_CONFIG["required_false_positives"] == 0
    assert GATE_CONFIG["required_false_negatives"] == 0


def test_p2_diagnosis_and_config_bind_one_selection_only_calibration() -> None:
    diagnosis_path = REVISION_ROOT / "P2_DIAGNOSIS.json"
    diagnosis = _json(diagnosis_path)
    config_path = REVISION_ROOT / "training/p2.json"
    config = _json(config_path)
    assert diagnosis["scope"] == "visible validation selection only"
    assert diagnosis["sweep_count"] == 1
    assert diagnosis["public_archive_opened"] is False
    assert diagnosis["chandler_included"] is False
    assert diagnosis["private_or_article_images"] is False
    assert diagnosis["observed_failure"]["exact_scene_count"] == 8
    assert diagnosis["observed_failure"]["scene_count"] == 12
    assert len(diagnosis["bounded_grid"]) == 9
    selected = next(
        item
        for item in diagnosis["bounded_grid"]
        if item["threshold"] == 0.25
        and item["minimum_center_separation"] == 6.5
    )
    assert selected == {
        "threshold": 0.25,
        "minimum_center_separation": 6.5,
        "exact_scenes": 12,
        "true_positives": 96,
        "false_positives": 0,
        "false_negatives": 0,
        "duplicates": 0,
    }
    assert config["diagnosis_sha256"] == sha256_file(diagnosis_path)
    assert config["optimizer_steps"] == 0
    assert config["weights_changed"] is False
    assert config["selected_threshold"] == 0.25
    assert config["minimum_center_separation"] == 6.5
    assert config["postprocess_revision"] == pipeline_p2.POSTPROCESS_REVISION
    assert pipeline_p2.MINIMUM_CENTER_SEPARATION == 6.5
    assert config["source_checkpoint_sha256"] == (
        "6b670a6f29454d7f63527f57210aa918540a817fca156a71b96872ff09aa2787"
    )
    assert config["source_onnx_sha256"] == (
        "924c555e2f27955c644143125d7abd3b05859ea9928ab9d1e741e0544fa19e8b"
    )
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_RUNNER_SOURCE_PATHS
    )


def test_p2_public_gate_binds_same_unopened_archive_and_calibrated_sources() -> None:
    config_path = REVISION_ROOT / "gates/sealed-public-p2.json"
    config = _json(config_path)
    seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
    assert config["evaluation_limit"] == 1
    assert config["expected_candidate_hash_keys"] == ["onnx_sha256"]
    assert config["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, P2_EVALUATOR_SOURCE_PATHS
    )
    assert config["expected_gate_config_sha256"] == sha256_bytes(
        canonical_json_bytes(P2_GATE_CONFIG)
    )
    assert P2_GATE_CONFIG["threshold"] == 0.25
    assert P2_GATE_CONFIG["minimum_center_separation"] == 6.5
    assert P2_GATE_CONFIG["postprocess_revision"] == pipeline_p2.POSTPROCESS_REVISION
    protocol = _json(REVISION_ROOT / "PROTOCOL.json")
    assert protocol["public_gate"]["archive_opened"] is False
    assert protocol["public_gate"]["evaluations"] == 0


def test_canonical_budget_consumes_p1_and_authorizes_only_unopened_p2() -> None:
    ledger = _json(
        REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
    )
    entry = next(
        item
        for item in ledger["revisions"]
        if item["task"] == "marker-center" and item["revision"] == REVISION
    )
    assert entry["status"] == "candidate_2_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P2"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P3"]
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P2"
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(
        REVISION_ROOT / "training/p2.json"
    )
    assert entry["protocol_sha256"] == sha256_file(REVISION_ROOT / "PROTOCOL.json")
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] == "P2"
    assert entry["public_gate_authorized_on_selection_pass"] is True
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    result_path = REVISION_ROOT / "P1_RESULT.json"
    result = _json(result_path)
    assert entry["p1_result_sha256"] == sha256_file(result_path)
    protocol = _json(REVISION_ROOT / "PROTOCOL.json")
    assert protocol["status"] == "candidate_2_preregistered"
    assert protocol["consumed_candidates"] == ["P1"]
    assert protocol["execution_authorized"] is True
    assert protocol["authorized_candidate"] == "P2"
    assert protocol["candidate_result"]["result_sha256"] == sha256_file(result_path)
    assert result["status"] == "failed_selection"
    assert result["selection_exact_scene_count"] == 8
    assert result["selection_scene_count"] == 12
    assert result["selection_false_positives"] == 3
    assert result["selection_false_negatives"] == 1
    assert result["selection_duplicate_count"] == 2
    assert result["selection_prohibited_structure_hits"] == 0
    assert result["onnx_parity_passed"] is True
    assert result["sealed_public_archive_opened"] is False
    assert result["public_gate_evaluations"] == 0
    assert result["opened_seal_sha256"] == sha256_file(
        REPO_ROOT / result["opened_seal_path"]
    )
    assert result["result_seal_sha256"] == sha256_file(
        REPO_ROOT / result["result_seal_path"]
    )
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
