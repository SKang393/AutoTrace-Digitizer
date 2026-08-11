# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from ml.markers.center.background_invariant_v3.pipeline import normalize_proposal_patches
from ml.markers.center.normalized_training_v4 import candidate_runner, prepare_split
from ml.markers.center.normalized_training_v4.dataset import (
    build_selection_scenes,
    load_sealed_public_archive,
    selection_manifest,
)
from ml.markers.center.normalized_training_v4.public_gate import GATE_CONFIG
from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/normalized_training_v4"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_training_uses_exact_median_normalized_patch_contract() -> None:
    patches = torch.zeros((2, 3, 33, 33), dtype=torch.float32)
    patches[0, 0] = 0.04
    patches[0, 0, 15:18, 15:18] = 0.90
    patches[0, 1, 4:8, 4:8] = 1.0
    patches[1, 0] = 0.02
    patches[1, 0, 10:12, 10:12] = 0.70
    patches[1, 2, 20:24, 20:24] = 0.75

    normalized = normalize_proposal_patches(patches)

    assert torch.count_nonzero(normalized[0, 0] == 0.0) == (33 * 33) - 9
    assert torch.isclose(normalized[0, 0, 16, 16], torch.tensor(0.86))
    assert torch.isclose(normalized[1, 0, 10, 10], torch.tensor(0.68))
    assert torch.equal(normalized[:, 1:], patches[:, 1:])
    source = (ROOT / "candidate_runner.py").read_text(encoding="utf-8")
    assert "_normalize_examples(" in source
    assert "normalize_proposal_patches(examples.patches)" in source


def test_visible_splits_are_fresh_synthetic_and_byte_unique() -> None:
    manifest_path = ROOT / "SELECTION_MANIFEST.json"
    manifest = load(manifest_path)
    assert canonical_json_bytes(selection_manifest()) == manifest_path.read_bytes()
    assert manifest["scene_count"] == 46
    assert manifest["train_scene_count"] == 30
    assert manifest["validation_scene_count"] == 16
    assert len({case["scene_id"] for case in manifest["cases"]}) == 46
    assert len({case["tensor_sha256"] for case in manifest["cases"]}) == 46
    assert manifest["synthetic_only"] is True
    assert manifest["private_or_article_images"] is False
    assert manifest["chandler_included"] is False
    assert len(build_selection_scenes("train")) == 30
    assert len(build_selection_scenes("validation")) == 16


def test_truth_hidden_public_split_is_bound_and_untracked() -> None:
    seal = load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    assert seal["scene_count"] == 20
    assert seal["truth_hidden_from_candidate_until_selection_pass"] is True
    assert seal["prior_exposed_fixture_bytes_reused"] is False
    assert seal["fixture_archive_sha256"] == "20e0bdf3bafceba2e72d0c595fe7e7b945d8232a436e0ea3ed90e5b4111378b7"
    assert sha256_file(archive_path) == seal["fixture_archive_sha256"]
    assert sha256_file(private_manifest_path) == seal["private_manifest_sha256"]
    scenes = load_sealed_public_archive(archive_path)
    assert len(scenes) == 20
    assert len({scene.scene_id for scene in scenes}) == 20
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            archive_path.relative_to(REPO_ROOT).as_posix(),
            private_manifest_path.relative_to(REPO_ROOT).as_posix(),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert tracked.stdout.strip() == ""


def test_consumed_p1_binds_code_budget_and_single_public_gate() -> None:
    protocol = load(ROOT / "PROTOCOL.json")
    config_path = ROOT / "training/p1.json"
    config = load(config_path)
    gate_path = ROOT / "gates/sealed-public-p1.json"
    gate = load(gate_path)
    ledger = load(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(
        item
        for item in ledger["revisions"]
        if item["task"] == candidate_runner.TASK and item["revision"] == candidate_runner.REVISION
    )
    assert protocol["status"] == "synthetic_gate_passed_production_blocked"
    assert protocol["experiment_budget"] == 3
    assert protocol["consumed_candidates"] == ["P1"]
    assert protocol["currently_preregistered_candidate"] is None
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert config["weights_changed"] is True
    assert config["selection_manifest_sha256"] == sha256_file(ROOT / "SELECTION_MANIFEST.json")
    assert config["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, candidate_runner.RUNNER_SOURCE_PATHS
    )
    assert prepare_split.RUNNER_SOURCE_PATHS == candidate_runner.RUNNER_SOURCE_PATHS
    assert gate["expected_candidate_hash_keys"] == ["onnx_sha256", "selection_report_sha256"]
    assert gate["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))
    assert entry["status"] == "synthetic_gate_passed_production_blocked"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["p1_result_sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized_on_selection_pass"] is False
    assert entry["public_gate_evaluations"] == 1
    assert entry["public_gate_archive_opened"] is True
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_consumed_candidate_and_public_gate_evidence_are_exactly_bound() -> None:
    output = REPO_ROOT / "ml/markers/center/artifacts/normalized-training-v4/P1-run"
    seal_root = REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-normalized-training-v4/P1"
    result = load(ROOT / "P1_RESULT.json")
    candidate_report = output / "candidate-report.json"
    public_report = output / "public-gate-report.json"
    public = load(public_report)
    gate_root = REPO_ROOT / "ml/markers/gate-seals/marker-center" / result["public_gate_canonical_seal_key"]
    assert sha256_file(candidate_report) == result["candidate_report_sha256"]
    assert sha256_file(public_report) == result["public_gate_report_sha256"]
    assert sha256_file(seal_root / "opened.json") == result["training_opened_seal_sha256"]
    assert sha256_file(seal_root / "result.json") == result["training_result_seal_sha256"]
    assert sha256_file(gate_root / "opened.json") == result["public_gate_opened_seal_sha256"]
    assert sha256_file(gate_root / "result.json") == result["public_gate_result_seal_sha256"]
    assert result["status"] == "synthetic_gate_passed_production_blocked"
    assert load(ROOT / "PROTOCOL.json")["result"]["sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert result["selection_exact_scene_count"] == result["selection_scene_count"] == 16
    assert result["selection_true_positives"] == 128
    assert result["selection_false_positives"] == 0
    assert result["selection_false_negatives"] == 0
    assert result["selection_duplicate_count"] == 0
    assert result["selection_prohibited_structure_hits"] == 0
    assert result["onnx_parity_passed"] is True
    assert result["public_exact_scene_count"] == result["public_scene_count"] == 20
    assert result["public_true_positives"] == 168
    assert result["public_false_positives"] == 0
    assert result["public_false_negatives"] == 0
    assert result["public_duplicate_count"] == 0
    assert result["public_prohibited_structure_hits"] == 0
    assert public["status"] == "pass"
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
    assert result["rerun_allowed"] is False
    assert '"production_approval": False' in (ROOT / "candidate_runner.py").read_text(encoding="utf-8")
    assert '"release_eligible": False' in (ROOT / "candidate_runner.py").read_text(encoding="utf-8")
