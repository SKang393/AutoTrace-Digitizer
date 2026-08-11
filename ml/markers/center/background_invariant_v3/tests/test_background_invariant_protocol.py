# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from ml.markers.center.background_invariant_v3 import candidate_runner, prepare_split
from ml.markers.center.background_invariant_v3.dataset import (
    load_sealed_public_archive,
    selection_manifest,
)
from ml.markers.center.background_invariant_v3.pipeline import normalize_proposal_patches
from ml.markers.center.background_invariant_v3.public_gate import GATE_CONFIG
from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT = REPO_ROOT / "ml/markers/center/background_invariant_v3"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_patch_normalization_removes_only_smooth_ink_background() -> None:
    patches = torch.zeros((2, 3, 33, 33), dtype=torch.float32)
    patches[0, 0] = 0.04
    patches[0, 0, 15:18, 15:18] = 0.90
    patches[0, 1, 4:8, 4:8] = 1.0
    patches[0, 2, 20:24, 20:24] = 0.75
    patches[1, 0] = 0.02
    patches[1, 0, 10:12, 10:12] = 0.70
    patches[1, 1] = torch.linspace(0.0, 1.0, 33)[None, :]
    patches[1, 2] = torch.linspace(0.0, 1.0, 33)[:, None]

    normalized = normalize_proposal_patches(patches)

    assert torch.count_nonzero(normalized[0, 0] == 0.0) == (33 * 33) - 9
    assert torch.isclose(normalized[0, 0, 16, 16], torch.tensor(0.86))
    assert torch.isclose(normalized[1, 0, 10, 10], torch.tensor(0.68))
    assert torch.equal(normalized[:, 1:], patches[:, 1:])
    assert torch.equal(normalize_proposal_patches(normalized), normalized)


def test_frozen_selection_and_truth_hidden_public_bytes_are_bound() -> None:
    selection_path = ROOT / "SELECTION_MANIFEST.json"
    seal_path = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    selection = load(selection_path)
    seal = load(seal_path)
    assert canonical_json_bytes(selection_manifest()) == selection_path.read_bytes()
    assert selection["scene_count"] == 16
    assert len({case["tensor_sha256"] for case in selection["cases"]}) == 16
    assert selection["synthetic_only"] is True
    assert selection["private_or_article_images"] is False
    assert selection["chandler_included"] is False
    archive_path = REPO_ROOT / seal["fixture_archive_path"]
    private_manifest_path = REPO_ROOT / seal["private_manifest_path"]
    assert sha256_file(archive_path) == seal["fixture_archive_sha256"]
    assert sha256_file(private_manifest_path) == seal["private_manifest_sha256"]
    assert seal["fixture_archive_sha256"] != "668d8274e3544945d1b6384bdd259cdee81942fd5c9cb36daa25c476574427b7"
    assert seal["prior_exposed_fixture_bytes_reused"] is False
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


def test_preregistration_binds_one_candidate_source_payload_and_gate() -> None:
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
    assert protocol["status"] == "candidate_1_preregistered"
    assert protocol["experiment_budget"] == 1
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert config["optimizer_steps"] == 0
    assert config["weights_changed"] is False
    assert sha256_file(REPO_ROOT / config["source_result_path"]) == config["source_result_sha256"]
    assert sha256_file(REPO_ROOT / config["source_training_report_path"]) == config["source_training_report_sha256"]
    assert sha256_file(REPO_ROOT / config["source_checkpoint_path"]) == config["source_checkpoint_sha256"]
    assert sha256_file(REPO_ROOT / config["source_onnx_path"]) == config["source_onnx_sha256"]
    assert config["selection_manifest_sha256"] == sha256_file(ROOT / "SELECTION_MANIFEST.json")
    assert config["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, candidate_runner.RUNNER_SOURCE_PATHS
    )
    assert gate["expected_candidate_hash_keys"] == ["onnx_sha256", "selection_report_sha256"]
    assert gate["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_no_candidate_or_public_output_exists_before_committed_execution() -> None:
    output = REPO_ROOT / "ml/markers/center/artifacts/background-invariant-v3/P1-run"
    assert not output.exists()
    source = (ROOT / "candidate_runner.py").read_text(encoding="utf-8")
    assert '"production_approval": False' in source
    assert '"release_eligible": False' in source
    assert '"optimizer_steps": 0' in source
    assert "build_selection_scenes()" in source
    assert "evaluate_candidate(" in source
    assert prepare_split.RUNNER_SOURCE_PATHS == candidate_runner.RUNNER_SOURCE_PATHS
