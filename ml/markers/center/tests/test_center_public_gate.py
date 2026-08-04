# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import pytest
import torch

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    require_evaluator_identity,
    sha256_file,
    source_bundle_sha256,
)
from ml.markers.training_budget import require_training_budget
from ml.markers.center.confirmation_gate import (
    build_confirmation_split,
    confirmation_manifest,
    evaluate_confirmation_gate,
)
from ml.markers.center.dataset import ARTIFACT_KINDS, build_fixed_dataset
from ml.markers.center.production_train import CANDIDATE_SEEDS, EXPERIMENTS, _batch, train_candidates
from ml.markers.center.production_train_v2 import CONFIG_PATH, RUNNER_SOURCE_PATHS, train_candidate
from ml.markers.center.public_gate import (
    PUBLIC_GATE_CONFIG,
    build_public_gate_split,
    center_gate_results,
    evaluate_public_gate,
    public_gate_manifest,
)
from ml.markers.center.tests._gate_seal_test_support import (
    _create_committed_gate_repo,
    _create_committed_training_budget_repo,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _tensor_hash(value) -> str:
    array = value.numpy() if hasattr(value, "numpy") else value
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _contains_approval(value: object) -> bool:
    if isinstance(value, dict):
        return value.get("production_approval") is True or any(_contains_approval(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_approval(item) for item in value)
    return False


def test_frozen_manifest_hashes_and_tensor_sets_are_selection_disjoint() -> None:
    public_payload = public_gate_manifest()
    confirmation_scenes = build_confirmation_split()
    confirmation_payload = confirmation_manifest(confirmation_scenes)
    assert hashlib.sha256(canonical_json_bytes(public_payload)).hexdigest() == "048815a821049cdba0ba69a980b8479ac851b2efa9a4a54b307678deaea0e83d"
    assert hashlib.sha256(canonical_json_bytes(confirmation_payload)).hexdigest() == "d8af15a9b1bceedd2310f898b661eb8f3c2d430d31b74e7dba13b20a74d6cbe5"
    selection = {
        _tensor_hash(scene.tensor)
        for split in ("train", "validation", "test")
        for scene in build_fixed_dataset(split)
    }
    public = {_tensor_hash(scene.tensor) for scene in build_public_gate_split()}
    confirmation = {_tensor_hash(scene.tensor) for scene in confirmation_scenes}
    assert selection.isdisjoint(public)
    assert selection.isdisjoint(confirmation)
    assert public.isdisjoint(confirmation)
    assert all({item[0] for item in scene.hard_negatives} == set(ARTIFACT_KINDS) for scene in confirmation_scenes)


def test_center_gate_passes_only_when_every_scientific_boundary_passes() -> None:
    exact_rows = [{"exact_count": True}, {"exact_count": True}]
    zero_hits = {kind: 0 for kind in ARTIFACT_KINDS}
    assert all(center_gate_results(exact_rows, 0, zero_hits).values())
    assert not center_gate_results([{"exact_count": True}, {"exact_count": False}], 0, zero_hits)["exact_count_every_fixture"]
    assert not center_gate_results(exact_rows, 1, zero_hits)["zero_duplicates"]
    one_hit = dict(zero_hits)
    one_hit["text"] = 1
    assert not center_gate_results(exact_rows, 0, one_hit)["zero_prohibited_structure_hits"]


def test_repository_scoped_seal_rejects_global_replay_and_binds_sources(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manifest_sha = "a" * 64
    source, split = _create_committed_gate_repo(repo, gate_config=PUBLIC_GATE_CONFIG, manifest_sha256=manifest_sha)
    candidate = {"onnx_sha256": "b" * 64}
    assert "ledger_root" not in inspect.signature(acquire_gate_seal).parameters
    assert "require_committed" not in inspect.signature(acquire_gate_seal).parameters
    assert "ledger_root" not in inspect.signature(evaluate_public_gate).parameters
    assert "require_committed" not in inspect.signature(evaluate_public_gate).parameters
    assert "revision" not in inspect.signature(evaluate_public_gate).parameters
    assert "manifest_payload" not in inspect.signature(evaluate_public_gate).parameters
    assert "revision" not in inspect.signature(evaluate_confirmation_gate).parameters
    first = acquire_gate_seal(
        repo_root=repo,
        task="marker-center",
        revision="test-v1",
        candidate_hashes=candidate,
        dataset_manifest_sha256=manifest_sha,
        split_config_path=split.relative_to(repo),
        evaluator_source_paths=(source.relative_to(repo),),
        gate_config=PUBLIC_GATE_CONFIG,
    )
    alternate_ledger = repo / "alternate-ledger"
    assert first.binding["evaluator_source_bundle_sha256"]
    assert first.binding["split_config_sha256"] == sha256_file(split)
    assert first.binding["gate_config_sha256"]
    assert first.binding["ledger_mode"] == "canonical_repository"
    assert first.binding["ledger_root"] == "ml/markers/gate-seals"
    assert first.binding["committed_source_enforcement"] is True
    assert all("output" not in key for key in first.binding)
    assert not alternate_ledger.exists()
    complete_gate_seal(first, status="fail", report_sha256="c" * 64)
    aliases = (
        {"task": "marker-center", "revision": "caller-changed-v2", "candidate_hashes": candidate, "message": "Gate revision"},
        {"task": "marker-centre-alias", "revision": "test-v1", "candidate_hashes": candidate, "message": "Gate task"},
        {"task": "marker-center", "revision": "test-v1", "candidate_hashes": {"model_sha256": "b" * 64}, "message": "Candidate hash key schema"},
        {"task": "marker-center", "revision": "test-v1", "candidate_hashes": {}, "message": "Candidate hash key schema"},
        {"task": "marker-center", "revision": "test-v1", "candidate_hashes": {"onnx_sha256": "b" * 64, "extra_sha256": "e" * 64}, "message": "Candidate hash key schema"},
    )
    for alias in aliases:
        with pytest.raises(RuntimeError, match=str(alias["message"])):
            acquire_gate_seal(
                repo_root=repo,
                task=str(alias["task"]),
                revision=str(alias["revision"]),
                candidate_hashes=alias["candidate_hashes"],
                dataset_manifest_sha256=manifest_sha,
                split_config_path=split.relative_to(repo),
                evaluator_source_paths=(source.relative_to(repo),),
                gate_config=PUBLIC_GATE_CONFIG,
            )
    with pytest.raises(RuntimeError, match="already opened"):
        acquire_gate_seal(
            repo_root=repo,
            task="marker-center",
            revision="test-v1",
            candidate_hashes=candidate,
            dataset_manifest_sha256=manifest_sha,
            split_config_path=split.relative_to(repo),
            evaluator_source_paths=(source.relative_to(repo),),
            gate_config=PUBLIC_GATE_CONFIG,
        )
    source.write_text("gate = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="committed revision"):
        acquire_gate_seal(
            repo_root=repo,
            task="marker-center",
            revision="test-v2",
            candidate_hashes={"onnx_sha256": "d" * 64},
            dataset_manifest_sha256=manifest_sha,
            split_config_path=split.relative_to(repo),
            evaluator_source_paths=(source.relative_to(repo),),
            gate_config=PUBLIC_GATE_CONFIG,
        )
    subprocess.run(("git", "add", "ml/markers/center/evaluator.py"), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-q", "-m", "Mutate evaluator"), cwd=repo, check=True)
    with pytest.raises(RuntimeError, match="does not match frozen configuration"):
        acquire_gate_seal(
            repo_root=repo,
            task="marker-center",
            revision="test-v2",
            candidate_hashes={"onnx_sha256": "d" * 64},
            dataset_manifest_sha256=manifest_sha,
            split_config_path=split.relative_to(repo),
            evaluator_source_paths=(source.relative_to(repo),),
            gate_config=PUBLIC_GATE_CONFIG,
        )
    assert not alternate_ledger.exists()


def test_candidate_hash_key_order_is_frozen(tmp_path: Path) -> None:
    repo = tmp_path / "ordered-schema-repo"
    manifest_sha = "a" * 64
    source, split = _create_committed_gate_repo(
        repo,
        gate_config=PUBLIC_GATE_CONFIG,
        manifest_sha256=manifest_sha,
        candidate_hash_keys=("checkpoint_sha256", "packed_onnx_sha256"),
    )
    reversed_keys = {"packed_onnx_sha256": "d" * 64, "checkpoint_sha256": "c" * 64}
    with pytest.raises(RuntimeError, match="Candidate hash key schema"):
        acquire_gate_seal(
            repo_root=repo,
            task="marker-center",
            revision="test-v1",
            candidate_hashes=reversed_keys,
            dataset_manifest_sha256=manifest_sha,
            split_config_path=split.relative_to(repo),
            evaluator_source_paths=(source.relative_to(repo),),
            gate_config=PUBLIC_GATE_CONFIG,
        )


def test_evaluator_boundaries_reject_manifest_seal_and_report_identity_mismatch() -> None:
    expected = {"task": "marker-center", "revision": "frozen-v1"}
    require_evaluator_identity(
        expected_task="marker-center",
        expected_revision="frozen-v1",
        manifest=expected,
        split_config=expected,
        seal_binding=expected,
        report=expected,
    )
    for boundary in ("manifest", "split_config", "seal_binding", "report"):
        values = {
            "manifest": expected,
            "split_config": expected,
            "seal_binding": expected,
            "report": expected,
        }
        values[boundary] = {"task": "marker-alias", "revision": "frozen-v1"}
        with pytest.raises(RuntimeError, match="task does not match frozen gate identity"):
            require_evaluator_identity(
                expected_task="marker-center",
                expected_revision="frozen-v1",
                **values,
            )
        values[boundary] = {"task": "marker-center", "revision": "changed-v2"}
        with pytest.raises(RuntimeError, match="revision does not match frozen gate identity"):
            require_evaluator_identity(
                expected_task="marker-center",
                expected_revision="frozen-v1",
                **values,
            )


def test_frozen_evaluator_source_bundles_match_configs() -> None:
    public_paths = (
        Path("ml/markers/center/public_gate.py"),
        Path("ml/markers/gate_seal.py"),
        Path("ml/markers/center/dataset.py"),
        Path("ml/markers/center/metrics.py"),
        Path("ml/markers/center/postprocess.py"),
    )
    confirmation_paths = public_paths[:1] + (Path("ml/markers/center/confirmation_gate.py"),) + public_paths[1:]
    public_config = json.loads((REPO_ROOT / "ml/markers/center/gates/public-v1.json").read_text(encoding="utf-8"))
    confirmation_config = json.loads((REPO_ROOT / "ml/markers/center/gates/confirmation-v1.json").read_text(encoding="utf-8"))
    assert public_config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, public_paths)
    assert confirmation_config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, confirmation_paths)


def test_candidate_budget_and_manifest_remain_fail_closed() -> None:
    assert len(EXPERIMENTS) == 3
    assert len({item["id"] for item in EXPERIMENTS}) == 3
    assert len(CANDIDATE_SEEDS) == len(EXPERIMENTS)
    assert len(set(CANDIDATE_SEEDS)) == len(CANDIDATE_SEEDS)
    scenes = build_fixed_dataset("train")
    inputs, _, _, _ = _batch(scenes, epoch=1, mode="mixed")
    original = torch.stack([scene.tensor for scene in scenes])
    assert torch.equal(inputs[:, 1:], original[:, 1:])
    manifest = json.loads((REPO_ROOT / "models/manifest/markers/graph-marker-center-0.1.0.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["benchmarks"] if item["profile"] == "production-public-and-confirmation-gates-20260804")
    assert entry["experiment_count"] == len(EXPERIMENTS)
    assert entry["status"] == "fail"
    assert entry["release_eligible"] is False
    assert entry["selection_status"] == "invalid_protocol"
    assert entry["evidence_validity"] == "invalid"
    assert entry["failed_public_gate_evidence_sha256"] == "a44ad0827f32c67b13ba492a528cec4b622b578bf15f85f64da89b259f9d8733"
    assert entry["failed_confirmation_evidence_sha256"] == "faef8737ea574eef42f7b24ef4e71f2aa1a99caff21a5d489627173e1c38b217"
    assert not _contains_approval(manifest)


def test_exhausted_center_revision_refuses_before_output(tmp_path: Path) -> None:
    output = tmp_path / "no-fourth-center-candidate"
    with pytest.raises(RuntimeError, match="committed before use|committed revision|budget is exhausted"):
        train_candidates(output)
    assert not output.exists()


def test_canonical_training_budget_rejects_exhaustion_and_local_mutation(tmp_path: Path) -> None:
    repo = tmp_path / "budget-repo"
    ledger = _create_committed_training_budget_repo(repo)
    with pytest.raises(RuntimeError, match="budget is exhausted"):
        require_training_budget(
            repo,
            task="marker-center",
            revision="marker-center-production-repair-v1",
        )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["revisions"][0]["status"] = "available"
    ledger.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="committed revision"):
        require_training_budget(
            repo,
            task="marker-center",
            revision="marker-center-production-repair-v1",
        )


def test_untracked_training_budget_is_never_trusted(tmp_path: Path) -> None:
    repo = tmp_path / "untracked-budget-repo"
    ledger = repo / "ml/markers/training-budgets/production-repair-v1.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "revisions": [
                    {
                        "task": "marker-center",
                        "revision": "marker-center-production-repair-v1",
                        "status": "available",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    with pytest.raises(RuntimeError, match="committed before use"):
        require_training_budget(
            repo,
            task="marker-center",
            revision="marker-center-production-repair-v1",
        )


def test_v2_candidate_one_is_hash_bound_consumed_and_cannot_rerun(tmp_path: Path) -> None:
    config_path = REPO_ROOT / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger = json.loads((REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["revision"] == "marker-center-production-repair-v2")
    assert config["candidate_id"] == "P1"
    assert config["experiment_ordinal"] == 1
    assert config["experiment_budget"] == 3
    assert config["public_gate_use_for_selection"] is False
    assert config["changes"]["marker_mask_channels_preserved"] is True
    assert config["changes"]["hard_negative_center_suppression_weight"] == 1.0
    assert config["changes"]["hard_negative_artifact_weight"] == 0.75
    historical_runner_source_sha256 = "0dc41fbb2b44e67267266b5d5d86c3433f14adfc909027b177bd87de415c6c7c"
    assert config["expected_runner_source_bundle_sha256"] == historical_runner_source_sha256
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(config_path)
    assert entry["status"] == "candidate_3_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P3"]
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P3"
    assert entry["candidate_checkpoint_sha256"]["P1"] == "2292f516ed7263f741549fb6b127a62d1d1cf4368153d23953bca3fa9812deab"
    assert entry["candidate_onnx_sha256"]["P1"] == "f8f543dee4e80e55f5e7ab316e6ddfd3884219c191b5378a967ed186f4c5b6a6"
    output = tmp_path / "v2-p1-must-not-rerun"
    with pytest.raises(RuntimeError, match="committed revision|not authorized|already opened"):
        train_candidate(output)
    assert not output.exists()
    seal_root = REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-production-repair-v2/P1"
    opened = json.loads((seal_root / "opened.json").read_text(encoding="utf-8"))
    assert opened["binding"]["runner_source_bundle_sha256"] == historical_runner_source_sha256
    assert sha256_file(seal_root / "opened.json") == entry["p1_training_opened_seal_sha256"]
    assert sha256_file(seal_root / "result.json") == entry["p1_training_result_seal_sha256"]
