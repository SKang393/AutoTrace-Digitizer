# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import CANONICAL_LEDGER_PATH, acquire_training_candidate
from ml.markers.center.dataset import build_fixed_dataset
from ml.markers.center.dataset_v2_p2 import build_p2_selection_dataset, p2_dataset_manifest
from ml.markers.center.production_train_v2_p2 import CONFIG_PATH, RUNNER_SOURCE_PATHS


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPECTED_MANIFEST_SHA256 = "afb792b635472bc76d950249dd366f6d9c6f71099a39733d0cd0eb9e6722b59a"
TRACKED_MANIFEST_PATH = Path(
    "ml/markers/center/artifacts/production-repair-v2/P2-preregistration/dataset-manifest.json"
)
TRACKED_MANIFEST_CHECKSUM_PATH = TRACKED_MANIFEST_PATH.with_suffix(".sha256")


def test_p2_fixed_split_adds_only_deterministic_tick_joint_selection_families() -> None:
    first = p2_dataset_manifest()
    second = p2_dataset_manifest()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert hashlib.sha256(canonical_json_bytes(first)).hexdigest() == EXPECTED_MANIFEST_SHA256
    tracked_bytes = (REPO_ROOT / TRACKED_MANIFEST_PATH).read_bytes()
    assert tracked_bytes == canonical_json_bytes(first)
    assert (REPO_ROOT / TRACKED_MANIFEST_CHECKSUM_PATH).read_text(encoding="ascii") == (
        f"{EXPECTED_MANIFEST_SHA256}  dataset-manifest.json\n"
    )
    assert first["public_split_included"] is False
    assert first["private_data"] is False
    assert len(first["cases"]) == 18
    assert sum(case["split"] == "train" for case in first["cases"]) == 12
    assert sum(case["split"] == "validation" for case in first["cases"]) == 6
    assert set(first["split_families"]["train"]).isdisjoint(first["split_families"]["validation"])

    for split in ("train", "validation"):
        base = build_fixed_dataset(split)
        p2 = build_p2_selection_dataset(split)
        for actual, expected in zip(p2[: len(base)], base, strict=True):
            assert actual.scene_id == expected.scene_id
            assert torch.equal(actual.tensor, expected.tensor)
            assert torch.equal(actual.center_target, expected.center_target)
            assert torch.equal(actual.radius_target, expected.radius_target)
            assert torch.equal(actual.artifact_target, expected.artifact_target)
        added = p2[len(base) :]
        assert len(added) == (4 if split == "train" else 3)
        for index, scene in enumerate(added):
            source = base[index]
            assert scene.centers == source.centers
            assert torch.equal(scene.center_target, source.center_target)
            assert torch.equal(scene.radius_target, source.radius_target)
            assert not torch.equal(scene.tensor, source.tensor)
            assert len(scene.hard_negatives) == len(source.hard_negatives) + 6
            assert all(kind in {"tick", "line_intersection"} for kind, _, _ in scene.hard_negatives[-6:])


def test_p2_is_hash_bound_consumed_and_cannot_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p1 = json.loads((REPO_ROOT / "ml/markers/center/training/production-repair-v2-p1.json").read_text(encoding="utf-8"))
    p2_path = REPO_ROOT / CONFIG_PATH
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    for field in ("seed", "epochs", "learning_rate", "weight_decay", "robustness_mode", "selection_order"):
        assert p2[field] == p1[field]
    for field in (
        "hard_negative_center_suppression_weight",
        "hard_negative_artifact_weight",
        "mask_consensus",
        "mask_consensus_threshold",
        "marker_mask_channels_preserved",
        "architecture_change",
        "postprocessing_change",
    ):
        assert p2["changes"][field] == p1["changes"][field]
    assert p2["selection_dataset_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert p2["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)

    ledger = json.loads((REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["revision"] == "marker-center-production-repair-v2")
    assert entry["status"] == "exhausted_failed_public_gate"
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["authorized_candidate_id"] is None
    assert entry["candidate_config_sha256"]["P2"] == sha256_file(p2_path)
    assert entry["p2_selection_dataset_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert entry["p2_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)

    assert entry["candidate_checkpoint_sha256"]["P2"] == (
        "e9bdf74d36d7d460a5c2222e5a777a113150f69b3a8d1232466806e0f67ff33f"
    )
    assert entry["candidate_onnx_sha256"]["P2"] == (
        "d0cbfdf691dc7c6c96bf59eaf2b67f969afaab8e25da502eba1e9c0b66e1f8ae"
    )
    seal_root = REPO_ROOT / "ml/markers/training-seals/marker-center/marker-center-production-repair-v2/P2"
    assert sha256_file(seal_root / "opened.json") == entry["p2_training_opened_seal_sha256"]
    assert sha256_file(seal_root / "result.json") == entry["p2_training_result_seal_sha256"]

    monkeypatch.setattr(
        "ml.markers.training_budget.require_committed_sources",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="not authorized"):
        acquire_training_candidate(
            REPO_ROOT,
            task="marker-center",
            revision="marker-center-production-repair-v2",
            candidate_id="P2",
            config_path=CONFIG_PATH,
            runner_source_paths=RUNNER_SOURCE_PATHS,
        )
    assert not (tmp_path / "ml/markers/training-seals").exists()


@pytest.mark.parametrize(
    ("status", "consumed", "error_match"),
    (
        ("candidate_2_consumed_failed_selection", [], "exact preregistered status"),
        ("candidate_2_preregistered", ["P2"], "unused single-candidate authorization"),
    ),
)
def test_candidate_status_or_consumption_mismatch_refuses_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    consumed: list[str],
    error_match: str,
) -> None:
    repo = tmp_path / "repo"
    config_path = Path("candidate.json")
    runner_path = Path("runner.py")
    ledger_path = repo / CANONICAL_LEDGER_PATH
    ledger_path.parent.mkdir(parents=True)
    (repo / runner_path).write_text("# fixed runner\n", encoding="utf-8")
    runner_sha256 = source_bundle_sha256(repo, (runner_path,))
    config = {
        "task": "marker-center",
        "revision": "marker-center-production-repair-v2",
        "candidate_id": "P2",
        "expected_runner_source_bundle_sha256": runner_sha256,
    }
    (repo / config_path).write_bytes(canonical_json_bytes(config))
    ledger = {
        "revisions": [
            {
                "task": "marker-center",
                "revision": "marker-center-production-repair-v2",
                "status": status,
                "execution_authorized": True,
                "authorized_candidate_id": "P2",
                "preregistered_candidate_ids": ["P2"],
                "consumed_candidate_ids": consumed,
                "candidate_config_paths": {"P2": config_path.as_posix()},
                "candidate_config_sha256": {"P2": sha256_file(repo / config_path)},
            }
        ]
    }
    ledger_path.write_bytes(canonical_json_bytes(ledger))
    monkeypatch.setattr("ml.markers.training_budget.require_committed_sources", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match=error_match):
        acquire_training_candidate(
            repo,
            task="marker-center",
            revision="marker-center-production-repair-v2",
            candidate_id="P2",
            config_path=config_path,
            runner_source_paths=(runner_path,),
        )
    assert not (repo / "ml/markers/training-seals").exists()
