# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from ml.markers.center import production_train_v2_p3 as p3_runner
from ml.markers.center.dataset import build_fixed_dataset
from ml.markers.center.dataset_v2_p3 import (
    MINIMUM_TRUE_CENTER_DISTANCE,
    build_p3_selection_dataset,
    p3_dataset_manifest,
)
from ml.markers.center.production_train_v2_p3 import CONFIG_PATH, RUNNER_SOURCE_PATHS
from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPECTED_MANIFEST_SHA256 = "5d2192aa2cefcb646abcddea4b87be05898a242d2d8f43a6b23100ec36cbfe02"
EXPECTED_RUNNER_SHA256 = "e24c1bb4384e85e6270f7f75c6c3ac6041ddc3da88d88f296868cdcfc11a792d"
TRACKED_MANIFEST_PATH = Path(
    "ml/markers/center/artifacts/production-repair-v2/P3-preregistration/dataset-manifest.json"
)


def test_p3_changes_only_hard_negative_placement_and_prevents_truth_overlap() -> None:
    manifest = p3_dataset_manifest()
    assert hashlib.sha256(canonical_json_bytes(manifest)).hexdigest() == EXPECTED_MANIFEST_SHA256
    assert (REPO_ROOT / TRACKED_MANIFEST_PATH).read_bytes() == canonical_json_bytes(manifest)
    assert manifest["public_split_included"] is False
    assert manifest["private_data"] is False
    assert manifest["minimum_true_center_distance"] == MINIMUM_TRUE_CENTER_DISTANCE == 32.0

    for split in ("train", "validation"):
        base = build_fixed_dataset(split)
        p3 = build_p3_selection_dataset(split)
        for actual, expected in zip(p3[: len(base)], base, strict=True):
            assert actual.scene_id == expected.scene_id
            assert torch.equal(actual.tensor, expected.tensor)
            assert torch.equal(actual.center_target, expected.center_target)
            assert torch.equal(actual.radius_target, expected.radius_target)
            assert torch.equal(actual.artifact_target, expected.artifact_target)
        added = p3[len(base) :]
        assert len(added) == (4 if split == "train" else 3)
        for index, scene in enumerate(added):
            source = base[index]
            assert scene.centers == source.centers
            assert torch.equal(scene.center_target, source.center_target)
            assert torch.equal(scene.radius_target, source.radius_target)
            assert len(scene.hard_negatives) == len(source.hard_negatives) + 6
            added_negatives = scene.hard_negatives[-6:]
            assert all(kind in {"tick", "line_intersection"} for kind, _, _ in added_negatives)
            assert min(
                math.dist((x, y), center)
                for _, x, y in added_negatives
                for center in scene.centers
            ) >= MINIMUM_TRUE_CENTER_DISTANCE


def test_p3_is_final_hash_bound_single_factor_and_refuses_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p2 = json.loads(
        (REPO_ROOT / "ml/markers/center/training/production-repair-v2-p2.json").read_text(
            encoding="utf-8"
        )
    )
    p3_path = REPO_ROOT / CONFIG_PATH
    p3 = json.loads(p3_path.read_text(encoding="utf-8"))
    for field in (
        "seed",
        "epochs",
        "learning_rate",
        "weight_decay",
        "robustness_mode",
        "selection_order",
    ):
        assert p3[field] == p2[field]
    for field in (
        "hard_negative_center_suppression_weight",
        "hard_negative_artifact_weight",
        "mask_consensus",
        "mask_consensus_threshold",
        "marker_mask_channels_preserved",
        "architecture_change",
        "postprocessing_change",
    ):
        assert p3["changes"][field] == p2["changes"][field]
    assert p3["changes"]["minimum_true_center_distance"] == 32.0
    assert p3["selection_dataset_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert p3["expected_runner_source_bundle_sha256"] == EXPECTED_RUNNER_SHA256
    assert source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS) == EXPECTED_RUNNER_SHA256

    ledger = json.loads(
        (REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item for item in ledger["revisions"] if item["revision"] == "marker-center-production-repair-v2"
    )
    assert entry["status"] == "candidate_3_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P3"]
    assert entry["consumed_candidate_ids"] == ["P1", "P2"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["authorized_candidate_id"] == "P3"
    assert entry["candidate_config_sha256"]["P3"] == sha256_file(p3_path)
    assert entry["p3_selection_dataset_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert entry["p3_runner_source_bundle_sha256"] == EXPECTED_RUNNER_SHA256

    output = tmp_path / "p3-must-not-run-without-authorization"

    def refuse_authorization(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("controlled authorization refusal")

    monkeypatch.setattr(p3_runner, "acquire_training_candidate", refuse_authorization)
    with pytest.raises(RuntimeError, match="controlled authorization refusal"):
        p3_runner.train_candidate(output)
    assert not output.exists()
    assert not (
        REPO_ROOT
        / "ml/markers/training-seals/marker-center/marker-center-production-repair-v2/P3"
    ).exists()
