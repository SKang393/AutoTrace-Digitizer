# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import ast
import json
from pathlib import Path

import torch

from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.train_family_v19 import protocol
from ml.markers.center.train_family_v19.train import RUNNER_SOURCE_PATHS
from ml.markers.center.train_family_v19.training_families import build_train_scenes
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


ROOT = Path(__file__).parents[5]
CONFIG_PATH = ROOT / "ml/markers/center/train_family_v19/training/p1.json"


def test_v19_binds_the_exact_v18_result_and_one_candidate_budget() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = ROOT / protocol.V18_RESULT_PATH
    assert sha256_file(result) == protocol.V18_RESULT_SHA256
    assert config["v18_result_sha256"] == protocol.V18_RESULT_SHA256
    assert config["sealed_candidate_budget"] == 1
    assert config["sealed_runs"] == 0
    assert config["public_gate_evaluations"] == 0


def test_families_cover_measured_ranges_without_touching_dev() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    families = config["train_family_specs"]
    assert set(families) == set(protocol.TRAIN_FAMILY_SPECS)
    assert min(value["source_size_range"][0] for value in families.values()) <= 360
    assert max(value["source_size_range"][2] for value in families.values()) >= 6352
    assert {mode for value in families.values() for mode in value["color_modes"]} == {"RGB", "RGBA"}
    assert all(value["jpeg_quality_range"][0] < 90 for value in families.values())
    scenes = build_train_scenes()
    assert len(scenes) == config["train_scene_count"] == 12
    assert len({scene.scene_id for scene in scenes}) == len(scenes)
    assert all(scene.split == "train" and "dev" not in scene.scene_id for scene in scenes)
    assert {scene.family for scene in scenes} == set(families)


def test_train_family_generation_is_deterministic_and_not_v13_dev() -> None:
    first, second = build_train_scenes(), build_train_scenes()
    assert [scene.scene_id for scene in first] == [scene.scene_id for scene in second]
    assert all(torch.equal(left.tensor, right.tensor) for left, right in zip(first, second, strict=True))
    assert not any(scene.scene_id.startswith("dev-") for scene in first)


def test_runtime_contract_is_unchanged_without_optimizer_step() -> None:
    model = ScaleClassifierNet(ModelConfig(seed=20260902)).eval()
    for count in (1, 8, 37):
        with torch.inference_mode():
            output = model(torch.zeros((count, 3, 33, 33)))
        assert tuple(output.shape) == (count, 4)
        assert float(output[:, 0].min()) >= 0.0
        assert float(output[:, 0].max()) <= 1.0
        assert float(output[:, 3].min()) >= 2.5
        assert float(output[:, 3].max()) <= 8.0


def test_runner_has_no_private_corpus_reference_or_sealed_split_read() -> None:
    source = (ROOT / "ml/markers/center/train_family_v19/train.py").read_text(encoding="utf-8")
    assert "data/manual data" not in source
    assert 'build_selection_scenes("sealed")' not in source
    ast.parse(source)


def test_runner_source_bundle_is_current() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    result_path = ROOT / "ml/markers/center/train_family_v19/P1_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["dev_metrics"]["precision"] == 1.0
    assert result["dev_metrics"]["recall"] == 0.7916666666666666
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    ledger_path = ROOT / "ml/markers/training-budgets/production-repair-v1.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
