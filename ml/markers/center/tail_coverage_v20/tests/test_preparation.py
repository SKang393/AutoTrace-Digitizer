# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import ast
import json
import math
from pathlib import Path

import torch

from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.tail_coverage_v20 import protocol
from ml.markers.center.tail_coverage_v20.train import RUNNER_SOURCE_PATHS
from ml.markers.center.tail_coverage_v20.training_families import build_train_scenes, geometry_consensus_veto_guard
from ml.markers.center.train_family_v19.training_families import build_train_scenes as build_v19_train_scenes
from ml.markers.gate_seal import sha256_file, source_bundle_sha256


ROOT = Path(__file__).parents[5]
CONFIG_PATH = ROOT / "ml/markers/center/tail_coverage_v20/training/p1.json"


def test_v20_binds_exact_v19_result_and_diagnostic_with_one_candidate_budget() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = ROOT / protocol.V19_RESULT_PATH
    diagnostic = ROOT / protocol.V19_DIAGNOSTIC_PATH
    assert sha256_file(result) == protocol.V19_RESULT_SHA256
    assert sha256_file(diagnostic) == protocol.V19_DIAGNOSTIC_SHA256
    assert config["v19_result_sha256"] == protocol.V19_RESULT_SHA256
    assert config["v19_diagnostic_sha256"] == protocol.V19_DIAGNOSTIC_SHA256
    assert config["architecture"] == protocol.ARCHITECTURE
    assert config["geometry_veto_guard"] == protocol.GEOMETRY_VETO_GUARD
    assert config["sealed_candidate_budget"] == 1
    assert config["sealed_runs"] == 0
    assert config["public_gate_evaluations"] == 0


def test_train_distribution_retains_v19_families_and_adds_disjoint_tail() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    families = config["train_family_specs"]
    assert set(families) == set(protocol.TRAIN_FAMILY_SPECS)
    assert {"realrange_small_rgb_train", "realrange_median_rgba_train", "realrange_wide_rgb_train", "realrange_large_rgba_train"} <= set(families)
    assert {"tail_open_square_train", "tail_radius_11_12_train", "tail_intersection_heavy_train"} <= set(families)
    assert len(build_train_scenes()) == config["train_scene_count"] == 21
    assert min(item["source_size_range"][0] for item in families.values()) <= 360
    assert max(item["source_size_range"][2] for item in families.values()) >= 6352
    assert {mode for item in families.values() for mode in item["color_modes"]} == {"RGB", "RGBA"}
    retained = build_v19_train_scenes()
    current = build_train_scenes()[: len(retained)]
    assert [scene.scene_id for scene in current] == [scene.scene_id for scene in retained]
    assert all(torch.equal(left.tensor, right.tensor) for left, right in zip(current, retained, strict=True))
    assert [scene.radii for scene in current] == [scene.radii for scene in retained]


def test_tail_variants_are_explicit_and_geometry_guarded() -> None:
    scenes = build_train_scenes()
    by_family = {family: [scene for scene in scenes if scene.family == family] for family in protocol.TRAIN_FAMILY_SPECS}
    assert all(any(scene.family == "tail_open_square_train" for scene in scenes) for _ in (0,))
    assert all(set(round(radius) for radius in scene.radii) <= {3, 4, 5} for scene in by_family["tail_open_square_train"])
    assert all(set(round(radius) for radius in scene.radii) <= {11, 12} for scene in by_family["tail_radius_11_12_train"])
    intersection_scenes = by_family["tail_intersection_heavy_train"]
    assert all(sum(item.kind == "line_intersection" for item in scene.prohibited) >= 8 for scene in intersection_scenes)
    assert all(geometry_consensus_veto_guard(scene) for scene in scenes)
    assert all(
        math.hypot(item.x - center[0], item.y - center[1]) > 3.0
        for scene in intersection_scenes
        for item in scene.prohibited
        if item.kind == "line_intersection"
        for center in scene.centers
    )


def test_train_generation_is_deterministic_and_uses_only_fresh_train_seeds() -> None:
    first, second = build_train_scenes(), build_train_scenes()
    assert [scene.scene_id for scene in first] == [scene.scene_id for scene in second]
    assert all(torch.equal(left.tensor, right.tensor) for left, right in zip(first, second, strict=True))
    assert all(scene.split == "train" and "dev" not in scene.scene_id for scene in first)
    assert all(scene.seed >= protocol.TRAIN_SEED_BASE for scene in first)
    assert not any(scene.seed == 1_570_000 for scene in first)


def test_runtime_contract_and_v16_loss_shape_remain_unchanged() -> None:
    model = ScaleClassifierNet(ModelConfig(seed=20260902)).eval()
    for count in (1, 8, 37):
        with torch.inference_mode():
            output = model(torch.zeros((count, 3, 33, 33)))
        assert tuple(output.shape) == (count, 4)
        assert float(output[:, 0].min()) >= 0.0
        assert float(output[:, 0].max()) <= 1.0
        assert float(output[:, 3].min()) >= 2.5
        assert float(output[:, 3].max()) <= 8.0


def test_runner_has_no_private_or_sealed_read() -> None:
    source = (ROOT / "ml/markers/center/tail_coverage_v20/train.py").read_text(encoding="utf-8")
    assert "data/manual data" not in source
    assert 'build_selection_scenes("sealed")' not in source
    ast.parse(source)


def test_runner_source_bundle_is_current() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)


def test_tracked_outcome_is_failed_dev_without_sealed_consumption() -> None:
    result = json.loads((Path(__file__).parents[1] / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed_dev_retired_unconsumed"
    assert result["dev_metrics"]["precision"] == 1.0
    assert result["dev_metrics"]["recall"] == 0.8645833333333334
    assert result["candidate_consumed"] is False
    assert result["sealed_runs"] == 0
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("revision") == result["revision"])
    assert entry["status"] == "retired_failed_dev"
    assert entry["execution_authorized"] is False
    assert entry["consumed_candidate_ids"] == []
