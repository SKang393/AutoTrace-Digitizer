# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import json
from pathlib import Path

from ml.markers.center.mask_preserving_v24.train_p1 import RUNNER_SOURCE_PATHS
from ml.markers.gate_seal import source_bundle_sha256

ROOT = Path(__file__).resolve().parents[5]

def test_training_contract_is_fixed_and_candidate_not_run():
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    assert config["seed"] == 20260903
    assert config["confidence_threshold"] == 0.25
    assert config["selection_thresholds"] == [0.40, 0.55, 0.70]
    assert config["optimizer_steps_expected"] == 10080
    assert config["optimizer_steps_maximum"] == 10080
    assert config["training_example_count_expected"] == 35838
    assert config["sealed_runs"] == 0 and config["private_data"] is False
    assert config["real_dev_reads"] == 0 and config["real_sealed_reads"] == 0

def test_runner_source_bundle_is_relative_and_present():
    assert all(not path.is_absolute() for path in RUNNER_SOURCE_PATHS)
    assert all((ROOT / path).is_file() for path in RUNNER_SOURCE_PATHS)
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]

def test_train_examples_preserve_mask_crossing_positive():
    from ml.markers.center.mask_preserving_v24.train_p1 import _examples
    from ml.markers.center.real_range_generator_v1.generator import build_split
    import torch
    scene = next(
        item for item in build_split("train")
        if any(
            float(item.tensor[1, int(y)-2:int(y)+3, int(x)-2:int(x)+3].max()) >= .35
            for x, y in item.centers
        )
    )
    patches, labels, _, radii, _ = _examples((scene,), 10, torch.Generator().manual_seed(20260904))
    assert patches.shape[1:] == (3, 33, 33)
    assert bool((labels > 0.5).any())
    assert bool((patches[labels > 0.5, 1].sum(dim=(1,2)) > 0).any())
    assert float(radii.max()) <= max(scene.diameters) / 2.0
