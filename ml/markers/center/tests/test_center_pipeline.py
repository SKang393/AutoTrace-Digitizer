# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ml.markers.center.dataset import (
    ARTIFACT_KINDS,
    DEGRADATION_BY_FAMILY,
    SPLIT_FAMILIES,
    build_fixed_dataset,
    seal_dataset_manifest,
)
from ml.markers.center.export import export_onnx
from ml.markers.center.metrics import center_metrics
from ml.markers.center.model import CompactCenterNet, DepthwiseSeparableBlock
from ml.markers.center.postprocess import Detection, detect_heads
from ml.markers.center.train import THRESHOLD_SWEEPS, train


def test_input_and_degradation_families_are_disjoint_but_target_geometry_repeats(tmp_path: Path) -> None:
    family_sets = [set(families) for families in SPLIT_FAMILIES.values()]
    assert all(not (left & right) for index, left in enumerate(family_sets) for right in family_sets[index + 1 :])
    degradation_sets = [
        {DEGRADATION_BY_FAMILY[family] for family in families}
        for families in SPLIT_FAMILIES.values()
    ]
    assert all(not (left & right) for index, left in enumerate(degradation_sets) for right in degradation_sets[index + 1 :])
    first_path, first_hash = seal_dataset_manifest(tmp_path / "first")
    second_path, second_hash = seal_dataset_manifest(tmp_path / "second")
    assert first_hash == second_hash
    assert first_path.read_bytes() == second_path.read_bytes()
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first_hash
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert all(set(case["hard_negative_kinds"]) == set(ARTIFACT_KINDS) for case in payload["cases"])
    assert build_fixed_dataset("train")[0].centers == build_fixed_dataset("validation")[0].centers


def test_trainable_depthwise_fpn_emits_single_three_channel_tensor_and_gradients() -> None:
    model = CompactCenterNet()
    assert sum(isinstance(module, DepthwiseSeparableBlock) for module in model.modules()) >= 7
    value = torch.rand(2, 3, 64, 80, requires_grad=True)
    output = model(value)
    assert output.shape == (2, 3, 64, 80)
    assert torch.all((output[:, 0] >= 0) & (output[:, 0] <= 1))
    assert torch.all(output[:, 1] > 1)
    assert torch.all((output[:, 2] >= 0) & (output[:, 2] <= 1))
    output.mean().backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in model.parameters())


def test_maximum_cardinality_matching_avoids_greedy_under_count() -> None:
    predictions = (
        Detection(1.0, 0.0, 3.0, 0.9, 0.0),
        Detection(0.0, 0.0, 3.0, 0.8, 0.0),
    )
    metrics = center_metrics(predictions, ((0.0, 0.0), (2.0, 0.0)), 1.1)
    assert metrics.true_positives == 2
    assert metrics.f1 == 1.0


def test_runtime_identical_local_max_and_radius_nms() -> None:
    heads = np.zeros((1, 3, 32, 32), dtype=np.float32)
    heads[:, 1] = 3.0
    heads[0, 0, 10, 10] = 0.9
    heads[0, 0, 10, 11] = 0.9
    heads[0, 0, 20, 20] = 0.8
    heads[0, 2, 20, 20] = 0.4
    empty_mask = np.zeros((32, 32), dtype=np.float32)
    detections = detect_heads(
        heads,
        text_mask=empty_mask,
        artifact_mask=empty_mask,
        center_threshold=0.36,
        artifact_threshold=0.35,
    )
    assert len(detections) == 1
    assert detections[0].x == 10.0 and detections[0].y == 10.0
    assert detections[0].radius == 3.0
    assert abs(detections[0].confidence - 0.9) < 1e-6


def test_raw_mask_point_four_suppresses_otherwise_identical_marker() -> None:
    heads = np.zeros((1, 3, 24, 24), dtype=np.float32)
    heads[:, 1] = 3.0
    heads[0, 0, 12, 12] = 0.9
    empty_mask = np.zeros((24, 24), dtype=np.float32)
    assert len(detect_heads(heads, text_mask=empty_mask, artifact_mask=empty_mask)) == 1
    for masked_channel in ("text", "artifact"):
        text_mask = empty_mask.copy()
        artifact_mask = empty_mask.copy()
        target = text_mask if masked_channel == "text" else artifact_mask
        target[12, 12] = 0.4
        assert not detect_heads(
            heads,
            text_mask=text_mask,
            artifact_mask=artifact_mask,
            artifact_threshold=0.35,
        )


def test_training_exports_exact_onnx_and_cpu_parity_without_heldout_evaluation(tmp_path: Path) -> None:
    checkpoint, training = train(tmp_path / "training")
    assert training["experiment_count"] == len(THRESHOLD_SWEEPS) == 3
    assert training["heldout_test_evaluations"] == 0
    assert training["dataset_manifest_sha256"]
    output = tmp_path / "marker-center.onnx"
    report = export_onnx(checkpoint, output, tmp_path / "onnx-parity.json")
    assert output.exists() and output.stat().st_size > 0
    assert report["status"] == "pass"
    assert report["provider"] == "cpu"
    assert report["parity_max_abs_difference"] <= report["parity_tolerance"]
    assert report["tensor_contract"]["input_channels"] == (
        "ink_probability",
        "text_mask",
        "artifact_mask",
    )
    assert report["tensor_contract"]["output_channels"] == (
        "center_probability",
        "radius_pixels",
        "artifact_probability",
    )
