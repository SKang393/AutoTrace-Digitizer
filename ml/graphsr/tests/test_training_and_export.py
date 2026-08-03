# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
import math
from pathlib import Path

import onnx
import pytest

from ml.graphsr.export import export_onnx
from ml.graphsr.losses import MAX_ADVERSARIAL_WEIGHT
from ml.graphsr.model import GraphSRConfig, ensure_artifact_outside_repository
from ml.graphsr.train import train


def _stable_training_fields(report: dict[str, object]) -> dict[str, object]:
    volatile = {"checkpoint", "checkpoint_sha256", "elapsed_ms"}
    return {key: value for key, value in report.items() if key not in volatile}


def test_tiny_training_is_deterministic_finite_and_bounded(
    artifact_root: Path,
    no_network: None,
) -> None:
    config = GraphSRConfig(channels=4, blocks=1, seed=31415)
    first_checkpoint, first = train(
        artifact_root / "first",
        seed=31415,
        epochs=1,
        max_steps=2,
        batch_size=2,
        config=config,
    )
    second_checkpoint, second = train(
        artifact_root / "second",
        seed=31415,
        epochs=1,
        max_steps=2,
        batch_size=2,
        config=config,
    )

    assert first_checkpoint.is_file() and second_checkpoint.is_file()
    assert _stable_training_fields(first) == _stable_training_fields(second)
    assert first["steps_completed"] == 2
    assert first["finite_loss_history"] is True
    assert first["heldout_test_evaluations"] == 0
    assert first["dataset_scope"] == "deterministic built-in synthetic smoke crops"
    assert first["adversarial_objective_implemented"] is False
    assert first["loss_weights"]["adversarial"] <= MAX_ADVERSARIAL_WEIGHT
    assert 0.0 <= first["input_bounds"][0] <= first["input_bounds"][1] <= 1.0
    assert 0.0 <= first["output_bounds"][0] <= first["output_bounds"][1] <= 1.0
    for step in first["loss_history"]:
        for key, value in step.items():
            if key not in {"epoch", "step"}:
                assert math.isfinite(value) and value >= 0.0
    json.dumps(first, sort_keys=True, allow_nan=False)


def test_onnx_export_has_dynamic_cpu_numerical_parity(
    artifact_root: Path,
    no_network: None,
) -> None:
    checkpoint, training = train(
        artifact_root / "training",
        seed=2718,
        epochs=1,
        max_steps=1,
        batch_size=2,
        config=GraphSRConfig(channels=4, blocks=1, seed=2718),
    )
    output = artifact_root / "export" / "graphsr-x2.onnx"
    report_path = artifact_root / "export" / "onnx-parity.json"

    report = export_onnx(checkpoint, output, report_path)

    assert report["status"] == "pass"
    assert report["provider"] == "cpu"
    assert report["onnx_checker"] == "pass"
    assert report["tolerance"] == 1e-5
    assert report["maximum_absolute_error"] <= 1e-5
    assert report["checkpoint_sha256"] == training["checkpoint_sha256"]
    assert len(report["onnx_sha256"]) == 64
    assert report["input_contract"]["shape"] == ["N", 3, "H", "W"]
    assert report["output_contract"]["shape"] == ["N", 3, "H*2", "W*2"]
    assert report["output_contract"]["coordinate_space"] == "enhanced_pixels"
    assert {case["case"] for case in report["parity_cases"]} == {
        "even_spatial_shape",
        "odd_dynamic_shape",
    }
    onnx.checker.check_model(onnx.load(output))
    assert json.loads(report_path.read_text(encoding="utf-8")) == json.loads(
        json.dumps(report, allow_nan=False)
    )


def test_model_artifacts_are_confined_to_explicit_ignored_roots(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="checkpoints|runs|cache"):
        ensure_artifact_outside_repository(repository_root / "ml" / "graphsr" / "graphsr-x2.pt")
    with pytest.raises(ValueError, match="current project"):
        ensure_artifact_outside_repository(tmp_path / "graphsr-x2.pt")
