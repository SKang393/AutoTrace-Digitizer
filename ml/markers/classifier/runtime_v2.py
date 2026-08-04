# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Probability-packed marker-classifier runtime contract and ONNX export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from .dataset import FILL_NAMES, SHAPE_NAMES, PatchSample
from .model import load_checkpoint


RUNTIME_V2_OUTPUT_NAME = "classification_probabilities"
RUNTIME_V2_OUTPUT_WIDTH = 25
RUNTIME_V2_SLICES = {
    "shape_probabilities": (0, 9),
    "fill_probabilities": (9, 12),
    "artifact_probability": (12, 13),
    "embedding": (13, 25),
}
PARITY_TOLERANCE = 1e-5


class ProbabilityPackedRuntimeClassifier(nn.Module):
    """Pack calibrated probabilities and the normalized embedding for transport."""

    def __init__(self, classifier: nn.Module, shape_temperature: float, fill_temperature: float) -> None:
        super().__init__()
        if not np.isfinite(shape_temperature) or shape_temperature <= 0:
            raise ValueError("shape_temperature must be finite and positive")
        if not np.isfinite(fill_temperature) or fill_temperature <= 0:
            raise ValueError("fill_temperature must be finite and positive")
        self.classifier = classifier
        self.shape_temperature = float(shape_temperature)
        self.fill_temperature = float(fill_temperature)

    def forward(self, value: Tensor) -> Tensor:
        shape, fill, artifact, embedding = self.classifier(value)
        return torch.cat(
            (
                torch.softmax(shape / self.shape_temperature, dim=1),
                torch.softmax(fill / self.fill_temperature, dim=1),
                torch.sigmoid(artifact),
                embedding,
            ),
            dim=1,
        )


def _validate_probability_contract(values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != RUNTIME_V2_OUTPUT_WIDTH:
        raise ValueError(f"Expected packed output [N,{RUNTIME_V2_OUTPUT_WIDTH}], received {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Runtime output contains a non-finite value")
    shape = values[:, 0:9]
    fill = values[:, 9:12]
    artifact = values[:, 12]
    if not np.allclose(shape.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Shape probabilities do not sum to one")
    if not np.allclose(fill.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Fill probabilities do not sum to one")
    if np.any(shape < 0.0) or np.any(shape > 1.0) or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("Shape or fill probability is outside [0,1]")
    if np.any(artifact < 0.0) or np.any(artifact > 1.0):
        raise ValueError("Artifact probability is outside [0,1]")


def run_probability_runtime(
    checkpoint: Path,
    onnx_path: Path,
    samples: tuple[PatchSample, ...],
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    """Run PyTorch and CPU ONNX over a fixed split and return direct parity."""

    model, payload = load_checkpoint(checkpoint)
    runtime_model = ProbabilityPackedRuntimeClassifier(
        model,
        float(payload["shape_temperature"]),
        float(payload["fill_temperature"]),
    ).eval()
    session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
    expected_rows: list[np.ndarray] = []
    actual_rows: list[np.ndarray] = []
    maximum_error = 0.0
    inference_ms = 0.0
    for start in range(0, len(samples), batch_size):
        tensor = torch.stack([sample.tensor for sample in samples[start : start + batch_size]])
        with torch.inference_mode():
            expected = runtime_model(tensor).numpy()
        inference_started = time.perf_counter()
        actual = session.run(
            [RUNTIME_V2_OUTPUT_NAME],
            {model.contract.input_name: tensor.numpy()},
        )[0]
        inference_ms += (time.perf_counter() - inference_started) * 1000.0
        _validate_probability_contract(actual)
        maximum_error = max(maximum_error, float(np.max(np.abs(expected - actual))))
        expected_rows.append(expected)
        actual_rows.append(actual)
    return (
        np.concatenate(expected_rows, axis=0),
        np.concatenate(actual_rows, axis=0),
        maximum_error,
        inference_ms,
        session.get_providers()[0],
    )


def export_probability_onnx(
    checkpoint: Path,
    output: Path,
    report_path: Path,
    selection_samples: tuple[PatchSample, ...],
) -> dict[str, object]:
    """Export one unchanged checkpoint and verify the whole fixed selection split."""

    started = time.perf_counter()
    model, payload = load_checkpoint(checkpoint)
    runtime_model = ProbabilityPackedRuntimeClassifier(
        model,
        float(payload["shape_temperature"]),
        float(payload["fill_temperature"]),
    ).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.stack([sample.tensor for sample in selection_samples[:8]])
    torch.onnx.export(
        runtime_model,
        example,
        output,
        input_names=[model.contract.input_name],
        output_names=[RUNTIME_V2_OUTPUT_NAME],
        dynamic_axes={
            model.contract.input_name: {0: "batch"},
            RUNTIME_V2_OUTPUT_NAME: {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    _, actual, maximum_error, inference_ms, provider = run_probability_runtime(
        checkpoint,
        output,
        selection_samples,
    )
    report: dict[str, object] = {
        "status": "pass" if maximum_error <= PARITY_TOLERANCE else "fail",
        "provider": provider,
        "opset": 18,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "onnx_size_bytes": output.stat().st_size,
        "maximum_absolute_error": maximum_error,
        "tolerance": PARITY_TOLERANCE,
        "selection_sample_count": len(selection_samples),
        "public_gate_evaluations": 0,
        "optimizer_steps": 0,
        "weights_changed": False,
        "runtime_tensor_contract": {
            "input_name": model.contract.input_name,
            "input_layout": "NCHW",
            "input_shape": ["N", 1, 32, 32],
            "output_name": RUNTIME_V2_OUTPUT_NAME,
            "output_layout": "NC",
            "output_shape": ["N", RUNTIME_V2_OUTPUT_WIDTH],
            "output_order": [
                "calibrated_shape_probabilities[9]",
                "calibrated_fill_probabilities[3]",
                "artifact_probability[1]",
                "l2_normalized_embedding[12]",
            ],
            "shape_temperature": float(payload["shape_temperature"]),
            "fill_temperature": float(payload["fill_temperature"]),
            "dynamic_axes": ["batch"],
        },
        "probability_contract": {
            "shape_sum_max_abs_error": float(np.max(np.abs(actual[:, 0:9].sum(axis=1) - 1.0))),
            "fill_sum_max_abs_error": float(np.max(np.abs(actual[:, 9:12].sum(axis=1) - 1.0))),
            "minimum_probability": float(actual[:, 0:13].min()),
            "maximum_probability": float(actual[:, 0:13].max()),
        },
        "inference_total_ms": round(inference_ms, 3),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "PARITY_TOLERANCE",
    "ProbabilityPackedRuntimeClassifier",
    "RUNTIME_V2_OUTPUT_NAME",
    "RUNTIME_V2_OUTPUT_WIDTH",
    "RUNTIME_V2_SLICES",
    "export_probability_onnx",
    "run_probability_runtime",
]
