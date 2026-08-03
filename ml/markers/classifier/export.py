# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export the selected classifier to ONNX and verify validation parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from .dataset import build_fixed_dataset
from .model import load_checkpoint


RUNTIME_OUTPUT_NAME = "classification_heads"
RUNTIME_OUTPUT_WIDTH = 25
RUNTIME_SLICES = {
    "shape_logits": (0, 9),
    "fill_logits": (9, 12),
    "artifact_logit": (12, 13),
    "embedding": (13, 25),
}


class PackedRuntimeClassifier(nn.Module):
    """Pack heads while applying checkpoint calibration to runtime logits."""

    def __init__(self, classifier: nn.Module, shape_temperature: float = 1.0, fill_temperature: float = 1.0) -> None:
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
                shape / self.shape_temperature,
                fill / self.fill_temperature,
                artifact,
                embedding,
            ),
            dim=1,
        )


def export_onnx(checkpoint: Path, output: Path, report_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    model, payload = load_checkpoint(checkpoint)
    shape_temperature = float(payload["shape_temperature"])
    fill_temperature = float(payload["fill_temperature"])
    runtime_model = PackedRuntimeClassifier(model, shape_temperature, fill_temperature).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.stack([sample.tensor for sample in build_fixed_dataset("validation")[:8]])
    torch.onnx.export(
        runtime_model,
        example,
        output,
        input_names=[model.contract.input_name],
        output_names=[RUNTIME_OUTPUT_NAME],
        dynamic_axes={model.contract.input_name: {0: "batch"}, RUNTIME_OUTPUT_NAME: {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    with torch.inference_mode():
        separate_outputs = [value.numpy() for value in model(example)]
        packed_expected = runtime_model(example).numpy()
    session = ort.InferenceSession(output.read_bytes(), providers=["CPUExecutionProvider"])
    packed_actual = session.run([RUNTIME_OUTPUT_NAME], {model.contract.input_name: example.numpy()})[0]
    if packed_actual.shape != (len(example), RUNTIME_OUTPUT_WIDTH):
        raise ValueError(f"Expected packed output [N,{RUNTIME_OUTPUT_WIDTH}], received {packed_actual.shape}")
    packed_error = float(np.max(np.abs(packed_expected - packed_actual)))
    slice_parity = {}
    runtime_expected = {
        "shape_logits": separate_outputs[0] / shape_temperature,
        "fill_logits": separate_outputs[1] / fill_temperature,
        "artifact_logit": separate_outputs[2],
        "embedding": separate_outputs[3],
    }
    for name in model.contract.output_names:
        separate = runtime_expected[name]
        start, end = RUNTIME_SLICES[name]
        expected_slice = packed_expected[:, start:end]
        if not np.array_equal(expected_slice, separate):
            raise ValueError(f"PyTorch packed slice {name} does not exactly match its runtime transform")
        slice_parity[name] = {
            "start_inclusive": start,
            "end_exclusive": end,
            "width": end - start,
            "pytorch_runtime_transform_exact": True,
            "onnx_max_abs_error": float(np.max(np.abs(separate - packed_actual[:, start:end]))),
        }
    maximum = max(value["onnx_max_abs_error"] for value in slice_parity.values())
    report: dict[str, object] = {
        "status": "pass" if maximum <= 1e-5 else "fail",
        "provider": session.get_providers()[0],
        "opset": 18,
        "training_tensor_contract": payload["tensor_contract"],
        "runtime_tensor_contract": {
            "input_name": model.contract.input_name,
            "input_layout": "NCHW",
            "input_shape": ["N", 1, 32, 32],
            "output_name": RUNTIME_OUTPUT_NAME,
            "output_layout": "NC",
            "output_shape": ["N", RUNTIME_OUTPUT_WIDTH],
            "output_order": [
                "temperature_scaled_shape_logits[9]",
                "temperature_scaled_fill_logits[3]",
                "artifact_logit[1]",
                "embedding[12]",
            ],
            "shape_temperature": shape_temperature,
            "fill_temperature": fill_temperature,
            "dynamic_axes": ["batch"],
        },
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "onnx_size_bytes": output.stat().st_size,
        "packed_max_abs_error": packed_error,
        "slice_parity": slice_parity,
        "maximum_absolute_error": maximum,
        "tolerance": 1e-5,
        "validation_sample_count": len(example),
        "heldout_test_evaluations": 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = export_onnx(args.checkpoint, args.output, args.report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PackedRuntimeClassifier",
    "RUNTIME_OUTPUT_NAME",
    "RUNTIME_OUTPUT_WIDTH",
    "RUNTIME_SLICES",
    "export_onnx",
]
