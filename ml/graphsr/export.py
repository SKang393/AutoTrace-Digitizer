# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export GraphSR-x2 to ONNX and verify CPU numerical parity."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime
import torch

from .model import ensure_artifact_outside_repository, load_checkpoint


ONNX_OPSET = 18
PARITY_TOLERANCE = 1e-5


def _parity_inputs() -> tuple[tuple[str, torch.Tensor], ...]:
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    return (
        ("even_spatial_shape", torch.rand((1, 3, 16, 18), generator=generator)),
        ("odd_dynamic_shape", torch.rand((2, 3, 17, 19), generator=generator)),
    )


def export_onnx(checkpoint: Path, output: Path, report_path: Path) -> dict[str, object]:
    """Export a trusted local checkpoint and fail if ONNX Runtime diverges."""

    started = time.perf_counter()
    output = ensure_artifact_outside_repository(output)
    report_path = ensure_artifact_outside_repository(report_path)
    model, payload = load_checkpoint(checkpoint)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = _parity_inputs()[0][1]
    torch.onnx.export(
        model,
        (sample,),
        output,
        input_names=[model.contract.input_name],
        output_names=[model.contract.output_name],
        dynamic_axes={
            model.contract.input_name: {0: "batch", 2: "height", 3: "width"},
            model.contract.output_name: {0: "batch", 2: "height_x2", 3: "width_x2"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    session = onnxruntime.InferenceSession(output.read_bytes(), providers=["CPUExecutionProvider"])
    comparisons: list[dict[str, object]] = []
    maximum = 0.0
    for case, tensor in _parity_inputs():
        with torch.inference_mode():
            expected = model(tensor).cpu().numpy()
        actual = session.run(
            [model.contract.output_name],
            {model.contract.input_name: tensor.numpy()},
        )[0]
        expected_shape = (tensor.shape[0], 3, tensor.shape[2] * 2, tensor.shape[3] * 2)
        if actual.shape != expected_shape:
            raise ValueError(f"ONNX output shape {actual.shape} did not match expected {expected_shape}")
        difference = float(np.max(np.abs(expected - actual)))
        maximum = max(maximum, difference)
        comparisons.append(
            {
                "case": case,
                "input_shape": list(tensor.shape),
                "output_shape": list(actual.shape),
                "maximum_absolute_error": difference,
            }
        )
    status = "pass" if maximum <= PARITY_TOLERANCE else "fail"
    report: dict[str, object] = {
        "status": status,
        "architecture": model.config.architecture,
        "training_revision": payload["training_revision"],
        "dataset_identity": payload["dataset_identity"],
        "input_contract": {
            "name": model.contract.input_name,
            "layout": model.contract.input_layout,
            "dtype": model.contract.input_dtype,
            "shape": ["N", 3, "H", "W"],
            "range": list(model.contract.input_range),
        },
        "output_contract": {
            "name": model.contract.output_name,
            "layout": model.contract.output_layout,
            "dtype": model.contract.output_dtype,
            "shape": ["N", 3, "H*2", "W*2"],
            "range": list(model.contract.output_range),
            "coordinate_space": model.contract.coordinate_space,
            "coordinate_mapping": model.contract.coordinate_mapping,
        },
        "tensor_contract": asdict(model.contract),
        "provider": "cpu",
        "provider_name": session.get_providers()[0],
        "provider_runtime": "onnxruntime",
        "provider_runtime_version": onnxruntime.__version__,
        "opset": ONNX_OPSET,
        "onnx_checker": "pass",
        "deterministic_export": True,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "onnx_size_bytes": output.stat().st_size,
        "maximum_absolute_error": maximum,
        "tolerance": PARITY_TOLERANCE,
        "parity_cases": comparisons,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
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


__all__ = ["ONNX_OPSET", "PARITY_TOLERANCE", "export_onnx"]
