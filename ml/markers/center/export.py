# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Exact ONNX export and ONNX Runtime CPU parity evidence."""

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

from .dataset import build_fixed_dataset
from .model import load_checkpoint


ONNX_OPSET = 18


def export_onnx(checkpoint: Path, output: Path, report_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    model, checkpoint_payload = load_checkpoint(checkpoint)
    validation_scene = build_fixed_dataset("validation")[0]
    sample = validation_scene.tensor.unsqueeze(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        output,
        input_names=[model.contract.input_name],
        output_names=[model.contract.output_name],
        dynamic_axes={
            model.contract.input_name: {0: "batch", 2: "height", 3: "width"},
            model.contract.output_name: {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    comparisons = []
    max_abs_difference = 0.0
    for name, tensor in (
        ("validation_standard", sample),
        ("validation_zero_masks", sample.clone()),
    ):
        if name.endswith("zero_masks"):
            tensor[:, 1:].zero_()
        with torch.inference_mode():
            torch_output = model(tensor).cpu().numpy()
        ort_output = session.run(
            [model.contract.output_name],
            {model.contract.input_name: tensor.cpu().numpy()},
        )[0]
        difference = float(np.max(np.abs(torch_output - ort_output)))
        max_abs_difference = max(max_abs_difference, difference)
        comparisons.append({"case": name, "max_abs_difference": difference})
    tolerance = 1e-5
    report: dict[str, object] = {
        "status": "pass" if max_abs_difference <= tolerance else "fail",
        "training_revision": checkpoint_payload["training_revision"],
        "dataset_manifest_sha256": checkpoint_payload["dataset_manifest_sha256"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_path": str(output),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "onnx_bytes": output.stat().st_size,
        "onnx_opset": ONNX_OPSET,
        "onnx_checker": "pass",
        "tensor_contract": asdict(model.contract),
        "provider": "cpu",
        "provider_runtime": "onnxruntime",
        "provider_runtime_version": onnxruntime.__version__,
        "parity_tolerance": tolerance,
        "parity_max_abs_difference": max_abs_difference,
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
