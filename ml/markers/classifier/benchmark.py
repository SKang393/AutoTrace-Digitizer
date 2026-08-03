# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use sealed held-out benchmark for the selected marker classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from .dataset import ARTIFACT_KINDS, FILL_NAMES, SHAPE_NAMES, PatchSample, build_fixed_dataset, dataset_manifest
from .metrics import classification_metrics
from .model import load_checkpoint
from .train import LOCAL_FILL_MACRO_F1_GATE, LOCAL_SHAPE_MACRO_F1_GATE, collect_outputs, summarize_outputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_subset(model, samples: tuple[PatchSample, ...], shape_temperature: float, fill_temperature: float) -> dict[str, object]:
    return summarize_outputs(collect_outputs(model, samples), shape_temperature, fill_temperature)


def benchmark(checkpoint: Path, onnx_path: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seal = output.parent / "heldout-evaluation.seal.json"
    if seal.exists():
        raise RuntimeError(f"Held-out split was already opened; refusing a repeated evaluation: {seal}")
    seal.write_text(json.dumps({"status": "opened", "evaluation_count": 1}, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        model, payload = load_checkpoint(checkpoint)
        test_samples = build_fixed_dataset("test")
        shape_temperature = float(payload["shape_temperature"])
        fill_temperature = float(payload["fill_temperature"])
        metrics = _metric_subset(model, test_samples, shape_temperature, fill_temperature)
        per_family = {
            family: _metric_subset(model, tuple(sample for sample in test_samples if sample.family == family), shape_temperature, fill_temperature)
            for family in sorted({sample.family for sample in test_samples})
        }
        per_template = {
            template: _metric_subset(model, tuple(sample for sample in test_samples if sample.template == template), shape_temperature, fill_temperature)
            for template in sorted({sample.template for sample in test_samples})
        }
        marker_samples = tuple(sample for sample in test_samples if sample.artifact < 0.5)
        outputs = collect_outputs(model, marker_samples)
        shape_probabilities = torch.softmax(outputs["shape_logits"] / shape_temperature, dim=1).numpy()
        shape_targets = outputs["shape_targets"].numpy()
        shape_detail = classification_metrics(shape_probabilities, shape_targets, len(SHAPE_NAMES))
        minority_indices = [SHAPE_NAMES.index(name) for name in ("star", "asterisk", "cross")]
        minority_macro_f1 = float(np.mean([shape_detail.per_class_f1[index] for index in minority_indices]))

        all_outputs = collect_outputs(model, test_samples)
        artifact_probabilities = torch.sigmoid(all_outputs["artifact_logits"]).numpy()
        artifact_by_kind = {}
        for kind in ARTIFACT_KINDS:
            indices = [index for index, sample in enumerate(test_samples) if sample.artifact_kind == kind]
            artifact_by_kind[kind] = {
                "count": len(indices),
                "detected_count": int((artifact_probabilities[indices] >= 0.5).sum()),
                "mean_probability": float(artifact_probabilities[indices].mean()),
            }

        session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
        parity_max = {name: 0.0 for name in model.contract.output_names}
        inference_started = time.perf_counter()
        for start in range(0, len(test_samples), 64):
            tensor = torch.stack([sample.tensor for sample in test_samples[start : start + 64]])
            with torch.inference_mode():
                expected = [value.numpy() for value in model(tensor)]
            actual = session.run(list(model.contract.output_names), {model.contract.input_name: tensor.numpy()})
            for name, left, right in zip(model.contract.output_names, expected, actual, strict=True):
                parity_max[name] = max(parity_max[name], float(np.max(np.abs(left - right))))
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        maximum_parity_error = max(parity_max.values())

        test_manifest_payload = json.dumps(dataset_manifest(include_test=True), indent=2, sort_keys=True) + "\n"
        test_manifest_sha256 = hashlib.sha256(test_manifest_payload.encode("utf-8")).hexdigest()
        shape_pass = float(metrics["shape"]["macro_f1"]) >= LOCAL_SHAPE_MACRO_F1_GATE
        fill_pass = float(metrics["fill"]["macro_f1"]) >= LOCAL_FILL_MACRO_F1_GATE
        parity_pass = maximum_parity_error <= 1e-5
        report: dict[str, object] = {
            "status": "pass" if shape_pass and fill_pass and parity_pass else "fail",
            "evaluation_count": 1,
            "selection_state": "test split opened only after checkpoint and ONNX selection",
            "checkpoint_sha256": _sha256(checkpoint),
            "onnx_sha256": _sha256(onnx_path),
            "heldout_dataset_manifest_sha256": test_manifest_sha256,
            "heldout_sample_count": len(test_samples),
            "heldout_families": sorted({sample.family for sample in test_samples}),
            "heldout_templates": sorted({sample.template for sample in test_samples}),
            "metrics": metrics,
            "per_family": per_family,
            "per_template": per_template,
            "minority_shape_macro_f1": minority_macro_f1,
            "artifact_by_kind": artifact_by_kind,
            "local_gates": {
                "shape_macro_f1": LOCAL_SHAPE_MACRO_F1_GATE,
                "fill_macro_f1": LOCAL_FILL_MACRO_F1_GATE,
                "authority": "session-local preregistration; not maintainer-agreed",
            },
            "gate_results": {"shape": shape_pass, "fill": fill_pass, "onnx_parity": parity_pass},
            "onnx_provider": session.get_providers()[0],
            "onnx_per_head_max_abs_error": parity_max,
            "onnx_maximum_absolute_error": maximum_parity_error,
            "onnx_tolerance": 1e-5,
            "inference_total_ms": round(inference_ms, 3),
            "inference_ms_per_patch": round(inference_ms / len(test_samples), 6),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "release_eligible": False,
            "release_blocker": "Model manifest, packaging, Windows runtime integration, and maintainer-agreed numeric gate are outside this ML-owned path.",
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal.write_text(
            json.dumps({"status": "completed", "evaluation_count": 1, "benchmark_sha256": _sha256(output)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception as error:
        seal.write_text(json.dumps({"status": "failed_after_open", "evaluation_count": 1, "error_type": type(error).__name__}, sort_keys=True) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.checkpoint, args.onnx, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
