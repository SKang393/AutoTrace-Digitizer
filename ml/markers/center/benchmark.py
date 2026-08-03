# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single sealed held-out ONNX Runtime CPU benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import onnxruntime

from .dataset import ARTIFACT_KINDS, SPLIT_FAMILIES, build_fixed_dataset, dataset_manifest
from .metrics import CenterMetrics, aggregate_scene_metrics, center_metrics
from .model import load_checkpoint
from .postprocess import detect_heads


F1_5PX_GATE = 0.90
DUPLICATE_RATE_GATE = 0.02


def _canonical_manifest_sha256() -> str:
    payload = json.dumps(dataset_manifest(), indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hard_negative_hits(scene, detections) -> dict[str, int]:
    hits = {kind: 0 for kind in ARTIFACT_KINDS}
    for kind, x, y in scene.hard_negatives:
        hits[kind] += sum(1 for item in detections if (item.x - x) ** 2 + (item.y - y) ** 2 <= 8.0**2)
    return hits


def _aggregate_mode(
    scene_rows: list[dict[str, object]],
    metrics_3: list[CenterMetrics],
    metrics_5: list[CenterMetrics],
    hard_negative_hits: dict[str, int],
    pixel_count: int,
) -> dict[str, object]:
    aggregate3 = aggregate_scene_metrics(metrics_3, 3.0)
    aggregate5 = aggregate_scene_metrics(metrics_5, 5.0)
    exact_scene_count = sum(
        1
        for row in scene_rows
        if row["metrics_5px"]["false_positives"] == 0
        and row["metrics_5px"]["false_negatives"] == 0
        and row["metrics_5px"]["duplicate_count"] == 0
    )
    return {
        "3px": aggregate3.to_dict(),
        "5px": aggregate5.to_dict(),
        "false_positives_per_megapixel_5px": aggregate5.false_positives / (pixel_count / 1_000_000.0),
        "hard_negative_hits": hard_negative_hits,
        "exact_scene_count": exact_scene_count,
        "scene_count": len(scene_rows),
        "per_scene": scene_rows,
    }


def benchmark(checkpoint: Path, onnx_path: Path, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    _, checkpoint_payload = load_checkpoint(checkpoint)
    current_manifest_sha256 = _canonical_manifest_sha256()
    sealed_manifest_sha256 = checkpoint_payload["dataset_manifest_sha256"]
    if current_manifest_sha256 != sealed_manifest_sha256:
        raise RuntimeError("Generated dataset no longer matches the manifest sealed before training")
    scenes = build_fixed_dataset("test")
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    threshold = float(checkpoint_payload["selected_threshold"])
    modes: dict[str, dict[str, object]] = {}
    inference_ms = 0.0
    for mode in ("standard_masks", "zero_masks"):
        rows: list[dict[str, object]] = []
        metrics_3: list[CenterMetrics] = []
        metrics_5: list[CenterMetrics] = []
        hard_negative_hits = {kind: 0 for kind in ARTIFACT_KINDS}
        pixels = 0
        for scene in scenes:
            tensor = scene.tensor.unsqueeze(0).numpy().copy()
            if mode == "zero_masks":
                tensor[:, 1:] = 0.0
            inference_started = time.perf_counter()
            heads = session.run([output_name], {input_name: tensor})[0]
            inference_ms += (time.perf_counter() - inference_started) * 1000.0
            detections = detect_heads(
                heads,
                text_mask=tensor[0, 1],
                artifact_mask=tensor[0, 2],
                center_threshold=threshold,
            )
            metric3 = center_metrics(detections, scene.centers, 3.0)
            metric5 = center_metrics(detections, scene.centers, 5.0)
            metrics_3.append(metric3)
            metrics_5.append(metric5)
            pixels += tensor.shape[-2] * tensor.shape[-1]
            hits = _hard_negative_hits(scene, detections)
            for kind, count in hits.items():
                hard_negative_hits[kind] += count
            rows.append(
                {
                    "scene_id": scene.scene_id,
                    "family": scene.family,
                    "degradation": scene.degradation,
                    "truth_count": len(scene.centers),
                    "prediction_count": len(detections),
                    "metrics_3px": metric3.to_dict(),
                    "metrics_5px": metric5.to_dict(),
                    "hard_negative_hits": hits,
                }
            )
        modes[mode] = _aggregate_mode(rows, metrics_3, metrics_5, hard_negative_hits, pixels)
    standard = modes["standard_masks"]
    gate_checks = {
        "f1_5px": standard["5px"]["f1"] >= F1_5PX_GATE,
        "duplicate_rate": standard["5px"]["duplicate_rate"] < DUPLICATE_RATE_GATE,
        "hard_negative_hits": not any(standard["hard_negative_hits"].values()),
        "one_center_per_golden_fixture": standard["exact_scene_count"] == standard["scene_count"],
    }
    report: dict[str, object] = {
        "status": "pass" if all(gate_checks.values()) else "fail",
        "heldout_evaluation_count": 1,
        "heldout_selected_after_model_selection": True,
        "heldout_split": "test",
        "heldout_families": SPLIT_FAMILIES["test"],
        "dataset_manifest_sha256": sealed_manifest_sha256,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "provider": "cpu",
        "provider_runtime": "onnxruntime",
        "provider_runtime_version": onnxruntime.__version__,
        "selected_threshold": threshold,
        "gate_values": {
            "f1_5px_min": F1_5PX_GATE,
            "duplicate_rate_max_exclusive": DUPLICATE_RATE_GATE,
            "hard_negative_hits_max": 0,
            "one_center_per_golden_fixture": True,
        },
        "gate_checks": gate_checks,
        "modes": modes,
        "timing_ms": {
            "inference_total_two_modes": round(inference_ms, 3),
            "inference_mean_per_scene_mode": round(inference_ms / (2 * len(scenes)), 3),
            "benchmark_total": round((time.perf_counter() - started) * 1000.0, 3),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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
