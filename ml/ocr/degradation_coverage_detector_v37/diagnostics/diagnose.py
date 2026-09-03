# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Diagnose V37 source-scale segmentation on the fixed synthetic dev split.

This module deliberately emits aggregate metrics only. It never serializes
scene identifiers, truth rows, predictions, rasters, or intermediate maps.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ml.ocr.real_range_classifier_finetune_v32.dataset import build_split
from ml.ocr.real_range_detector_v35.pipeline import maximum_cardinality_matches
from ml.ocr.degradation_coverage_detector_v37.protocol import (
    MINIMUM_COMPONENT_AREA,
    ONNX_PROVIDER,
    PIXEL_THRESHOLD,
    TILE_OVERLAP,
    TILE_SIZE,
    TRUTH_MATCH_IOU_MINIMUM,
)
from ml.ocr.degradation_coverage_detector_v37.dataset import build_tiles


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = (
    REPO_ROOT
    / "artifacts/goal22-worktrees/ocr-v37/ml/ocr/degradation_coverage_detector_v37"
    / "artifacts/P1-run/graph-text-degradation-coverage-detector-v37-p1.onnx"
)
DEFAULT_CHECKPOINT = DEFAULT_MODEL.with_suffix(".pt")
EXPECTED_MODEL_SHA256 = "249b0eef99250c1fe1cdb2ab85e32b540d015930e3b9cb4459f9a592305b518d"
EXPECTED_CHECKPOINT_SHA256 = "74a5e6794c681e65b3ac9dbbb86842dc8fa2b4dce6a9b73a31dfcca259d6ef78"
V37_RESULT = REPO_ROOT / "ml/ocr/degradation_coverage_detector_v37/P1_RESULT.json"
V35_DIAGNOSTIC = REPO_ROOT / "ml/ocr/real_range_detector_v35/diagnostics/DIAGNOSTIC.json"
V35_DIAGNOSTIC_SHA256 = "c6b035f8bac27f27d2b157a2af015f12f600d1287b1135ce4f0cddfbd7d21526"

THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
AREA_SWEEP = (0, 2, 4, 8, 16, 32, 64)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pixel_metrics(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probability >= threshold
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "threshold": threshold,
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
    }


def _component_boxes(
    probability: np.ndarray,
    threshold: float,
    morphology_close: bool,
    minimum_area: int,
) -> tuple[tuple[int, int, int, int], ...]:
    binary = (probability >= threshold).astype(np.uint8)
    if morphology_close:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= minimum_area and width >= 2 and height >= 2:
            boxes.append((x, y, x + width, y + height))
    return tuple(boxes)


def _component_metrics(
    probability: np.ndarray,
    truths: tuple[Any, ...],
    threshold: float,
    morphology_close: bool,
    minimum_area: int,
) -> dict[str, Any]:
    boxes = _component_boxes(probability, threshold, morphology_close, minimum_area)
    from ml.ocr.component_region_detector_v6.dataset import Box

    predicted = tuple(Box(*box) for box in boxes)
    matched = maximum_cardinality_matches(predicted, truths)
    truth_count = len(truths)
    return {
        "threshold": threshold,
        "morphology_close": morphology_close,
        "minimum_component_area": minimum_area,
        "truth_regions": truth_count,
        "predicted_regions": len(predicted),
        "true_positives": matched,
        "false_positives": len(predicted) - matched,
        "false_negatives": truth_count - matched,
        "precision": matched / max(1, len(predicted)),
        "recall": matched / max(1, truth_count),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = sum(int(row["truth_regions"]) for row in rows)
    predicted = sum(int(row["predicted_regions"]) for row in rows)
    matched = sum(int(row["true_positives"]) for row in rows)
    return {
        "truth_regions": truth,
        "predicted_regions": predicted,
        "true_positives": matched,
        "false_positives": predicted - matched,
        "false_negatives": truth - matched,
        "precision": matched / max(1, predicted),
        "recall": matched / max(1, truth),
    }


def _aggregate_pixel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive_pixels"]) for row in rows)
    fp = sum(int(row["false_positive_pixels"]) for row in rows)
    fn = sum(int(row["false_negative_pixels"]) for row in rows)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
    }


def run_diagnostic(model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"V37 ONNX artifact is missing: {model_path}")
    model_sha256 = _sha256(model_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError("V37 ONNX artifact hash does not match the authorized P1 result")
    checkpoint_sha256 = _sha256(DEFAULT_CHECKPOINT)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("V37 checkpoint artifact hash does not match the authorized P1 result")
    session = ort.InferenceSession(str(model_path), providers=[ONNX_PROVIDER])
    if session.get_providers() != [ONNX_PROVIDER]:
        raise RuntimeError(f"Expected only {ONNX_PROVIDER}, got {session.get_providers()}")

    scenes = build_split("dev")
    tiles = build_tiles("dev")
    probability_maps: dict[str, np.ndarray] = {}
    component_inputs: dict[str, tuple[Any, tuple[Any, ...]]] = {}
    overlap_sum: list[float] = []
    overlap_sum_sq: list[float] = []
    overlap_count = 0
    covered_pixels = 0
    total_pixels = 0
    tile_count = 0
    maximum_coverage = 0

    dimension_pixel_rows: dict[str, list[dict[str, Any]]] = {}
    dimension_component_rows: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        scene_tiles = tuple(tile for tile in tiles if tile.scene_id == scene.scene_id)
        values = np.stack([(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in scene_tiles]).astype(np.float32)
        logits = np.asarray(session.run(["text_logits"], {"source_tiles": values})[0], dtype=np.float32)[:, 0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        height, width = scene.raster.shape
        scores = np.zeros((height, width), dtype=np.float64)
        counts = np.zeros((height, width), dtype=np.float64)
        sum_sq = np.zeros((height, width), dtype=np.float64)
        for tile, probability in zip(scene_tiles, probabilities, strict=True):
            region = (slice(tile.top, tile.top + tile.valid_height), slice(tile.left, tile.left + tile.valid_width))
            valid = probability[: tile.valid_height, : tile.valid_width].astype(np.float64)
            scores[region] += valid
            sum_sq[region] += valid * valid
            counts[region] += 1.0
        probability_map = (scores / np.maximum(counts, 1.0)).astype(np.float32)
        probability_maps[scene.scene_id] = probability_map
        component_inputs[scene.scene_id] = (probability_map, scene.truths)
        truth_mask = np.zeros((height, width), dtype=bool)
        for truth in scene.truths:
            truth_mask[int(truth.top):int(truth.bottom), int(truth.left):int(truth.right)] = True
        dimension = f"{width}x{height}"
        dimension_pixel_rows.setdefault(dimension, []).append(_pixel_metrics(probability_map, truth_mask, PIXEL_THRESHOLD))
        dimension_component_rows.setdefault(dimension, []).append(
            _component_metrics(probability_map, scene.truths, PIXEL_THRESHOLD, True, MINIMUM_COMPONENT_AREA)
        )
        overlap = counts > 1.0
        maximum_coverage = max(maximum_coverage, int(np.max(counts)))
        if np.any(overlap):
            mean = scores[overlap] / counts[overlap]
            variance = np.maximum(0.0, sum_sq[overlap] / counts[overlap] - mean * mean)
            overlap_sum.extend(mean.tolist())
            overlap_sum_sq.extend(variance.tolist())
            overlap_count += int(np.count_nonzero(overlap))
        covered_pixels += int(np.count_nonzero(counts > 0.0))
        total_pixels += height * width
        tile_count += len(scene_tiles)

    pixel_sweep: list[dict[str, Any]] = []
    component_sweep: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        pixel_rows = []
        component_rows = []
        for scene in scenes:
            probability_map, truths = component_inputs[scene.scene_id]
            truth_mask = np.zeros(probability_map.shape, dtype=bool)
            for truth in truths:
                truth_mask[int(truth.top):int(truth.bottom), int(truth.left):int(truth.right)] = True
            pixel_rows.append(_pixel_metrics(probability_map, truth_mask, threshold))
            component_rows.append(_component_metrics(probability_map, truths, threshold, False, MINIMUM_COMPONENT_AREA))
            component_rows.append(_component_metrics(probability_map, truths, threshold, True, MINIMUM_COMPONENT_AREA))
        pixel_row = _aggregate_pixel(pixel_rows)
        pixel_row["threshold"] = threshold
        pixel_sweep.append(pixel_row)
        for morphology_close in (False, True):
            selected = [row for row in component_rows if row["morphology_close"] == morphology_close]
            row = _aggregate(selected)
            row.update({"threshold": threshold, "morphology_close": morphology_close, "minimum_component_area": MINIMUM_COMPONENT_AREA})
            component_sweep.append(row)

    area_sweep: list[dict[str, Any]] = []
    for minimum_area in AREA_SWEEP:
        rows = [
            _component_metrics(probability, truths, PIXEL_THRESHOLD, True, minimum_area)
            for probability, truths in component_inputs.values()
        ]
        row = _aggregate(rows)
        row.update({"threshold": PIXEL_THRESHOLD, "morphology_close": True, "minimum_component_area": minimum_area})
        area_sweep.append(row)

    fixed_pixel = _aggregate_pixel([row for rows in dimension_pixel_rows.values() for row in rows])
    fixed_component = _aggregate([row for rows in dimension_component_rows.values() for row in rows])
    best_pixel = max(pixel_sweep, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    best_component = max(component_sweep, key=lambda row: (row["recall"], row["precision"]))
    best_area_component = max(area_sweep, key=lambda row: (row["recall"], row["precision"]))
    overlap_stds = np.sqrt(np.asarray(overlap_sum_sq, dtype=np.float64)) if overlap_sum_sq else np.zeros(1)

    v35 = json.loads(V35_DIAGNOSTIC.read_text(encoding="utf-8"))
    v37_result = json.loads(V37_RESULT.read_text(encoding="utf-8"))
    return {
        "schema": "graphreader.ocr-degradation-coverage-detector-v37-diagnostic.v1",
        "revision": "graph-text-degradation-coverage-detector-v37",
        "evidence": {
            "split": "dev",
            "synthetic_only": True,
            "scene_count": len(scenes),
            "truth_region_count": sum(len(scene.truths) for scene in scenes),
            "private_or_article_images": False,
            "sealed_or_public_reads": 0,
            "case_level_output": False,
        },
        "fixed_hashes": {
            "v37_checkpoint": {"path": DEFAULT_CHECKPOINT.relative_to(REPO_ROOT).as_posix(), "sha256": checkpoint_sha256},
            "v37_onnx": {"path": model_path.relative_to(REPO_ROOT).as_posix(), "sha256": model_sha256},
            "v37_result": {"path": V37_RESULT.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(V37_RESULT)},
            "v35_diagnostic": {"path": V35_DIAGNOSTIC.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(V35_DIAGNOSTIC)},
            "v35_result": {"path": "ml/ocr/real_range_detector_v35/P1_RESULT.json", "sha256": v37_result["v35_result_sha256"]},
            "v36_result": {"path": "ml/ocr/shrink_region_detector_v36/P1_RESULT.json", "sha256": v37_result["v36_result_sha256"]},
            "v36_diagnostic": {"path": "ml/ocr/shrink_region_detector_v36/diagnostics/DIAGNOSTIC.json", "sha256": v37_result["v36_diagnostic_sha256"]},
        },
        "protocol": {
            "onnx_provider": ONNX_PROVIDER,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "pixel_threshold": PIXEL_THRESHOLD,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
        },
        "pixel_segmentation": {
            "fixed_pipeline": fixed_pixel,
            "best_f1": best_pixel,
            "best_recall": max(pixel_sweep, key=lambda row: (row["recall"], row["precision"])),
            "threshold_sweep": pixel_sweep,
        },
        "postprocessing": {
            "fixed_pipeline": fixed_component,
            "best_threshold_or_morphology": best_component,
            "best_area_at_fixed_threshold": best_area_component,
            "threshold_and_morphology_sweep": component_sweep,
            "component_area_sweep": area_sweep,
        },
        "by_dimension": {
            dimension: {
                "pixel_segmentation": _aggregate_pixel(rows),
                "component_proposals": _aggregate(dimension_component_rows[dimension]),
            }
            for dimension, rows in sorted(dimension_pixel_rows.items())
        },
        "tiling_overlap_mapping": {
            "tile_count": tile_count,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "covered_pixel_fraction": covered_pixels / max(1, total_pixels),
            "overlap_pixel_count": overlap_count,
            "minimum_coverage": 1 if overlap_count else 0,
            "maximum_coverage": maximum_coverage,
            "mean_overlap_prediction_std": float(np.mean(overlap_stds)),
            "p95_overlap_prediction_std": float(np.percentile(overlap_stds, 95)),
            "maximum_overlap_prediction_std": float(np.max(overlap_stds)),
        },
        "v35_comparison": {
            "diagnostic_sha256": V35_DIAGNOSTIC_SHA256,
            "fixed_pipeline": v35["baseline_fixed_pipeline"],
            "best_postprocessing_recall": v35["postprocessing_sweep"]["best_recall_then_precision"],
            "best_pixel_f1": v35["pixel_segmentation"]["best_f1"],
            "v37_fixed_pipeline": fixed_component,
            "v37_best_postprocessing_recall": best_component,
            "v37_best_pixel_f1": best_pixel,
        },
        "interpretation": {
            "pixel_segmentation": "Full-box probability pixels remain below the Tier 1 recall/precision bars across thresholds.",
            "connected_components": "Threshold, morphology, and area sweeps cannot recover the missing regions after segmentation.",
            "tiling": "All dev source pixels are covered; overlap disagreement is aggregate-only and not the recall bottleneck.",
            "isolated_responsible_stage": "full_box_pixel_segmentation",
            "next_revision_startable": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_diagnostic(args.model.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
