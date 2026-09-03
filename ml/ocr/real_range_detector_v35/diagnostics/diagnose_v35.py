# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Explain V35 proposal failure using aggregate synthetic-dev evidence only."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ml.ocr.real_range_classifier_finetune_v32.dataset import build_split
from ml.ocr.real_range_detector_v35.dataset import build_tiles
from ml.ocr.real_range_detector_v35.protocol import MINIMUM_COMPONENT_AREA, PIXEL_THRESHOLD, TILE_SIZE


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probabilities(scene, tiles, session: ort.InferenceSession) -> tuple[np.ndarray, list[np.ndarray]]:
    values = np.stack(
        [(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in tiles]
    ).astype(np.float32)
    logits = np.asarray(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: values})[0], dtype=np.float32)[:, 0]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    scores = np.zeros(scene.raster.shape, dtype=np.float32)
    counts = np.zeros(scene.raster.shape, dtype=np.float32)
    valid_maps: list[np.ndarray] = []
    for tile, probability in zip(tiles, probabilities, strict=True):
        valid = probability[: tile.valid_height, : tile.valid_width]
        valid_maps.append(valid)
        scores[tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] += valid
        counts[tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] += 1.0
    return scores / np.maximum(counts, 1.0), valid_maps


def _truth_mask(scene) -> np.ndarray:
    mask = np.zeros(scene.raster.shape, dtype=bool)
    for truth in scene.truths:
        mask[int(truth.top) : int(truth.bottom), int(truth.left) : int(truth.right)] = True
    return mask


def _pixel_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probability >= threshold
    tp = int(np.count_nonzero(predicted & target))
    fp = int(np.count_nonzero(predicted & ~target))
    fn = int(np.count_nonzero(~predicted & target))
    return {
        "threshold": threshold,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2.0 * tp / max(1, 2 * tp + fp + fn),
    }


def _boxes(probability: np.ndarray, threshold: float, minimum_area: int, close: bool) -> tuple[tuple[int, int, int, int], ...]:
    binary = (probability >= threshold).astype(np.uint8)
    if close:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= minimum_area and width >= 2 and height >= 2:
            result.append((x, y, x + width, y + height))
    return tuple(result)


def _iou(left: tuple[int, int, int, int], right) -> float:
    x0, y0 = max(left[0], right.left), max(left[1], right.top)
    x1, y1 = min(left[2], right.right), min(left[3], right.bottom)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (left[2] - left[0]) * (left[3] - left[1]) + right.width * right.height - intersection
    return intersection / max(1, union)


def _matched_count(predicted, truths) -> int:
    edges = [[j for j, truth in enumerate(truths) if _iou(candidate, truth) >= 0.5] for candidate in predicted]
    owners = [-1] * len(truths)

    def visit(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if owners[j] == -1 or visit(owners[j], seen):
                owners[j] = i
                return True
        return False

    return sum(int(visit(i, set())) for i in range(len(predicted)))


def _proposal_metrics(rows: list[tuple[tuple[tuple[int, int, int, int], ...], object]], threshold: float, minimum_area: int, close: bool) -> dict[str, float | int | bool]:
    truth = matched = predicted = 0
    for _, scene in rows:
        boxes = _boxes(_, threshold, minimum_area, close)
        truth += len(scene.truths)
        matched += _matched_count(boxes, scene.truths)
        predicted += len(boxes)
    return {
        "threshold": threshold,
        "minimum_component_area": minimum_area,
        "morphology_close": close,
        "truth_regions": truth,
        "predicted_regions": predicted,
        "true_positives": matched,
        "false_positives": predicted - matched,
        "false_negatives": truth - matched,
        "precision": matched / max(1, predicted),
        "recall": matched / max(1, truth),
    }


def diagnose(checkpoint: Path) -> dict[str, object]:
    scenes = build_split("dev")
    all_tiles = build_tiles("dev")
    session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
    rows: list[tuple[np.ndarray, object]] = []
    overlap_values: list[float] = []
    coverage_counts: list[int] = []
    tile_count = 0
    for scene in scenes:
        tiles = tuple(tile for tile in all_tiles if tile.scene_id == scene.scene_id)
        probability, valid_maps = _probabilities(scene, tiles, session)
        rows.append((probability, scene))
        tile_count += len(tiles)
        count_map = np.zeros(scene.raster.shape, dtype=np.uint8)
        stack = np.zeros((len(tiles), *scene.raster.shape), dtype=np.float32)
        for index, (tile, valid) in enumerate(zip(tiles, valid_maps, strict=True)):
            count_map[tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] += 1
            stack[index, tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] = valid
        overlap = count_map > 1
        if np.any(overlap):
            overlap_values.extend(np.std(stack[:, overlap], axis=0).tolist())
        coverage_counts.extend(count_map.ravel().tolist())

    targets = [_truth_mask(scene) for _, scene in rows]
    pixel_rows = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        parts = [_pixel_metrics(probability, target, round(float(threshold), 2)) for (probability, _), target in zip(rows, targets, strict=True)]
        totals = {key: sum(int(part[key]) for part in parts) for key in ("true_positive_pixels", "false_positive_pixels", "false_negative_pixels")}
        tp, fp, fn = totals.values()
        pixel_rows.append({"threshold": round(float(threshold), 2), **totals, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2.0 * tp / max(1, 2 * tp + fp + fn)})

    proposal_rows = []
    for threshold in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        for close in (False, True):
            proposal_rows.append(_proposal_metrics(rows, threshold, MINIMUM_COMPONENT_AREA, close))
    for minimum_area in (0, 2, 4, 8, 16, 32):
        proposal_rows.append(_proposal_metrics(rows, PIXEL_THRESHOLD, minimum_area, True))

    baseline = next(row for row in proposal_rows if row["threshold"] == PIXEL_THRESHOLD and row["minimum_component_area"] == MINIMUM_COMPONENT_AREA and row["morphology_close"] is True)
    best = max(proposal_rows, key=lambda row: (float(row["recall"]), float(row["precision"])))
    best_pixel = max(pixel_rows, key=lambda row: float(row["f1"]))
    diagnosis = "pixel_segmentation" if float(best["recall"]) < 0.80 else "threshold_behavior"
    if float(best["recall"]) >= 0.80 and baseline["recall"] < 0.80:
        diagnosis = "threshold_behavior"
    return {
        "schema": "graphreader.ocr-real-range-detector-v35-diagnostic.v1",
        "revision": "graph-text-real-range-detector-v35",
        "evidence": {"split": "dev", "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes), "synthetic_only": True, "private_or_article_images": False, "sealed_or_public_reads": 0},
        "checkpoint": {"artifact": checkpoint.name, "sha256": _sha256(checkpoint), "provider": "CPUExecutionProvider"},
        "baseline_fixed_pipeline": baseline,
        "pixel_segmentation": {"threshold_sweep": pixel_rows, "best_f1": best_pixel},
        "postprocessing_sweep": {"rows": proposal_rows, "best_recall_then_precision": best},
        "tiling_overlap_mapping": {"tile_size": TILE_SIZE, "tile_count": tile_count, "covered_pixel_fraction": sum(value >= 1 for value in coverage_counts) / max(1, len(coverage_counts)), "minimum_coverage": min(coverage_counts), "maximum_coverage": max(coverage_counts), "overlap_pixel_count": len(overlap_values), "mean_overlap_prediction_std": float(np.mean(overlap_values)) if overlap_values else 0.0, "p95_overlap_prediction_std": float(np.percentile(overlap_values, 95)) if overlap_values else 0.0, "fraction_overlap_std_gt_0_1": float(np.mean(np.asarray(overlap_values) > 0.1)) if overlap_values else 0.0, "maximum_overlap_prediction_std": float(np.max(overlap_values)) if overlap_values else 0.0},
        "isolated_responsible_stage": diagnosis,
        "interpretation": "Threshold/postprocessing changes cannot recover the missing regions when the best aggregate proposal recall remains low; segmentation is the primary V35 failure." if diagnosis == "pixel_segmentation" else "The proposal ceiling is materially threshold-sensitive; adjust threshold/postprocessing before changing the model.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "isolated_responsible_stage": report["isolated_responsible_stage"], "baseline": report["baseline_fixed_pipeline"], "best": report["postprocessing_sweep"]["best_recall_then_precision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
