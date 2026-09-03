# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Separate V36 core pixels, component extraction, and expansion failures."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.real_range_classifier_finetune_v32.dataset import SceneSample, build_split
from ml.ocr.shrink_region_detector_v36.dataset import build_tiles, expand_core_box, shrink_box
from ml.ocr.shrink_region_detector_v36.protocol import (
    DB_SHRINK_RATIO,
    MINIMUM_COMPONENT_AREA,
    ONNX_PROVIDER,
    PIXEL_THRESHOLD,
    TILE_OVERLAP,
    TILE_SIZE,
    TRUTH_MATCH_IOU_MINIMUM,
)


PIXEL_THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05))
PROPOSAL_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probabilities(
    scene: SceneSample, tiles: tuple[object, ...], session: ort.InferenceSession
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    values = np.stack(
        [(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in tiles]
    ).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = np.asarray(session.run([output_name], {input_name: values})[0], dtype=np.float32)[:, 0]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    scores = np.zeros(scene.raster.shape, dtype=np.float32)
    counts = np.zeros(scene.raster.shape, dtype=np.uint8)
    valid_maps: list[np.ndarray] = []
    for tile, probability in zip(tiles, probabilities, strict=True):
        valid = probability[: tile.valid_height, : tile.valid_width]
        valid_maps.append(valid)
        scores[tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] += valid
        counts[tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] += 1
    return scores / np.maximum(counts, 1), valid_maps, counts


def _core_boxes(scene: SceneSample) -> tuple[Box, ...]:
    height, width = scene.raster.shape
    return tuple(shrink_box(truth, canvas_width=width, canvas_height=height).core for truth in scene.truths)


def _mask(boxes: Iterable[Box], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        mask[box.top : box.bottom, box.left : box.right] = True
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


def _iou(left: Box, right: Box) -> float:
    x0, y0 = max(left.left, right.left), max(left.top, right.top)
    x1, y1 = min(left.right, right.right), min(left.bottom, right.bottom)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / max(1, union)


def _matched_count(predicted: tuple[Box, ...], truths: tuple[Box, ...]) -> int:
    edges = [[j for j, truth in enumerate(truths) if _iou(candidate, truth) >= TRUTH_MATCH_IOU_MINIMUM] for candidate in predicted]
    owners = [-1] * len(truths)

    def visit(candidate_index: int, seen: set[int]) -> bool:
        for truth_index in edges[candidate_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            if owners[truth_index] == -1 or visit(owners[truth_index], seen):
                owners[truth_index] = candidate_index
                return True
        return False

    return sum(int(visit(index, set())) for index in range(len(predicted)))


def _connected_boxes(probability: np.ndarray, threshold: float) -> tuple[Box, ...]:
    binary = (probability >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes: list[Box] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= MINIMUM_COMPONENT_AREA and width >= 1 and height >= 1:
            boxes.append(Box(x, y, x + width, y + height))
    return tuple(sorted(boxes, key=lambda item: (item.top, item.left, item.bottom, item.right)))


def _proposal_metrics(
    rows: list[tuple[np.ndarray, SceneSample]], threshold: float, *, expand: bool
) -> dict[str, float | int | bool]:
    truth = matched = predicted_count = 0
    for probability, scene in rows:
        cores = _connected_boxes(probability, threshold)
        if expand:
            height, width = scene.raster.shape
            predicted = tuple(expand_core_box(core, canvas_width=width, canvas_height=height) for core in cores)
            truths = scene.truths
        else:
            predicted = cores
            truths = _core_boxes(scene)
        truth += len(truths)
        matched += _matched_count(predicted, truths)
        predicted_count += len(predicted)
    return {
        "threshold": threshold,
        "truth_regions": truth,
        "predicted_regions": predicted_count,
        "true_positives": matched,
        "false_positives": predicted_count - matched,
        "false_negatives": truth - matched,
        "precision": matched / max(1, predicted_count),
        "recall": matched / max(1, truth),
        "expanded": expand,
    }


def _oracle_metrics(scenes: tuple[SceneSample, ...]) -> dict[str, float | int]:
    truth = matched = 0
    ious: list[float] = []
    for scene in scenes:
        height, width = scene.raster.shape
        expanded = tuple(
            expand_core_box(core, canvas_width=width, canvas_height=height)
            for core in _core_boxes(scene)
        )
        truth += len(scene.truths)
        matched += _matched_count(expanded, scene.truths)
        ious.extend(_iou(candidate, target) for candidate, target in zip(expanded, scene.truths, strict=True))
    return {
        "truth_regions": truth,
        "predicted_regions": truth,
        "true_positives": matched,
        "false_positives": truth - matched,
        "false_negatives": truth - matched,
        "precision": matched / max(1, truth),
        "recall": matched / max(1, truth),
        "mean_source_box_iou": float(np.mean(ious)) if ious else 0.0,
        "minimum_source_box_iou": float(np.min(ious)) if ious else 0.0,
    }


def _best(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(rows, key=lambda row: (float(row["recall"]), float(row["precision"])))


def diagnose(checkpoint: Path, repository_root: Path) -> dict[str, object]:
    scenes = tuple(build_split("dev"))
    all_tiles = build_tiles("dev")
    session = ort.InferenceSession(str(checkpoint), providers=[ONNX_PROVIDER])
    rows: list[tuple[np.ndarray, SceneSample]] = []
    overlap_stds: list[float] = []
    coverage_counts: list[int] = []
    tile_count = 0
    for scene in scenes:
        tiles = tuple(tile for tile in all_tiles if tile.scene_id == scene.scene_id)
        probability, valid_maps, counts = _probabilities(scene, tiles, session)
        rows.append((probability, scene))
        tile_count += len(tiles)
        sum_map = np.zeros(scene.raster.shape, dtype=np.float64)
        sum_squares = np.zeros(scene.raster.shape, dtype=np.float64)
        for tile, valid in zip(tiles, valid_maps, strict=True):
            region = (slice(tile.top, tile.top + tile.valid_height), slice(tile.left, tile.left + tile.valid_width))
            sum_map[region] += valid
            sum_squares[region] += valid.astype(np.float64) ** 2
        overlap = counts > 1
        if np.any(overlap):
            means = sum_map[overlap] / counts[overlap]
            variances = np.maximum(0.0, sum_squares[overlap] / counts[overlap] - means * means)
            overlap_stds.extend(np.sqrt(variances).tolist())
        coverage_counts.extend(counts.ravel().tolist())

    targets = [_mask(_core_boxes(scene), scene.raster.shape) for scene in scenes]
    pixel_rows: list[dict[str, float | int]] = []
    for threshold in PIXEL_THRESHOLDS:
        parts = [_pixel_metrics(probability, target, threshold) for (probability, _), target in zip(rows, targets, strict=True)]
        totals = {key: sum(int(part[key]) for part in parts) for key in ("true_positive_pixels", "false_positive_pixels", "false_negative_pixels")}
        tp, fp, fn = totals.values()
        pixel_rows.append({"threshold": threshold, **totals, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2.0 * tp / max(1, 2 * tp + fp + fn)})

    core_rows = [_proposal_metrics(rows, threshold, expand=False) for threshold in PROPOSAL_THRESHOLDS]
    expanded_rows = [_proposal_metrics(rows, threshold, expand=True) for threshold in PROPOSAL_THRESHOLDS]
    fixed_core = next(row for row in core_rows if row["threshold"] == PIXEL_THRESHOLD)
    fixed_expanded = next(row for row in expanded_rows if row["threshold"] == PIXEL_THRESHOLD)
    best_pixel = max(pixel_rows, key=lambda row: float(row["f1"]))
    best_pixel_recall = max(pixel_rows, key=lambda row: (float(row["recall"]), float(row["precision"])))
    best_core = _best(core_rows)
    best_expanded = _best(expanded_rows)
    oracle = _oracle_metrics(scenes)

    if float(best_pixel_recall["recall"]) < 0.95:
        responsible_stage = "core_pixel_segmentation"
    elif float(best_core["recall"]) < 0.95:
        responsible_stage = "core_component_extraction"
    elif float(best_expanded["recall"]) < 0.95 and float(oracle["recall"]) >= 0.95:
        responsible_stage = "core_expansion"
    else:
        responsible_stage = "none"

    source_paths = {
        "v32_data_module": repository_root / "ml/ocr/real_range_classifier_finetune_v32/dataset.py",
        "v36_dataset": repository_root / "ml/ocr/shrink_region_detector_v36/dataset.py",
        "v36_pipeline": repository_root / "ml/ocr/shrink_region_detector_v36/pipeline.py",
        "v36_protocol": repository_root / "ml/ocr/shrink_region_detector_v36/protocol.py",
        "v36_result": repository_root / "ml/ocr/shrink_region_detector_v36/P1_RESULT.json",
    }
    hashes = {key: {"path": path.relative_to(repository_root).as_posix(), "sha256": _sha256(path)} for key, path in source_paths.items()}
    hashes["checkpoint_onnx"] = {"path": checkpoint.as_posix(), "sha256": _sha256(checkpoint)}
    pt_checkpoint = checkpoint.with_suffix(".pt")
    if pt_checkpoint.exists():
        hashes["checkpoint_pt"] = {"path": pt_checkpoint.as_posix(), "sha256": _sha256(pt_checkpoint)}

    return {
        "schema": "graphreader.ocr-shrink-region-detector-v36-diagnostic.v1",
        "revision": "graph-text-shrink-region-detector-v36",
        "evidence": {
            "split": "dev",
            "scene_count": len(scenes),
            "truth_region_count": sum(len(scene.truths) for scene in scenes),
            "synthetic_only": True,
            "private_or_article_images": False,
            "sealed_or_public_reads": 0,
            "case_level_output": False,
        },
        "fixed_hashes": hashes,
        "protocol": {
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "db_shrink_ratio": DB_SHRINK_RATIO,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "onnx_provider": ONNX_PROVIDER,
        },
        "baseline_fixed_pipeline": {
            "threshold": PIXEL_THRESHOLD,
            "predicted_core": fixed_core,
            "expanded_proposal": fixed_expanded,
        },
        "core_pixel_segmentation": {"threshold_sweep": pixel_rows, "best_f1": best_pixel, "best_recall_then_precision": best_pixel_recall},
        "predicted_core_proposals": {"threshold_sweep": core_rows, "best_recall_then_precision": best_core},
        "expanded_proposals": {"threshold_sweep": expanded_rows, "best_recall_then_precision": best_expanded},
        "ground_truth_core_expansion_oracle": oracle,
        "tiling_overlap_mapping": {
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "tile_count": tile_count,
            "covered_pixel_fraction": sum(value >= 1 for value in coverage_counts) / max(1, len(coverage_counts)),
            "minimum_coverage": min(coverage_counts),
            "maximum_coverage": max(coverage_counts),
            "overlap_pixel_count": len(overlap_stds),
            "mean_overlap_prediction_std": float(np.mean(overlap_stds)) if overlap_stds else 0.0,
            "p95_overlap_prediction_std": float(np.percentile(overlap_stds, 95)) if overlap_stds else 0.0,
            "fraction_overlap_std_gt_0_1": float(np.mean(np.asarray(overlap_stds) > 0.1)) if overlap_stds else 0.0,
            "maximum_overlap_prediction_std": float(np.max(overlap_stds)) if overlap_stds else 0.0,
        },
        "isolated_responsible_stage": responsible_stage,
        "next_revision_startable": responsible_stage != "none",
        "interpretation": {
            "core_pixel_segmentation": "Core probability separation remains below the 0.95 recall bar." if float(best_pixel_recall["recall"]) < 0.95 else "Core probability separation clears the 0.95 recall bar.",
            "core_component_extraction": "Connected core extraction remains below the 0.95 proposal recall bar after pixel thresholding." if float(best_core["recall"]) < 0.95 else "Connected core extraction clears the 0.95 proposal recall bar.",
            "core_expansion": "Ground-truth-core expansion passes perfectly, so deterministic expansion is not the responsible stage." if float(oracle["recall"]) >= 0.95 else "Expansion geometry remains below the 0.95 oracle bar.",
            "tiling": "Source coverage is complete; overlap mapping is not the recall bottleneck." if min(coverage_counts) >= 1 else "Source coverage is incomplete and must be repaired before model attribution.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    report = diagnose(args.checkpoint, args.repository_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": "ok", "isolated_responsible_stage": report["isolated_responsible_stage"], "next_revision_startable": report["next_revision_startable"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
