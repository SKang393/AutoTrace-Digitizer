# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Tile merge, postprocessing, and aggregate raw-proposal metrics."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.real_range_classifier_finetune_v32.dataset import SceneSample

from .dataset import TileSample, build_tiles
from .protocol import MINIMUM_COMPONENT_AREA, PIXEL_THRESHOLD, TILE_SIZE


def tile_to_source_box(tile: TileSample, x: int, y: int, width: int, height: int) -> Box:
    """Map a tile-local connected-component box to original pixels exactly."""
    return Box(
        tile.left + x,
        tile.top + y,
        min(tile.left + x + width, tile.left + tile.valid_width),
        min(tile.top + y + height, tile.top + tile.valid_height),
    )


def reconstruct_probability_map(scene: SceneSample, tiles: tuple[TileSample, ...], runner: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    values = np.stack([(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in tiles]).astype(np.float32)
    logits = np.asarray(runner(values), dtype=np.float32)[:, 0]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    scores = np.zeros(scene.raster.shape, dtype=np.float32)
    counts = np.zeros(scene.raster.shape, dtype=np.float32)
    for tile, probability in zip(tiles, probabilities, strict=True):
        scores[tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] += probability[: tile.valid_height, : tile.valid_width]
        counts[tile.top : tile.top + tile.valid_height, tile.left : tile.left + tile.valid_width] += 1.0
    return scores / np.maximum(counts, 1.0)


def postprocess_probability_map(probability: np.ndarray, tiles: tuple[TileSample, ...]) -> tuple[Box, ...]:
    binary = (probability >= PIXEL_THRESHOLD).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes: list[Box] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < MINIMUM_COMPONENT_AREA or width < 2 or height < 2:
            continue
        # The probability map is already in source coordinates. A synthetic
        # tile carrying this component is selected only for the reversible map.
        containing = next((tile for tile in tiles if tile.left <= x and tile.top <= y and x + width <= tile.left + tile.valid_width and y + height <= tile.top + tile.valid_height), None)
        if containing is None:
            boxes.append(Box(x, y, x + width, y + height))
        else:
            boxes.append(tile_to_source_box(containing, x - containing.left, y - containing.top, width, height))
    return tuple(sorted(boxes, key=lambda item: (item.top, item.left, item.bottom, item.right)))


def maximum_cardinality_matches(predicted: tuple[Box, ...], truths: tuple[Box, ...]) -> int:
    def iou(left: Box, right: Box) -> float:
        x0, y0 = max(left.left, right.left), max(left.top, right.top)
        x1, y1 = min(left.right, right.right), min(left.bottom, right.bottom)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        union = left.width * left.height + right.width * right.height - intersection
        return intersection / max(1, union)

    edges = [[index for index, truth in enumerate(truths) if iou(candidate, truth) >= 0.5] for candidate in predicted]
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


def evaluate_scenes(scenes: tuple[SceneSample, ...], runner: Callable[[np.ndarray], np.ndarray]) -> dict[str, object]:
    truth = true_positive = false_positive = false_negative = 0
    calls = 0
    by_dimension: dict[str, list[int]] = {}
    for scene in scenes:
        tiles = build_tiles("dev" if scene.split == "dev" else "train")
        scene_tiles = tuple(tile for tile in tiles if tile.scene_id == scene.scene_id)
        predicted = postprocess_probability_map(reconstruct_probability_map(scene, scene_tiles, runner), scene_tiles)
        matched = maximum_cardinality_matches(predicted, scene.truths)
        current_truth = len(scene.truths)
        current_fp = len(predicted) - matched
        truth += current_truth; true_positive += matched; false_positive += current_fp; false_negative += current_truth - matched
        calls += 1
        row = by_dimension.setdefault(f"{scene.raster.shape[1]}x{scene.raster.shape[0]}", [0, 0, 0, 0])
        row[0] += current_truth; row[1] += matched; row[2] += current_fp; row[3] += current_truth - matched
    return {
        "scene_count": len(scenes), "truth_region_count": truth,
        "true_positives": true_positive, "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, truth), "inference_calls": calls,
        "by_dimension": {key: {"truth_region_count": row[0], "true_positives": row[1], "false_positives": row[2], "false_negatives": row[3]} for key, row in sorted(by_dimension.items())},
    }
