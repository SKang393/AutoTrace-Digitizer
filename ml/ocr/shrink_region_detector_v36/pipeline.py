# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""V36 tiled map reconstruction and fixed DB core expansion."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.real_range_classifier_finetune_v32.dataset import SceneSample

from .dataset import TileSample, build_tiles, expand_core_box
from .protocol import MINIMUM_COMPONENT_AREA, PIXEL_THRESHOLD


def reconstruct_probability_map(
    scene: SceneSample,
    tiles: tuple[TileSample, ...],
    runner: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
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
    """Extract separated DB cores, then expand each once into source pixels."""
    if probability.ndim != 2:
        raise ValueError("V36 probability map must be two-dimensional")
    binary = (probability >= PIXEL_THRESHOLD).astype(np.uint8)
    # Do not close or dilate. Adjacent text boxes must remain separate cores.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes: list[Box] = []
    height, width = probability.shape
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if area < MINIMUM_COMPONENT_AREA or box_width < 1 or box_height < 1:
            continue
        core = Box(x, y, x + box_width, y + box_height)
        boxes.append(expand_core_box(core, canvas_width=width, canvas_height=height))
    return tuple(sorted(set(boxes), key=lambda item: (item.top, item.left, item.bottom, item.right)))


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
        all_tiles = build_tiles("dev" if scene.split == "dev" else "train")
        scene_tiles = tuple(tile for tile in all_tiles if tile.scene_id == scene.scene_id)
        probability = reconstruct_probability_map(scene, scene_tiles, runner)
        predicted = postprocess_probability_map(probability, scene_tiles)
        matched = maximum_cardinality_matches(predicted, scene.truths)
        current_truth = len(scene.truths)
        current_fp = len(predicted) - matched
        truth += current_truth
        true_positive += matched
        false_positive += current_fp
        false_negative += current_truth - matched
        calls += 1
        row = by_dimension.setdefault(f"{scene.raster.shape[1]}x{scene.raster.shape[0]}", [0, 0, 0, 0])
        row[0] += current_truth
        row[1] += matched
        row[2] += current_fp
        row[3] += current_truth - matched
    return {
        "scene_count": len(scenes),
        "truth_region_count": truth,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, truth),
        "inference_calls": calls,
        "by_dimension": {
            key: {
                "truth_region_count": row[0],
                "true_positives": row[1],
                "false_positives": row[2],
                "false_negatives": row[3],
            }
            for key, row in sorted(by_dimension.items())
        },
    }


__all__ = [
    "evaluate_scenes", "maximum_cardinality_matches", "postprocess_probability_map",
    "reconstruct_probability_map",
]
