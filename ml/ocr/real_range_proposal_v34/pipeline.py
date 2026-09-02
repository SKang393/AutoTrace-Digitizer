# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate raw-proposal metrics using maximum-cardinality IoU matching."""

from __future__ import annotations

from typing import Callable

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.component_context_detector_v7.dataset import Component, box_iou

from .dataset import SceneSample


def maximum_cardinality_matches(predicted: tuple[Component, ...], truths: tuple[Box, ...]) -> int:
    edges = [[index for index, truth in enumerate(truths) if box_iou(item.box, truth) >= 0.5] for item in predicted]
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


def score_proposals(scenes: tuple[SceneSample, ...], proposal_builder: Callable[..., tuple[Component, ...]]) -> dict[str, object]:
    truth_count = true_positives = false_positives = false_negatives = 0
    dimensions: dict[str, list[int]] = {}
    for scene in scenes:
        candidates = proposal_builder(scene.raster)
        matched = maximum_cardinality_matches(candidates, scene.truths)
        current_truth = len(scene.truths)
        current_fp = len(candidates) - matched
        current_fn = current_truth - matched
        truth_count += current_truth
        true_positives += matched
        false_positives += current_fp
        false_negatives += current_fn
        row = dimensions.setdefault(f"{scene.raster.shape[1]}x{scene.raster.shape[0]}", [0, 0, 0, 0])
        row[0] += current_truth; row[1] += matched; row[2] += current_fp; row[3] += current_fn
    return {
        "scene_count": len(scenes),
        "truth_region_count": truth_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / max(1, true_positives + false_positives),
        "recall": true_positives / max(1, truth_count),
        "by_dimension": {
            key: {"truth_region_count": row[0], "true_positives": row[1], "false_positives": row[2], "false_negatives": row[3]}
            for key, row in sorted(dimensions.items())
        },
    }
