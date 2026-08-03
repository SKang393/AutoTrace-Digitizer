# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Maximum-cardinality one-to-one center matching and scene aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .postprocess import Detection


@dataclass(frozen=True)
class CenterMetrics:
    tolerance_px: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    duplicate_count: int
    duplicate_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _maximum_matching(
    predicted: tuple[Detection, ...],
    actual: tuple[tuple[float, float], ...],
    tolerance_px: float,
) -> dict[int, int]:
    adjacency: list[list[int]] = []
    for item in predicted:
        candidates = [
            (math.hypot(item.x - x, item.y - y), truth_index)
            for truth_index, (x, y) in enumerate(actual)
            if math.hypot(item.x - x, item.y - y) <= tolerance_px
        ]
        adjacency.append([index for _, index in sorted(candidates)])
    truth_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or augment(previous, visited):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    order = sorted(range(len(predicted)), key=lambda index: (-predicted[index].confidence, predicted[index].y, predicted[index].x))
    for prediction_index in order:
        augment(prediction_index, set())
    return truth_to_prediction


def center_metrics(
    predictions: Iterable[Detection],
    truth: Iterable[tuple[float, float]],
    tolerance_px: float,
) -> CenterMetrics:
    predicted = tuple(predictions)
    actual = tuple(truth)
    matching = _maximum_matching(predicted, actual, tolerance_px)
    matched_predictions = set(matching.values())
    true_positives = len(matching)
    false_positives = len(predicted) - true_positives
    false_negatives = len(actual) - true_positives
    precision = true_positives / len(predicted) if predicted else (1.0 if not actual else 0.0)
    recall = true_positives / len(actual) if actual else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    duplicate_count = sum(
        1
        for index, item in enumerate(predicted)
        if index not in matched_predictions
        and any(math.hypot(item.x - x, item.y - y) <= tolerance_px for x, y in actual)
    )
    return CenterMetrics(
        tolerance_px,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
        duplicate_count,
        duplicate_count / len(actual) if actual else 0.0,
    )


def aggregate_scene_metrics(metrics: Iterable[CenterMetrics], tolerance_px: float) -> CenterMetrics:
    values = tuple(metrics)
    true_positives = sum(item.true_positives for item in values)
    false_positives = sum(item.false_positives for item in values)
    false_negatives = sum(item.false_negatives for item in values)
    duplicate_count = sum(item.duplicate_count for item in values)
    prediction_count = true_positives + false_positives
    truth_count = true_positives + false_negatives
    precision = true_positives / prediction_count if prediction_count else (1.0 if not truth_count else 0.0)
    recall = true_positives / truth_count if truth_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CenterMetrics(
        tolerance_px,
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
        duplicate_count,
        duplicate_count / truth_count if truth_count else 0.0,
    )


__all__ = ["CenterMetrics", "aggregate_scene_metrics", "center_metrics"]
