# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic public OCR gate metrics from sealed truth and predictions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class OcrGateMetrics:
    exact_match: float
    character_error_rate: float
    role_accuracy: float


def _edit_distance(expected: str, actual: str) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_character in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_character in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1]
                    + (expected_character != actual_character),
                )
            )
        previous = current
    return previous[-1]


def evaluate_partition(
    records: Iterable[tuple[str, str, str, str]],
) -> OcrGateMetrics:
    """Evaluate (truth text, predicted text, truth role, predicted role)."""

    rows = tuple(records)
    if not rows:
        raise ValueError("At least one OCR record is required.")
    exact = sum(truth == prediction for truth, prediction, _, _ in rows)
    role_exact = sum(truth_role == predicted_role for _, _, truth_role, predicted_role in rows)
    truth_characters = sum(len(truth) for truth, _, _, _ in rows)
    if truth_characters == 0:
        raise ValueError("OCR truth must contain at least one character.")
    edits = sum(_edit_distance(truth, prediction) for truth, prediction, _, _ in rows)
    return OcrGateMetrics(
        exact_match=exact / len(rows),
        character_error_rate=edits / truth_characters,
        role_accuracy=role_exact / len(rows),
    )


def maximum_absolute_error(pairs: Iterable[tuple[float, float]]) -> float:
    values = tuple(pairs)
    if not values:
        raise ValueError("At least one parity pair is required.")
    return max(abs(reference - onnx) for reference, onnx in values)
