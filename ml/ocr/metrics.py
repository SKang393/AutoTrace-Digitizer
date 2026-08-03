# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Exact-match and character-error metrics for OCR benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RecognitionMetrics:
    sample_count: int
    exact_count: int
    exact_match: float
    character_errors: int
    reference_characters: int
    character_error_rate: float


def edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for row_index, reference_character in enumerate(reference, start=1):
        current = [row_index]
        for column_index, prediction_character in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + int(reference_character != prediction_character),
                )
            )
        previous = current
    return previous[-1]


def evaluate_predictions(pairs: Iterable[tuple[str, str]]) -> RecognitionMetrics:
    materialized = tuple(pairs)
    if not materialized:
        raise ValueError("At least one reference/prediction pair is required")
    exact_count = sum(reference == prediction for reference, prediction in materialized)
    errors = sum(edit_distance(reference, prediction) for reference, prediction in materialized)
    reference_characters = sum(len(reference) for reference, _ in materialized)
    if reference_characters == 0:
        raise ValueError("At least one reference character is required")
    return RecognitionMetrics(
        sample_count=len(materialized),
        exact_count=exact_count,
        exact_match=exact_count / len(materialized),
        character_errors=errors,
        reference_characters=reference_characters,
        character_error_rate=errors / reference_characters,
    )
