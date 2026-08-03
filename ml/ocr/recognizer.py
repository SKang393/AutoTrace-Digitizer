# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""A tiny trainable prototype recognizer used as a transparent baseline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import inf
from typing import Iterable

from .synthetic import Raster, SyntheticLabelSample

_NORMALIZED_WIDTH = 5
_NORMALIZED_HEIGHT = 7


@dataclass(frozen=True)
class TrainingSummary:
    samples_seen: int
    labels_used: int
    labels_skipped: int
    prototype_counts: dict[str, int]


def _rotate_counterclockwise(raster: Raster) -> Raster:
    return tuple(tuple(row) for row in zip(*raster, strict=True))[::-1]


def _upright(raster: Raster, orientation_degrees: int) -> Raster:
    result = raster
    for _ in range((orientation_degrees % 360) // 90):
        result = _rotate_counterclockwise(result)
    return result


def _binary(raster: Raster) -> list[list[int]]:
    values = [value for row in raster for value in row]
    threshold = (min(values) + max(values)) / 2
    return [[int(value <= threshold) for value in row] for row in raster]


def _segments(raster: Raster, orientation_degrees: int) -> list[list[list[int]]]:
    pixels = _binary(_upright(raster, orientation_degrees))
    active_columns = [any(row[column] for row in pixels) for column in range(len(pixels[0]))]
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for column, active in enumerate(active_columns + [False]):
        if active and start is None:
            start = column
        elif not active and start is not None:
            spans.append((start, column))
            start = None

    results: list[list[list[int]]] = []
    for left, right in spans:
        active_rows = [
            row_index
            for row_index, row in enumerate(pixels)
            if any(row[left:right])
        ]
        if not active_rows:
            continue
        top, bottom = min(active_rows), max(active_rows) + 1
        results.append([row[left:right] for row in pixels[top:bottom]])
    return results


def _normalize(segment: list[list[int]]) -> tuple[float, ...]:
    source_height = len(segment)
    source_width = len(segment[0])
    result = []
    for target_y in range(_NORMALIZED_HEIGHT):
        source_y = round(target_y * (source_height - 1) / (_NORMALIZED_HEIGHT - 1))
        for target_x in range(_NORMALIZED_WIDTH):
            source_x = round(target_x * (source_width - 1) / (_NORMALIZED_WIDTH - 1))
            result.append(float(segment[source_y][source_x]))
    # Geometry preserves punctuation distinctions that disappear when a dot
    # and a dash are both resized to the same fixed rectangle.
    result.extend(
        (
            2.0 * source_width / source_height,
            2.0 * source_height / source_width,
            2.0 * sum(sum(row) for row in segment) / (source_width * source_height),
        )
    )
    return tuple(result)


class PrototypeRecognizer:
    """Nearest-centroid character recognizer with no runtime dependency."""

    def __init__(self) -> None:
        self._centroids: dict[str, tuple[float, ...]] = {}

    @property
    def trained(self) -> bool:
        return bool(self._centroids)

    def fit(self, samples: Iterable[SyntheticLabelSample]) -> TrainingSummary:
        vectors: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        samples_seen = labels_used = labels_skipped = 0
        for sample in samples:
            samples_seen += 1
            segments = _segments(sample.raster, sample.orientation_degrees)
            if len(segments) != len(sample.target_text):
                labels_skipped += 1
                continue
            labels_used += 1
            for character, segment in zip(sample.target_text, segments, strict=True):
                vectors[character].append(_normalize(segment))

        self._centroids = {}
        for character, character_vectors in vectors.items():
            vector_length = len(character_vectors[0])
            self._centroids[character] = tuple(
                sum(vector[index] for vector in character_vectors) / len(character_vectors)
                for index in range(vector_length)
            )
        return TrainingSummary(
            samples_seen=samples_seen,
            labels_used=labels_used,
            labels_skipped=labels_skipped,
            prototype_counts={key: len(value) for key, value in sorted(vectors.items())},
        )

    def predict(self, raster: Raster, orientation_degrees: int = 0) -> str:
        if not self._centroids:
            raise RuntimeError("Recognizer must be fitted before prediction")
        result = []
        for segment in _segments(raster, orientation_degrees):
            vector = _normalize(segment)
            best_character = ""
            best_distance = inf
            for character, centroid in sorted(self._centroids.items()):
                distance = sum((left - right) ** 2 for left, right in zip(vector, centroid, strict=True))
                if distance < best_distance:
                    best_character = character
                    best_distance = distance
            result.append(best_character)
        return "".join(result)


class ConstantBaseline:
    """An intentionally weak baseline that makes improvement measurable."""

    def __init__(self, value: str = "0") -> None:
        self._value = value

    def predict(self, raster: Raster, orientation_degrees: int = 0) -> str:
        del raster, orientation_degrees
        return self._value
