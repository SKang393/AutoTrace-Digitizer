# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Chart-fidelity metrics for the local GraphSR benchmark.

The metrics deliberately measure graph structure rather than perceptual
quality.  Inputs are high-resolution grayscale or RGB arrays with values in
either ``[0, 1]`` or ``[0, 255]``.  Annotation coordinates are expressed in
the high-resolution image's pixel space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


METRIC_NAMES = (
    "numeric_ocr_exact_match",
    "numeric_ocr_proxy_exact_match",
    "marker_center_f1",
    "shape_fill_classification_f1",
    "marker_center_mean_error_pixels",
    "axis_thin_line_recall",
    "axis_line_localization_error_pixels",
    "open_marker_preservation_rate",
    "hallucinated_structure_rate",
    "runtime_ms_mean",
    "peak_memory_bytes",
)


@dataclass(frozen=True)
class QualityMetrics:
    """Downstream image metrics before runtime observations are attached."""

    numeric_ocr_exact_match: float | None
    numeric_ocr_proxy_exact_match: float | None
    marker_center_f1: float | None
    shape_fill_classification_f1: float | None
    marker_center_mean_error_pixels: float | None
    axis_thin_line_recall: float | None
    axis_line_localization_error_pixels: float | None
    open_marker_preservation_rate: float | None
    hallucinated_structure_rate: float

    def to_dict(self) -> dict[str, float | None]:
        return {
            "numeric_ocr_exact_match": self.numeric_ocr_exact_match,
            "numeric_ocr_proxy_exact_match": self.numeric_ocr_proxy_exact_match,
            "marker_center_f1": self.marker_center_f1,
            "shape_fill_classification_f1": self.shape_fill_classification_f1,
            "marker_center_mean_error_pixels": self.marker_center_mean_error_pixels,
            "axis_thin_line_recall": self.axis_thin_line_recall,
            "axis_line_localization_error_pixels": self.axis_line_localization_error_pixels,
            "open_marker_preservation_rate": self.open_marker_preservation_rate,
            "hallucinated_structure_rate": self.hallucinated_structure_rate,
        }


def _grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] in (3, 4):
            rgb = array[:, :, :3].astype(np.float64, copy=False)
            array = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        else:
            raise ValueError("Images must have one, three, or four channels")
    if array.ndim != 2 or not array.size:
        raise ValueError("Images must be nonempty two-dimensional rasters")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Images must contain only finite samples")
    maximum = float(result.max(initial=0.0))
    minimum = float(result.min(initial=0.0))
    if minimum < 0:
        raise ValueError("Images must not contain negative samples")
    if maximum > 1.0:
        divisor = 65535.0 if maximum > 255.0 else 255.0
        result = result / divisor
    return np.clip(result, 0.0, 1.0)


def _ink(image: np.ndarray, threshold: float = 0.72) -> np.ndarray:
    if not 0.0 < threshold < 1.0:
        raise ValueError("Ink threshold must be strictly between zero and one")
    return _grayscale(image) < threshold


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius < 0:
        raise ValueError("Dilation radius must be nonnegative")
    result = np.asarray(mask, dtype=bool)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(
            tuple(
                padded[y : y + mask.shape[0], x : x + mask.shape[1]]
                for y in range(3)
                for x in range(3)
            )
        )
    return result


def _bounded_box(box: Sequence[float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError("An OCR box must contain four finite values")
    x, y, width, height = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("An OCR box must have positive width and height")
    image_height, image_width = shape
    left = max(0, min(image_width, math.floor(x)))
    top = max(0, min(image_height, math.floor(y)))
    right = max(left, min(image_width, math.ceil(x + width)))
    bottom = max(top, min(image_height, math.ceil(y + height)))
    if left == right or top == bottom:
        raise ValueError("An OCR box must overlap the image")
    return left, top, right, bottom


def _foreground_f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_count = int(reference.sum())
    candidate_count = int(candidate.sum())
    if reference_count == 0:
        return 1.0 if candidate_count == 0 else 0.0
    matched_reference = int(np.logical_and(reference, _dilate(candidate)).sum())
    matched_candidate = int(np.logical_and(candidate, _dilate(reference)).sum())
    recall = matched_reference / reference_count
    precision = matched_candidate / candidate_count if candidate_count else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def numeric_ocr_proxy_exact_match(
    reference: np.ndarray,
    candidate: np.ndarray,
    regions: Iterable[Sequence[float]],
    *,
    foreground_f1_threshold: float = 0.90,
) -> float | None:
    """Return the fraction of numeric regions retaining matching ink topology.

    This is intentionally called a proxy.  It does not claim text recognition;
    a downstream OCR engine must supply that evidence for a production default.
    """

    if not 0.0 <= foreground_f1_threshold <= 1.0:
        raise ValueError("OCR proxy threshold must be in [0, 1]")
    reference_ink = _ink(reference)
    candidate_ink = _ink(candidate)
    if reference_ink.shape != candidate_ink.shape:
        raise ValueError("Reference and candidate images must have identical dimensions")
    boxes = tuple(regions)
    if not boxes:
        return None
    matches = 0
    for box in boxes:
        left, top, right, bottom = _bounded_box(box, reference_ink.shape)
        score = _foreground_f1(
            reference_ink[top:bottom, left:right],
            candidate_ink[top:bottom, left:right],
        )
        matches += score >= foreground_f1_threshold
    return matches / len(boxes)


def _marker(marker: object) -> tuple[float, float, float]:
    if isinstance(marker, Mapping):
        center = marker.get("center")
        radius = marker.get("radius", 4.0)
    else:
        center = marker
        radius = 4.0
    if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) != 2:
        raise ValueError("A marker must provide a two-value center")
    x, y, radius_value = float(center[0]), float(center[1]), float(radius)
    if not all(math.isfinite(value) for value in (x, y, radius_value)) or radius_value <= 0:
        raise ValueError("Marker center and radius must be finite, with positive radius")
    return x, y, radius_value


def _estimated_ink_center(
    image: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float] | None:
    gray = _grayscale(image)
    x, y = center
    extent = max(2.0, radius * 1.5)
    left = max(0, math.floor(x - extent))
    right = min(gray.shape[1], math.ceil(x + extent + 1))
    top = max(0, math.floor(y - extent))
    bottom = min(gray.shape[0], math.ceil(y + extent + 1))
    if left >= right or top >= bottom:
        return None
    yy, xx = np.mgrid[top:bottom, left:right]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= extent**2
    weights = np.where(disk, np.maximum(0.0, 0.80 - gray[top:bottom, left:right]), 0.0)
    total = float(weights.sum())
    if total <= 1e-9:
        return None
    return float((xx * weights).sum() / total), float((yy * weights).sum() / total)


def marker_center_metrics(
    candidate: np.ndarray,
    markers: Iterable[object],
    *,
    predicted_markers: Iterable[object] | None = None,
    tolerance_pixels: float = 1.0,
) -> tuple[float | None, float | None]:
    """Return detector F1, when supplied, and local mean displacement.

    The image-only local estimator can measure displacement around labelled
    centers, but it cannot observe false marker detections. A mathematically
    valid F1 is therefore emitted only when an independent detector supplies
    its predicted centers.
    """

    if not math.isfinite(tolerance_pixels) or tolerance_pixels <= 0:
        raise ValueError("Marker-center tolerance must be finite and positive")
    values = tuple(_marker(marker) for marker in markers)
    if not values:
        return None, None
    errors: list[float] = []
    for x, y, radius in values:
        estimated = _estimated_ink_center(candidate, (x, y), radius)
        if estimated is None:
            errors.append(max(tolerance_pixels * 2.0, radius))
            continue
        error = math.hypot(estimated[0] - x, estimated[1] - y)
        errors.append(error)

    if predicted_markers is None:
        f1: float | None = None
    else:
        predictions = tuple(_marker(marker) for marker in predicted_markers)
        adjacency: dict[int, tuple[int, ...]] = {}
        for expected_index, (expected_x, expected_y, _expected_radius) in enumerate(values):
            eligible = sorted(
                (
                    math.hypot(predicted_x - expected_x, predicted_y - expected_y),
                    predicted_index,
                )
                for predicted_index, (predicted_x, predicted_y, _predicted_radius) in enumerate(predictions)
                if math.hypot(predicted_x - expected_x, predicted_y - expected_y)
                <= tolerance_pixels
            )
            adjacency[expected_index] = tuple(predicted_index for _distance, predicted_index in eligible)

        prediction_matches: dict[int, int] = {}

        def augment(expected_index: int, visited: set[int]) -> bool:
            for predicted_index in adjacency[expected_index]:
                if predicted_index in visited:
                    continue
                visited.add(predicted_index)
                prior = prediction_matches.get(predicted_index)
                if prior is None or augment(prior, visited):
                    prediction_matches[predicted_index] = expected_index
                    return True
            return False

        expected_order = sorted(adjacency, key=lambda index: (len(adjacency[index]), index))
        true_positives = sum(augment(expected_index, set()) for expected_index in expected_order)
        false_positives = len(predictions) - true_positives
        false_negatives = len(values) - true_positives
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = 2 * true_positives / denominator if denominator else 1.0
    return f1, sum(errors) / len(errors)


def _line_points(line: object, shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    if isinstance(line, Mapping):
        start = line.get("start")
        end = line.get("end")
        values = (*start, *end) if isinstance(start, Sequence) and isinstance(end, Sequence) else ()
    else:
        values = tuple(line) if isinstance(line, Sequence) and not isinstance(line, (str, bytes)) else ()
    if len(values) != 4:
        raise ValueError("An axis line must provide x1, y1, x2, y2")
    x1, y1, x2, y2 = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("Axis coordinates must be finite")
    steps = max(1, int(math.ceil(max(abs(x2 - x1), abs(y2 - y1)))))
    height, width = shape
    points = {
        (
            max(0, min(width - 1, int(round(x1 + (x2 - x1) * index / steps)))),
            max(0, min(height - 1, int(round(y1 + (y2 - y1) * index / steps)))),
        )
        for index in range(steps + 1)
    }
    return tuple(sorted(points, key=lambda point: (point[1], point[0])))


def axis_thin_line_metrics(
    candidate: np.ndarray,
    lines: Iterable[object],
    *,
    search_radius: int = 4,
) -> tuple[float | None, float | None]:
    """Return thin-line recall and mean nearest-ink localization error."""

    if search_radius < 1:
        raise ValueError("Axis search radius must be at least one pixel")
    ink = _ink(candidate)
    points = tuple(point for line in lines for point in _line_points(line, ink.shape))
    if not points:
        return None, None
    matched = 0
    distances: list[float] = []
    for x, y in points:
        nearest = float(search_radius + 1)
        for yy in range(max(0, y - search_radius), min(ink.shape[0], y + search_radius + 1)):
            for xx in range(max(0, x - search_radius), min(ink.shape[1], x + search_radius + 1)):
                if ink[yy, xx]:
                    nearest = min(nearest, math.hypot(xx - x, yy - y))
        matched += nearest <= 1.0
        distances.append(nearest)
    return matched / len(points), sum(distances) / len(distances)


def open_marker_preservation_rate(
    reference: np.ndarray,
    candidate: np.ndarray,
    markers: Iterable[object],
) -> float | None:
    """Return the fraction of open markers whose light interiors remain open."""

    reference_gray = _grayscale(reference)
    candidate_gray = _grayscale(candidate)
    if reference_gray.shape != candidate_gray.shape:
        raise ValueError("Reference and candidate images must have identical dimensions")
    values = tuple(_marker(marker) for marker in markers)
    if not values:
        return None
    preserved = 0
    for x, y, radius in values:
        yy, xx = np.ogrid[: reference_gray.shape[0], : reference_gray.shape[1]]
        interior_radius = max(0.75, radius * 0.42)
        interior = (xx - x) ** 2 + (yy - y) ** 2 <= interior_radius**2
        if not interior.any():
            continue
        reference_dark = float((reference_gray[interior] < 0.72).mean())
        candidate_dark = float((candidate_gray[interior] < 0.72).mean())
        preserved += candidate_dark <= max(0.20, reference_dark + 0.12)
    return preserved / len(values)


def hallucinated_structure_rate(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Measure candidate ink lying outside a one-pixel ground-truth envelope."""

    reference_ink = _ink(reference)
    candidate_ink = _ink(candidate)
    if reference_ink.shape != candidate_ink.shape:
        raise ValueError("Reference and candidate images must have identical dimensions")
    false_ink = np.logical_and(candidate_ink, np.logical_not(_dilate(reference_ink, radius=1)))
    denominator = int(candidate_ink.sum())
    return float(false_ink.sum()) / denominator if denominator else 0.0


def evaluate_quality_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    annotations: Mapping[str, object],
) -> QualityMetrics:
    """Evaluate all structural metrics for one benchmark image pair."""

    reference_gray = _grayscale(reference)
    candidate_gray = _grayscale(candidate)
    if reference_gray.shape != candidate_gray.shape:
        raise ValueError(
            f"Candidate shape {candidate_gray.shape} does not match reference shape {reference_gray.shape}"
        )
    ocr_regions = annotations.get("ocr_regions", ())
    marker_centers = annotations.get("marker_centers", ())
    axis_lines = annotations.get("axis_lines", ())
    open_markers = annotations.get("open_markers", ())
    if not all(isinstance(value, Iterable) and not isinstance(value, (str, bytes)) for value in (
        ocr_regions,
        marker_centers,
        axis_lines,
        open_markers,
    )):
        raise ValueError("Annotation collections must be arrays")
    ocr_boxes = tuple(
        region["box"] if isinstance(region, Mapping) and "box" in region else region
        for region in ocr_regions
    )
    center_f1, center_error = marker_center_metrics(candidate_gray, marker_centers)
    line_recall, line_error = axis_thin_line_metrics(candidate_gray, axis_lines)
    return QualityMetrics(
        # These two downstream metrics require approved OCR and marker
        # classifier adapters.  Pixel topology must not be relabeled as actual
        # recognition/classification evidence.
        numeric_ocr_exact_match=None,
        numeric_ocr_proxy_exact_match=numeric_ocr_proxy_exact_match(
            reference_gray,
            candidate_gray,
            ocr_boxes,
        ),
        marker_center_f1=center_f1,
        shape_fill_classification_f1=None,
        marker_center_mean_error_pixels=center_error,
        axis_thin_line_recall=line_recall,
        axis_line_localization_error_pixels=line_error,
        open_marker_preservation_rate=open_marker_preservation_rate(
            reference_gray,
            candidate_gray,
            open_markers,
        ),
        hallucinated_structure_rate=hallucinated_structure_rate(reference_gray, candidate_gray),
    )


def mean_quality_metrics(values: Iterable[QualityMetrics]) -> dict[str, float | None]:
    """Aggregate per-case metrics while retaining unavailable observations."""

    materialized = tuple(values)
    if not materialized:
        raise ValueError("At least one quality observation is required")
    result: dict[str, float | None] = {}
    for name in METRIC_NAMES[:-2]:
        observations = tuple(getattr(value, name) for value in materialized)
        if any(observation is None for observation in observations):
            result[name] = None
        else:
            numeric = tuple(float(observation) for observation in observations if observation is not None)
            result[name] = sum(numeric) / len(numeric)
    return result


__all__ = [
    "METRIC_NAMES",
    "QualityMetrics",
    "axis_thin_line_metrics",
    "evaluate_quality_metrics",
    "hallucinated_structure_rate",
    "marker_center_metrics",
    "mean_quality_metrics",
    "numeric_ocr_proxy_exact_match",
    "open_marker_preservation_rate",
]
