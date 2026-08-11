# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed structural rejection followed by the checksum-bound OCR V4 P3 model."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Callable, Iterable

import numpy as np

from ml.ocr.component_geometric_v4.dataset import LabelSample
from ml.ocr.component_geometric_v4.p3_dataset import isolate_glyphs_shape_and_geometry
from ml.ocr.component_geometric_v4.protocol import ALPHABET, GLYPH_WIDTH, REJECT_CLASS_INDEX

from .protocol import STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO


Runner = Callable[[np.ndarray], np.ndarray]
_NUMERIC_GRAMMAR = re.compile(r"^-?\d+(?:\.\d+)?%?$")


def component_height_ratios(raster: np.ndarray) -> tuple[float, ...]:
    glyphs = isolate_glyphs_shape_and_geometry(raster)
    return tuple(float(glyph[0, 0, GLYPH_WIDTH]) for glyph in glyphs)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def decode_raster(
    raster: np.ndarray,
    runner: Runner,
    confidence_threshold: float,
    structural_reject_minimum_height_ratio: float = STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
) -> tuple[str, bool, tuple[float, ...]]:
    glyphs = isolate_glyphs_shape_and_geometry(raster)
    heights = tuple(float(glyph[0, 0, GLYPH_WIDTH]) for glyph in glyphs)
    if not glyphs or len(glyphs) > 8:
        return "", False, heights
    if any(height >= structural_reject_minimum_height_ratio for height in heights):
        return "", True, heights
    logits = np.asarray(runner(np.stack(glyphs).astype(np.float32)), dtype=np.float32)
    if logits.shape != (len(glyphs), len(ALPHABET) + 1) or not np.isfinite(logits).all():
        raise RuntimeError("Structural-filter runner returned an invalid glyph-logit tensor")
    probabilities = _softmax(logits)
    indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    if np.any(confidences < confidence_threshold) or np.any(indices == REJECT_CLASS_INDEX):
        return "", False, heights
    text = "".join(ALPHABET[int(index)] for index in indices)
    return (text if _NUMERIC_GRAMMAR.fullmatch(text) else ""), False, heights


def _distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def evaluate_samples(
    samples: Iterable[LabelSample],
    runner: Runner,
    confidence_threshold: float,
) -> dict[str, object]:
    positives = exact = character_errors = character_count = 0
    exclusions = exclusion_correct = role_correct = total = structural_rejections = 0
    case_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    predictions: list[dict[str, object]] = []
    for sample in samples:
        prediction, structural_rejected, heights = decode_raster(
            sample.raster,
            runner,
            confidence_threshold,
        )
        structural_rejections += int(structural_rejected)
        total += 1
        predicted_role = "numeric_text" if prediction else "non_numeric"
        role_correct += int(sample.role == predicted_role)
        if sample.exclusion_kind is None:
            positives += 1
            match = prediction == sample.target_text
            exact += int(match)
            character_errors += _distance(sample.target_text, prediction)
            character_count += len(sample.target_text)
            case_counts[sample.case][0] += int(match)
            case_counts[sample.case][1] += 1
        else:
            exclusions += 1
            exclusion_correct += int(prediction == "")
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "target_text": sample.target_text,
                "prediction": prediction,
                "role": predicted_role,
                "exclusion_kind": sample.exclusion_kind,
                "structural_rejected": structural_rejected,
                "component_height_ratios": list(heights),
            }
        )
    return {
        "confidence_threshold": confidence_threshold,
        "structural_reject_minimum_height_ratio": STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
        "positive_count": positives,
        "exclusion_count": exclusions,
        "structural_rejection_count": structural_rejections,
        "exact_match": exact / max(1, positives),
        "character_error_rate": character_errors / max(1, character_count),
        "role_accuracy": role_correct / max(1, total),
        "marker_exclusion_accuracy": exclusion_correct / max(1, exclusions),
        "case_exact_match": {
            key: values[0] / max(1, values[1]) for key, values in sorted(case_counts.items())
        },
        "predictions": predictions,
    }


__all__ = ["component_height_ratios", "decode_raster", "evaluate_samples"]
