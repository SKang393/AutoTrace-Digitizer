# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Training examples, decoding, and fixed OCR V5 metrics."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Callable, Iterable

import numpy as np

from .dataset import LabelSample, isolate_glyphs
from .protocol import (
    ALPHABET,
    GLYPH_WIDTH,
    REJECT_CLASS_INDEX,
    STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
)


Runner = Callable[[np.ndarray], np.ndarray]
_NUMERIC_GRAMMAR = re.compile(r"^-?\d+(?:\.\d+)?%?$")


def glyph_training_examples(samples: Iterable[LabelSample]) -> tuple[np.ndarray, np.ndarray]:
    glyphs: list[np.ndarray] = []
    labels: list[int] = []
    for sample in samples:
        isolated = isolate_glyphs(sample.raster)
        if sample.exclusion_kind is None:
            if len(isolated) != len(sample.target_text):
                raise RuntimeError(
                    f"OCR V5 isolation changed text length for {sample.sample_id}: "
                    f"{len(isolated)} != {len(sample.target_text)}"
                )
            for glyph, character in zip(isolated, sample.target_text, strict=True):
                glyphs.append(glyph)
                labels.append(ALPHABET.index(character))
        else:
            for glyph in isolated:
                glyphs.append(glyph)
                labels.append(REJECT_CLASS_INDEX)
    if not glyphs:
        raise RuntimeError("OCR V5 training split produced no glyphs")
    return np.stack(glyphs).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def decode_raster(raster: np.ndarray, runner: Runner, threshold: float) -> str:
    glyphs = isolate_glyphs(raster)
    if not glyphs or len(glyphs) > 8:
        return ""
    values = np.stack(glyphs).astype(np.float32)
    height_ratios = values[:, 0, 0, GLYPH_WIDTH]
    if np.any(height_ratios >= STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO):
        return ""
    logits = np.asarray(runner(values), dtype=np.float32)
    if logits.shape != (len(glyphs), len(ALPHABET) + 1) or not np.isfinite(logits).all():
        raise RuntimeError("OCR V5 runner returned an invalid glyph-logit tensor")
    probabilities = _softmax(logits)
    indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    if np.any(confidences < threshold) or np.any(indices == REJECT_CLASS_INDEX):
        return ""
    text = "".join(ALPHABET[int(index)] for index in indices)
    return text if _NUMERIC_GRAMMAR.fullmatch(text) else ""


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


def evaluate_samples(samples: Iterable[LabelSample], runner: Runner, threshold: float) -> dict[str, object]:
    positives = exact = character_errors = character_count = 0
    exclusions = exclusion_correct = role_correct = total = 0
    case_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    predictions: list[dict[str, object]] = []
    for sample in samples:
        prediction = decode_raster(sample.raster, runner, threshold)
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
            }
        )
    return {
        "threshold": threshold,
        "positive_count": positives,
        "exclusion_count": exclusions,
        "exact_match": exact / max(1, positives),
        "character_error_rate": character_errors / max(1, character_count),
        "role_accuracy": role_correct / max(1, total),
        "marker_exclusion_accuracy": exclusion_correct / max(1, exclusions),
        "case_exact_match": {
            key: values[0] / max(1, values[1]) for key, values in sorted(case_counts.items())
        },
        "predictions": predictions,
    }


__all__ = ["decode_raster", "evaluate_samples", "glyph_training_examples"]
