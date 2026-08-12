# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Exact three-model execution and fixed OCR composition metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from typing import Any

import numpy as np

from ml.ocr.component_context_detector_v7.dataset import box_iou, encode_proposal, proposals
from ml.ocr.component_ensemble_v5.dataset import isolate_glyphs
from ml.ocr.component_ensemble_v5.protocol import ALPHABET as NUMERIC_ALPHABET

from .dataset import CompositionScene, TextTruth
from .protocol import (
    DETECTOR_THRESHOLD,
    NUMERIC_THRESHOLD,
    PLOT_BOUNDS,
    TRUTH_MATCH_IOU_MINIMUM,
)


_GRAPH_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?$")
_PHASE_TERMS = {
    "a",
    "b",
    "ab",
    "baseline",
    "intervention",
    "treatment",
    "maintenance",
    "generalization",
    "follow up",
}


@dataclass(frozen=True)
class DirectTensorEvidence:
    calls: int
    input_tensor_stream_sha256: str
    output_tensor_stream_sha256: str


class DirectRunner:
    def __init__(self, session: Any, input_name: str) -> None:
        self.session = session
        self.input_name = input_name
        self.calls = 0
        self.input_digest = sha256()
        self.output_digest = sha256()

    def run(self, value: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        self.input_digest.update(contiguous.tobytes(order="C"))
        output = np.asarray(self.session.run(None, {self.input_name: contiguous})[0], dtype=np.float32)
        if not np.isfinite(output).all():
            raise RuntimeError("OCR composition model returned non-finite output")
        self.output_digest.update(np.ascontiguousarray(output).tobytes(order="C"))
        self.calls += 1
        return output

    def evidence(self) -> DirectTensorEvidence:
        return DirectTensorEvidence(
            self.calls,
            self.input_digest.hexdigest(),
            self.output_digest.hexdigest(),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _sample_bilinear(gray: np.ndarray, x: float, y: float) -> float:
    bounded_x = float(np.clip(x, 0.0, gray.shape[1] - 1.0))
    bounded_y = float(np.clip(y, 0.0, gray.shape[0] - 1.0))
    x0 = int(math.floor(bounded_x))
    y0 = int(math.floor(bounded_y))
    x1 = min(x0 + 1, gray.shape[1] - 1)
    y1 = min(y0 + 1, gray.shape[0] - 1)
    x_weight = bounded_x - x0
    y_weight = bounded_y - y0
    top = gray[y0, x0] * (1.0 - x_weight) + gray[y0, x1] * x_weight
    bottom = gray[y1, x0] * (1.0 - x_weight) + gray[y1, x1] * x_weight
    return float(top * (1.0 - y_weight) + bottom * y_weight)


def _crop(
    gray: np.ndarray,
    box: Any,
    *,
    target_width: int,
    target_height: int,
    horizontal_padding: float,
    vertical_padding: float,
    vertical_content_padding_ratio: float,
    padding_value: float,
) -> np.ndarray:
    effective_vertical_padding = vertical_padding + box.height * vertical_content_padding_ratio
    left = box.left - horizontal_padding
    top = box.top - effective_vertical_padding
    width = box.width + 2.0 * horizontal_padding
    height = box.height + 2.0 * effective_vertical_padding
    content_width = int(np.clip(math.ceil(target_height * width / height), 1, target_width))
    output = np.full((target_height, target_width), padding_value, dtype=np.float32)
    for target_y in range(target_height):
        source_y = top + ((target_y + 0.5) / target_height) * height - 0.5
        for target_x in range(content_width):
            source_x = left + ((target_x + 0.5) / content_width) * width - 0.5
            output[target_y, target_x] = _sample_bilinear(gray, source_x, source_y) / 255.0
    return output


def _decode_ctc(output: np.ndarray, alphabet: str) -> tuple[str, float]:
    values = np.asarray(output, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != len(alphabet) + 1:
        raise RuntimeError("Official recognizer output violates the frozen CTC contract")
    rows = values[0]
    sums = rows.sum(axis=1)
    if np.all((rows >= 0.0) & (rows <= 1.0)) and np.allclose(sums, 1.0, rtol=0.0, atol=max(1e-5, rows.shape[1] * 1e-6)):
        probabilities = rows
    else:
        probabilities = _softmax(rows)
    classes = probabilities.argmax(axis=1)
    result: list[str] = []
    accepted: list[float] = []
    prior = -1
    for index, value in enumerate(classes.tolist()):
        if value != 0 and value != prior:
            result.append(alphabet[value - 1])
            accepted.append(float(probabilities[index, value]))
        prior = value
    return "".join(result), float(np.mean(accepted)) if accepted else 0.0


def _official_recognize(gray: np.ndarray, box: Any, runner: DirectRunner, alphabet: str) -> tuple[str, float]:
    crop = _crop(
        gray,
        box,
        target_width=320,
        target_height=48,
        horizontal_padding=8.0,
        vertical_padding=2.0,
        vertical_content_padding_ratio=0.0,
        padding_value=0.5,
    )
    normalized = (crop - 0.5) * 2.0
    tensor = np.ascontiguousarray(np.broadcast_to(normalized, (3, 48, 320))[None, :, :, :])
    return _decode_ctc(runner.run(tensor), alphabet)


def _numeric_recognize(gray: np.ndarray, box: Any, runner: DirectRunner) -> tuple[str, float]:
    crop = _crop(
        gray,
        box,
        target_width=128,
        target_height=32,
        horizontal_padding=12.0,
        vertical_padding=1.0,
        vertical_content_padding_ratio=0.25,
        padding_value=1.0,
    )
    quantized = np.rint(crop * 255.0).clip(0, 255).astype(np.uint8)
    glyphs = isolate_glyphs(quantized)
    if not glyphs or len(glyphs) > 8:
        return "", 0.0
    values = np.stack(glyphs).astype(np.float32)
    if np.any(values[:, 0, 0, 20] >= 0.75):
        return "", 0.0
    logits = runner.run(values)
    if logits.shape != (len(glyphs), len(NUMERIC_ALPHABET) + 1):
        raise RuntimeError("Numeric recognizer output violates the frozen component contract")
    probabilities = _softmax(logits)
    indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    if np.any(confidences < NUMERIC_THRESHOLD) or np.any(indices == len(NUMERIC_ALPHABET)):
        return "", float(np.mean(confidences))
    text = "".join(NUMERIC_ALPHABET[int(index)] for index in indices)
    return (text, float(np.mean(confidences))) if _GRAPH_NUMBER.fullmatch(text) else ("", float(np.mean(confidences)))


def _role(box: Any, recognized_text: str) -> str:
    left, top, right, bottom = PLOT_BOUNDS
    center_x = (box.left + box.right) / 2.0
    center_y = (box.top + box.bottom) / 2.0
    horizontal_tolerance = max(4.0, (right - left) * 0.05)
    vertical_tolerance = max(4.0, (bottom - top) * 0.05)
    within_x = left - horizontal_tolerance <= center_x <= right + horizontal_tolerance
    within_y = top - vertical_tolerance <= center_y <= bottom + vertical_tolerance
    numeric = _GRAPH_NUMBER.fullmatch(recognized_text.strip()) is not None or recognized_text.strip() in {"O", "o", "I", "l"}
    if numeric and center_y > bottom and within_x:
        return "x_tick"
    if numeric and center_x < left and within_y:
        return "y_tick"
    normalized = recognized_text.strip().replace("_", " ").replace("-", " ").lower()
    above_plot = box.bottom <= top + vertical_tolerance
    right_of_plot = box.left >= right - horizontal_tolerance
    inside_plot = left <= center_x <= right and top <= center_y <= bottom
    if not numeric and above_plot and within_x and (normalized in _PHASE_TERMS or normalized.startswith("phase")):
        return "phase_heading"
    if not numeric and right_of_plot and within_y:
        return "legend_text"
    if not numeric and inside_plot:
        return "annotation"
    return "other"


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


def evaluate_scenes(
    scenes: tuple[CompositionScene, ...],
    detector_runner: DirectRunner,
    official_runner: DirectRunner,
    numeric_runner: DirectRunner,
    official_alphabet: str,
) -> dict[str, object]:
    case_results: list[dict[str, object]] = []
    true_positives = false_positives = false_negatives = duplicates = 0
    exact = character_errors = character_count = role_correct = 0
    numeric_exact = numeric_count = word_exact = word_count = 0
    forbidden_numeric_routes = 0
    route_counts: dict[str, int] = defaultdict(int)
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = detector_runner.run(values)
        if logits.shape != (len(candidates), 2):
            raise RuntimeError("Detector output violates the frozen proposal contract")
        probabilities = _softmax(logits)[:, 1]
        accepted = [item for item, probability in zip(candidates, probabilities, strict=True) if probability >= DETECTOR_THRESHOLD]
        matched_truths: set[int] = set()
        scene_false_positives = scene_duplicates = 0
        predictions: list[dict[str, object]] = []
        for candidate in accepted:
            matches = [
                index
                for index, truth in enumerate(scene.truths)
                if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
            ]
            if not matches:
                scene_false_positives += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index].box))
            if best in matched_truths:
                scene_duplicates += 1
                continue
            matched_truths.add(best)
            truth: TextTruth = scene.truths[best]
            official_text, official_confidence = _official_recognize(
                scene.raster, candidate.box, official_runner, official_alphabet
            )
            numeric_text, numeric_confidence = _numeric_recognize(scene.raster, candidate.box, numeric_runner)
            general_role = _role(candidate.box, official_text)
            numeric_role = _role(candidate.box, numeric_text)
            select_numeric = (
                bool(numeric_text)
                and numeric_confidence >= NUMERIC_THRESHOLD
                and numeric_role in {"x_tick", "y_tick"}
            )
            if select_numeric:
                prediction = numeric_text
                predicted_role = numeric_role
                route = "numeric_specialist"
            else:
                prediction = official_text
                predicted_role = general_role
                route = "general_recognizer"
            route_counts[route] += 1
            forbidden_numeric_routes += int(select_numeric and truth.role not in {"x_tick", "y_tick"})
            matched = prediction == truth.truth_text
            exact += int(matched)
            character_errors += _distance(truth.truth_text, prediction)
            character_count += len(truth.truth_text)
            role_correct += int(predicted_role == truth.role)
            if truth.role in {"x_tick", "y_tick"}:
                numeric_count += 1
                numeric_exact += int(matched)
            else:
                word_count += 1
                word_exact += int(matched)
            predictions.append(
                {
                    "truth_text": truth.truth_text,
                    "display_text": truth.display_text,
                    "truth_role": truth.role,
                    "prediction": prediction,
                    "predicted_role": predicted_role,
                    "route": route,
                    "official_prediction": official_text,
                    "official_confidence": official_confidence,
                    "numeric_prediction": numeric_text,
                    "numeric_confidence": numeric_confidence,
                    "proposal_bbox": [candidate.box.left, candidate.box.top, candidate.box.right, candidate.box.bottom],
                    "truth_bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom],
                    "exact": matched,
                }
            )
        scene_false_negatives = len(scene.truths) - len(matched_truths)
        true_positives += len(matched_truths)
        false_positives += scene_false_positives
        false_negatives += scene_false_negatives
        duplicates += scene_duplicates
        case_results.append(
            {
                "scene_id": scene.scene_id,
                "source_raster_sha256": sha256(scene.raster.tobytes(order="C")).hexdigest(),
                "truth_region_count": len(scene.truths),
                "proposal_count": len(candidates),
                "accepted_region_count": len(accepted),
                "true_positives": len(matched_truths),
                "false_positives": scene_false_positives,
                "false_negatives": scene_false_negatives,
                "duplicate_region_count": scene_duplicates,
                "prohibited_structure_hits": scene_false_positives,
                "exact_detection": scene_false_positives == scene_false_negatives == scene_duplicates == 0,
                "predictions": predictions,
            }
        )
    total_truths = sum(len(scene.truths) for scene in scenes)
    return {
        "scene_count": len(scenes),
        "truth_region_count": total_truths,
        "exact_detection_scene_count": sum(int(item["exact_detection"]) for item in case_results),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "duplicate_region_count": duplicates,
        "prohibited_structure_hits": false_positives,
        "recognition_exact_match": exact / max(1, total_truths),
        "character_error_rate": character_errors / max(1, character_count),
        "role_accuracy": role_correct / max(1, total_truths),
        "numeric_exact_match": numeric_exact / max(1, numeric_count),
        "word_exact_match": word_exact / max(1, word_count),
        "numeric_case_count": numeric_count,
        "word_case_count": word_count,
        "forbidden_numeric_route_count": forbidden_numeric_routes,
        "route_counts": dict(sorted(route_counts.items())),
        "marker_creation_evaluated": False,
        "cases": case_results,
    }


__all__ = ["DirectRunner", "DirectTensorEvidence", "evaluate_scenes"]
