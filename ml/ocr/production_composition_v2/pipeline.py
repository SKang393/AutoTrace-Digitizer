# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Exact V9, official spacing-P2, and numeric-V5 composed execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals
from ml.ocr.component_ensemble_v5.dataset import isolate_glyphs
from ml.ocr.component_ensemble_v5.protocol import ALPHABET as NUMERIC_ALPHABET
from ml.ocr.component_recall_detector_v9.dataset import encode_proposal
from ml.ocr.official_bakeoff.production_evaluate import decode_ctc, recognition_tensor
from ml.ocr.official_recognition_spacing_v2.spacing import restore_source_evidenced_spaces_and_vertical_case

from .dataset import CompositionScene, TextTruth
from .protocol import DETECTOR_THRESHOLD, NUMERIC_THRESHOLD, PLOT_BOUNDS, TRUTH_MATCH_IOU_MINIMUM


_GRAPH_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?$")
_PHASE_TERMS = {"a", "b", "ab", "baseline", "intervention", "treatment", "maintenance", "followup", "follow up"}


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
            raise RuntimeError("OCR composition V2 model returned non-finite output")
        self.output_digest.update(np.ascontiguousarray(output).tobytes(order="C"))
        self.calls += 1
        return output

    def evidence(self) -> DirectTensorEvidence:
        return DirectTensorEvidence(self.calls, self.input_digest.hexdigest(), self.output_digest.hexdigest())


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _source_crop(gray: np.ndarray, box: Any, horizontal: int, vertical: int) -> Image.Image:
    left = max(0, int(math.floor(box.left)) - horizontal)
    top = max(0, int(math.floor(box.top)) - vertical)
    right = min(gray.shape[1], int(math.ceil(box.right)) + horizontal)
    bottom = min(gray.shape[0], int(math.ceil(box.bottom)) + vertical)
    if right <= left or bottom <= top:
        raise RuntimeError("OCR composition V2 recognition crop is empty")
    return Image.fromarray(gray[top:bottom, left:right], mode="L").convert("RGB")


def _official_recognize(gray: np.ndarray, box: Any, runner: DirectRunner, alphabet: str) -> tuple[str, str]:
    crop = _source_crop(gray, box, 8, 2)
    raw = decode_ctc(runner.run(recognition_tensor(crop)), alphabet)
    return raw, restore_source_evidenced_spaces_and_vertical_case(crop, raw)


def _sample_bilinear(gray: np.ndarray, x: float, y: float) -> float:
    bounded_x = float(np.clip(x, 0.0, gray.shape[1] - 1.0))
    bounded_y = float(np.clip(y, 0.0, gray.shape[0] - 1.0))
    x0, y0 = int(math.floor(bounded_x)), int(math.floor(bounded_y))
    x1, y1 = min(x0 + 1, gray.shape[1] - 1), min(y0 + 1, gray.shape[0] - 1)
    x_weight, y_weight = bounded_x - x0, bounded_y - y0
    top = gray[y0, x0] * (1.0 - x_weight) + gray[y0, x1] * x_weight
    bottom = gray[y1, x0] * (1.0 - x_weight) + gray[y1, x1] * x_weight
    return float(top * (1.0 - y_weight) + bottom * y_weight)


def _numeric_crop(gray: np.ndarray, box: Any) -> np.ndarray:
    target_width, target_height = 128, 32
    horizontal_padding, vertical_padding = 12.0, 1.0 + box.height * 0.25
    left, top = box.left - horizontal_padding, box.top - vertical_padding
    width, height = box.width + 2 * horizontal_padding, box.height + 2 * vertical_padding
    content_width = int(np.clip(math.ceil(target_height * width / height), 1, target_width))
    output = np.ones((target_height, target_width), dtype=np.float32)
    for target_y in range(target_height):
        source_y = top + ((target_y + 0.5) / target_height) * height - 0.5
        for target_x in range(content_width):
            source_x = left + ((target_x + 0.5) / content_width) * width - 0.5
            output[target_y, target_x] = _sample_bilinear(gray, source_x, source_y) / 255.0
    return np.rint(output * 255.0).clip(0, 255).astype(np.uint8)


def _numeric_recognize(gray: np.ndarray, box: Any, runner: DirectRunner) -> tuple[str, float]:
    glyphs = isolate_glyphs(_numeric_crop(gray, box))
    if not glyphs or len(glyphs) > 8:
        return "", 0.0
    values = np.stack(glyphs).astype(np.float32)
    if np.any(values[:, 0, 0, 20] >= 0.75):
        return "", 0.0
    logits = runner.run(values)
    if logits.shape != (len(glyphs), len(NUMERIC_ALPHABET) + 1):
        raise RuntimeError("OCR composition V2 numeric output contract changed")
    probabilities = _softmax(logits)
    indices, confidences = probabilities.argmax(axis=1), probabilities.max(axis=1)
    confidence = float(np.mean(confidences))
    if np.any(confidences < NUMERIC_THRESHOLD) or np.any(indices == len(NUMERIC_ALPHABET)):
        return "", confidence
    text = "".join(NUMERIC_ALPHABET[int(index)] for index in indices)
    return (text, confidence) if _GRAPH_NUMBER.fullmatch(text) else ("", confidence)


def _role(box: Any, text: str) -> str:
    left, top, right, bottom = PLOT_BOUNDS
    center_x, center_y = (box.left + box.right) / 2.0, (box.top + box.bottom) / 2.0
    horizontal_tolerance = max(4.0, (right - left) * 0.05)
    vertical_tolerance = max(4.0, (bottom - top) * 0.05)
    within_x = left - horizontal_tolerance <= center_x <= right + horizontal_tolerance
    within_y = top - vertical_tolerance <= center_y <= bottom + vertical_tolerance
    numeric = _GRAPH_NUMBER.fullmatch(text.strip()) is not None
    if numeric and center_y > bottom and within_x:
        return "x_tick"
    if numeric and center_x < left and within_y:
        return "y_tick"
    normalized = text.strip().replace("_", " ").replace("-", " ").lower()
    if not numeric and box.bottom <= top + vertical_tolerance and within_x and (normalized in _PHASE_TERMS or normalized.startswith("phase")):
        return "phase_heading"
    if not numeric and box.left >= right - horizontal_tolerance and within_y:
        return "legend_text"
    if not numeric and left <= center_x <= right and top <= center_y <= bottom:
        return "annotation"
    return "other"


def _distance(left: str, right: str) -> int:
    prior = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(current[-1] + 1, prior[right_index] + 1, prior[right_index - 1] + (left_character != right_character)))
        prior = current
    return prior[-1]


def evaluate_scenes(
    scenes: tuple[CompositionScene, ...], detector_runner: DirectRunner,
    official_runner: DirectRunner, numeric_runner: DirectRunner, official_alphabet: str,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    true_positives = false_positives = false_negatives = duplicates = 0
    exact = errors = characters = role_correct = changed_nonspace = 0
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    forbidden_numeric_routes = spacing_changes = 0
    route_counts: dict[str, int] = defaultdict(int)
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        probabilities = _softmax(detector_runner.run(values))[:, 1]
        accepted = [item for item, probability in zip(candidates, probabilities, strict=True) if probability >= DETECTOR_THRESHOLD]
        matched_truths: set[int] = set()
        scene_false_positives = scene_duplicates = 0
        predictions: list[dict[str, object]] = []
        for candidate in accepted:
            matches = [index for index, truth in enumerate(scene.truths) if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM]
            if not matches:
                scene_false_positives += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index].box))
            if best in matched_truths:
                scene_duplicates += 1
                continue
            matched_truths.add(best)
            truth: TextTruth = scene.truths[best]
            raw, official_text = _official_recognize(scene.raster, candidate.box, official_runner, official_alphabet)
            numeric_text, numeric_confidence = _numeric_recognize(scene.raster, candidate.box, numeric_runner)
            official_role, numeric_role = _role(candidate.box, official_text), _role(candidate.box, numeric_text)
            select_numeric = bool(numeric_text) and numeric_confidence >= NUMERIC_THRESHOLD and numeric_role in {"x_tick", "y_tick"}
            prediction, predicted_role, route = (
                (numeric_text, numeric_role, "numeric_specialist") if select_numeric
                else (official_text, official_role, "general_recognizer")
            )
            matched = prediction == truth.truth_text
            route_counts[route] += 1
            forbidden_numeric_routes += int(select_numeric and truth.role not in {"x_tick", "y_tick"})
            exact += int(matched)
            errors += _distance(truth.truth_text, prediction)
            characters += len(truth.truth_text)
            role_correct += int(predicted_role == truth.role)
            family_counts[truth.family][0] += int(matched)
            family_counts[truth.family][1] += 1
            spacing_changes += int(official_text != raw)
            changed_nonspace += int(official_text != raw and " " not in truth.truth_text)
            predictions.append({
                "truth_text": truth.truth_text, "display_text": truth.display_text, "truth_role": truth.role,
                "text_family": truth.family, "prediction": prediction, "predicted_role": predicted_role,
                "route": route, "official_raw_prediction": raw, "official_prediction": official_text,
                "numeric_prediction": numeric_text, "numeric_confidence": numeric_confidence,
                "proposal_bbox": [candidate.box.left, candidate.box.top, candidate.box.right, candidate.box.bottom],
                "truth_bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom], "exact": matched,
            })
        scene_false_negatives = len(scene.truths) - len(matched_truths)
        true_positives += len(matched_truths)
        false_positives += scene_false_positives
        false_negatives += scene_false_negatives
        duplicates += scene_duplicates
        cases.append({
            "scene_id": scene.scene_id, "source_raster_sha256": sha256(scene.raster.tobytes(order="C")).hexdigest(),
            "truth_region_count": len(scene.truths), "proposal_count": len(candidates),
            "accepted_region_count": len(accepted), "true_positives": len(matched_truths),
            "false_positives": scene_false_positives, "false_negatives": scene_false_negatives,
            "duplicate_region_count": scene_duplicates, "prohibited_structure_hits": scene_false_positives,
            "exact_detection": scene_false_positives == scene_false_negatives == scene_duplicates == 0,
            "predictions": predictions,
        })
    total = sum(len(scene.truths) for scene in scenes)
    return {
        "scene_count": len(scenes), "truth_region_count": total,
        "exact_detection_scene_count": sum(int(item["exact_detection"]) for item in cases),
        "true_positives": true_positives, "false_positives": false_positives,
        "false_negatives": false_negatives, "duplicate_region_count": duplicates,
        "prohibited_structure_hits": false_positives, "recognition_exact_match": exact / max(1, total),
        "character_error_rate": errors / max(1, characters), "role_accuracy": role_correct / max(1, total),
        "numeric_exact_match": family_counts["numeric"][0] / max(1, family_counts["numeric"][1]),
        "word_exact_match": family_counts["word"][0] / max(1, family_counts["word"][1]),
        "ambiguity_exact_match": family_counts["ambiguity"][0] / max(1, family_counts["ambiguity"][1]),
        "spacing_changed_count": spacing_changes,
        "spacing_changed_nonspace_truth_count": changed_nonspace,
        "forbidden_numeric_route_count": forbidden_numeric_routes,
        "route_counts": dict(sorted(route_counts.items())), "marker_creation_evaluated": False, "cases": cases,
    }


__all__ = ["DirectRunner", "DirectTensorEvidence", "evaluate_scenes"]
