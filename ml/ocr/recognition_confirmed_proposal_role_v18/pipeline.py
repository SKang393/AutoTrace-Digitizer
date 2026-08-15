# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Direct two-model composition metrics for OCR V18."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable

import numpy as np
from PIL import Image

from ml.ocr.component_region_detector_v6.dataset import box_iou
from ml.ocr.official_bakeoff.production_evaluate import recognition_tensor
from .dataset import SceneSample, encode_proposal, proposals
from .protocol import (
    CHARACTER_ERROR_RATE_MAXIMUM,
    DETECTOR_THRESHOLD,
    RECOGNITION_CONFIDENCE_THRESHOLD,
    RECOGNITION_EXACT_MINIMUM,
    ROLE_ACCURACY_MINIMUM,
    ROLE_CLASS_ACCURACY_MINIMUM,
    ROLE_ORDER,
    TRUTH_MATCH_IOU_MINIMUM,
)


DetectorRunner = Callable[[np.ndarray], np.ndarray]
RecognizerRunner = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RecognitionPrediction:
    text: str
    confidence: float


def _probabilities(output: np.ndarray) -> np.ndarray:
    value = np.asarray(output, dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise RuntimeError("OCR V18 recognizer returned invalid output")
    sums = value.sum(axis=2)
    if float(value.min()) < 0.0 or float(value.max()) > 1.0 or not np.allclose(
        sums, 1.0, rtol=2e-3, atol=2e-3,
    ):
        shifted = value - value.max(axis=2, keepdims=True)
        exponent = np.exp(shifted)
        value = exponent / exponent.sum(axis=2, keepdims=True)
    return value


def decode_with_confidence(output: np.ndarray, alphabet: str) -> tuple[RecognitionPrediction, ...]:
    probabilities = _probabilities(output)
    if probabilities.shape[2] != len(alphabet) + 1:
        raise RuntimeError("OCR V18 recognizer alphabet does not match output classes")
    predictions: list[RecognitionPrediction] = []
    for row in probabilities:
        classes = row.argmax(axis=1).tolist()
        prior = -1
        characters: list[str] = []
        confidence: list[float] = []
        for time_index, value in enumerate(classes):
            if value != 0 and value != prior:
                characters.append(alphabet[value - 1])
                confidence.append(float(row[time_index, value]))
            prior = value
        predictions.append(RecognitionPrediction(
            "".join(characters),
            sum(confidence) / len(confidence) if confidence else 0.0,
        ))
    return tuple(predictions)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def evaluate_composition(
    scenes: tuple[SceneSample, ...],
    detector_runner: DetectorRunner,
    recognizer_runner: RecognizerRunner,
    alphabet: str,
    *,
    recognition_batch_size: int = 64,
) -> dict[str, object]:
    if recognition_batch_size <= 0:
        raise ValueError("OCR V18 recognition batch size must be positive")
    detector_input_stream = sha256()
    detector_output_stream = sha256()
    recognition_input_stream = sha256()
    recognition_output_stream = sha256()
    cached: list[tuple[SceneSample, object, np.ndarray, np.ndarray, list[int]]] = []
    recognition_tensors: list[np.ndarray] = []
    detector_calls = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        if not candidates:
            raise RuntimeError("OCR V18 scene produced no proposals")
        values = np.stack([
            encode_proposal(scene.raster, candidate, scene.plot) for candidate in candidates
        ]).astype(np.float32)
        detector_input_stream.update(np.ascontiguousarray(values).tobytes(order="C"))
        logits = np.asarray(detector_runner(values), dtype=np.float32)
        detector_calls += 1
        if logits.shape != (len(candidates), 2 + len(ROLE_ORDER)) or not np.isfinite(logits).all():
            raise RuntimeError("OCR V18 detector returned invalid logits")
        detector_output_stream.update(np.ascontiguousarray(logits).tobytes(order="C"))
        shifted = logits[:, :2] - logits[:, :2].max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        detector_probability = exponent[:, 1] / exponent.sum(axis=1)
        accepted_indices = [
            index for index, value in enumerate(detector_probability)
            if float(value) >= DETECTOR_THRESHOLD
        ]
        image = Image.fromarray(scene.raster, mode="L")
        tensor_indices: list[int] = []
        for candidate_index in accepted_indices:
            box = candidates[candidate_index].box
            crop = image.crop((box.left, box.top, box.right, box.bottom))
            tensor_indices.append(len(recognition_tensors))
            recognition_tensors.append(recognition_tensor(crop)[0])
        cached.append((scene, candidates, logits, detector_probability, tensor_indices))

    predictions: list[RecognitionPrediction] = []
    recognition_batches = 0
    for start in range(0, len(recognition_tensors), recognition_batch_size):
        batch = np.stack(recognition_tensors[start:start + recognition_batch_size]).astype(np.float32)
        recognition_input_stream.update(np.ascontiguousarray(batch).tobytes(order="C"))
        output = np.asarray(recognizer_runner(batch), dtype=np.float32)
        recognition_batches += 1
        if output.shape[0] != len(batch):
            raise RuntimeError("OCR V18 recognizer changed the requested batch size")
        recognition_output_stream.update(np.ascontiguousarray(output).tobytes(order="C"))
        predictions.extend(decode_with_confidence(output, alphabet))
    if len(predictions) != len(recognition_tensors):
        raise RuntimeError("OCR V18 recognition result count changed")

    truth_count = sum(len(scene.truths) for scene in scenes)
    exact_scenes = true_positives = false_positives = false_negatives = duplicates = 0
    correct_roles = recognition_exact = edit_distance = truth_character_count = 0
    accepted_after_confirmation = 0
    per_role = {role: {"correct": 0, "total": 0} for role in ROLE_ORDER}
    prediction_cursor = 0
    for scene, candidates, logits, detector_probability, tensor_indices in cached:
        accepted_indices = [
            index for index, value in enumerate(detector_probability)
            if float(value) >= DETECTOR_THRESHOLD
        ]
        if len(accepted_indices) != len(tensor_indices):
            raise RuntimeError("OCR V18 proposal and recognition index streams diverged")
        scene_predictions = predictions[prediction_cursor:prediction_cursor + len(tensor_indices)]
        prediction_cursor += len(tensor_indices)
        confirmed = [
            (candidate_index, prediction)
            for candidate_index, prediction in zip(accepted_indices, scene_predictions, strict=True)
            if prediction.confidence >= RECOGNITION_CONFIDENCE_THRESHOLD and prediction.text
        ]
        accepted_after_confirmation += len(confirmed)
        matched_truths: set[int] = set()
        scene_fp = scene_dup = scene_role_correct = 0
        predicted_text_by_truth: dict[int, str] = {}
        for candidate_index, prediction in confirmed:
            candidate = candidates[candidate_index]
            matches = [
                index for index, truth in enumerate(scene.truths)
                if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
            ]
            if not matches:
                scene_fp += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index].box))
            if best in matched_truths:
                scene_dup += 1
                continue
            matched_truths.add(best)
            predicted_text_by_truth[best] = prediction.text
            truth_role = scene.truths[best].role
            predicted_role = ROLE_ORDER[int(np.argmax(logits[candidate_index, 2:]))]
            if predicted_role == truth_role:
                correct_roles += 1
                per_role[truth_role]["correct"] += 1
                scene_role_correct += 1
        for truth_index, truth in enumerate(scene.truths):
            per_role[truth.role]["total"] += 1
            predicted_text = predicted_text_by_truth.get(truth_index, "")
            recognition_exact += int(predicted_text == truth.text)
            edit_distance += _edit_distance(predicted_text, truth.text)
            truth_character_count += len(truth.text)
        scene_fn = len(scene.truths) - len(matched_truths)
        true_positives += len(matched_truths)
        false_positives += scene_fp
        false_negatives += scene_fn
        duplicates += scene_dup
        if scene_fp == scene_fn == scene_dup == 0 and scene_role_correct == len(scene.truths):
            exact_scenes += 1

    metrics: dict[str, object] = {
        "scene_count": len(scenes),
        "truth_region_count": truth_count,
        "exact_scene_count": exact_scenes,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "duplicate_region_count": duplicates,
        "prohibited_structure_hits": false_positives,
        "recognition_exact_count": recognition_exact,
        "recognition_exact": recognition_exact / truth_count if truth_count else 0.0,
        "character_error_count": edit_distance,
        "truth_character_count": truth_character_count,
        "character_error_rate": edit_distance / truth_character_count if truth_character_count else 0.0,
        "role_accuracy": correct_roles / truth_count if truth_count else 0.0,
        "per_role_accuracy": {
            role: values["correct"] / values["total"] if values["total"] else 0.0
            for role, values in per_role.items()
        },
        "detector_accepted_proposal_count": len(recognition_tensors),
        "recognition_confirmed_proposal_count": accepted_after_confirmation,
        "detector_inference_calls": detector_calls,
        "recognizer_region_calls": len(recognition_tensors),
        "recognizer_batch_calls": recognition_batches,
        "detector_input_tensor_stream_sha256": detector_input_stream.hexdigest(),
        "detector_output_tensor_stream_sha256": detector_output_stream.hexdigest(),
        "recognizer_input_tensor_stream_sha256": recognition_input_stream.hexdigest(),
        "recognizer_output_tensor_stream_sha256": recognition_output_stream.hexdigest(),
    }
    return metrics


def passes_selection(metrics: dict[str, object]) -> bool:
    per_role = metrics["per_role_accuracy"]
    assert isinstance(per_role, dict)
    return (
        metrics["exact_scene_count"] == metrics["scene_count"]
        and metrics["true_positives"] == metrics["truth_region_count"]
        and metrics["false_positives"] == metrics["false_negatives"]
        == metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
        and float(metrics["recognition_exact"]) >= RECOGNITION_EXACT_MINIMUM
        and float(metrics["character_error_rate"]) <= CHARACTER_ERROR_RATE_MAXIMUM
        and float(metrics["role_accuracy"]) >= ROLE_ACCURACY_MINIMUM
        and min(float(value) for value in per_role.values()) >= ROLE_CLASS_ACCURACY_MINIMUM
    )


__all__ = ["RecognitionPrediction", "decode_with_confidence", "evaluate_composition", "passes_selection"]
