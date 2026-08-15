# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Direct three-model feature extraction and fixed OCR V20 metrics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Literal

import numpy as np
from PIL import Image

from ml.ocr.component_region_detector_v6.dataset import box_iou
from ml.ocr.official_bakeoff.production_evaluate import recognition_tensor
from .dataset import SceneSample, encode_proposal, proposal_targets, proposals
from .protocol import (
    CHARACTER_ERROR_RATE_MAXIMUM, DETECTOR_FLOOR, FEATURE_COUNT,
    RECOGNITION_EXACT_MINIMUM, ROLE_ACCURACY_MINIMUM, ROLE_CLASS_ACCURACY_MINIMUM,
    ROLE_ORDER, ROBUST_THRESHOLD_RUN_LENGTH, TRUTH_MATCH_IOU_MINIMUM,
)


Runner = Callable[[np.ndarray], np.ndarray]
Mode = Literal["train", "evaluate"]


@dataclass(frozen=True)
class ProposalRecord:
    scene_index: int
    candidate_index: int
    truth_index: int
    predicted_text: str
    predicted_role: str


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _recognition_probabilities(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise RuntimeError("OCR V20 recognizer returned invalid output")
    if (
        float(values.min()) < 0.0 or float(values.max()) > 1.0
        or not np.allclose(values.sum(axis=2), 1.0, rtol=2e-3, atol=2e-3)
    ):
        values = _softmax(values)
    return values


def _decode_stats(row: np.ndarray, alphabet: str) -> tuple[str, np.ndarray]:
    classes = row.argmax(axis=1)
    sorted_probabilities = np.sort(row, axis=1)
    top1 = sorted_probabilities[:, -1]
    margin = top1 - sorted_probabilities[:, -2]
    entropy = -np.sum(row * np.log(np.clip(row, 1e-8, 1.0)), axis=1) / np.log(row.shape[1])
    prior = -1
    characters: list[str] = []
    selected: list[float] = []
    for index, value in enumerate(classes.tolist()):
        if value != 0 and value != prior:
            characters.append(alphabet[value - 1])
            selected.append(float(row[index, value]))
        prior = value
    text = "".join(characters)
    length = max(1, len(text))
    statistics = np.asarray((
        sum(selected) / len(selected) if selected else 0.0,
        float(top1.mean()),
        float(margin.mean()),
        float(entropy.mean()),
        float(np.mean(classes == 0)),
        min(1.0, len(text) / 16.0),
        sum(character.isdigit() for character in text) / length,
        sum(character.isalpha() for character in text) / length,
    ), dtype=np.float32)
    return text, statistics


def _geometry(scene: SceneSample, box: object) -> np.ndarray:
    width = float(box.width)
    height = float(box.height)
    center_x = float(box.left + box.right) / 2.0
    center_y = float(box.top + box.bottom) / 2.0
    plot_width = max(1.0, float(scene.plot.width))
    plot_height = max(1.0, float(scene.plot.height))
    return np.asarray((
        width / scene.raster.shape[1], height / scene.raster.shape[0],
        min(8.0, width / max(1.0, height)) / 8.0,
        center_x / scene.raster.shape[1], center_y / scene.raster.shape[0],
        width / plot_width, height / plot_height,
        (center_x - scene.plot.left) / plot_width,
        (center_y - scene.plot.top) / plot_height,
    ), dtype=np.float32)


def _morphology(scene: SceneSample, box: object) -> np.ndarray:
    crop = scene.raster[max(0, box.top):min(scene.raster.shape[0], box.bottom),
                        max(0, box.left):min(scene.raster.shape[1], box.right)]
    if not crop.size:
        return np.zeros(4, dtype=np.float32)
    ink = 1.0 - crop.astype(np.float32) / 255.0
    threshold = ink >= max(0.08, float(ink.mean()))
    return np.asarray((
        float(ink.mean()), float(ink.std()),
        float(np.mean(np.any(threshold, axis=1))),
        float(np.mean(np.any(threshold, axis=0))),
    ), dtype=np.float32)


def extract_features(
    scenes: tuple[SceneSample, ...], detector_runner: Runner, recognizer_runner: Runner,
    alphabet: str, *, mode: Mode, negative_cap_per_scene: int = 4,
    recognition_batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, tuple[ProposalRecord, ...], dict[str, object]]:
    detector_inputs = sha256()
    detector_outputs = sha256()
    recognizer_inputs = sha256()
    recognizer_outputs = sha256()
    pending: list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    recognition_tensors: list[np.ndarray] = []
    detector_calls = 0
    for scene_index, scene in enumerate(scenes):
        candidates = proposals(scene.raster)
        targets, _ = proposal_targets(scene, candidates)
        values = np.stack([encode_proposal(scene.raster, item, scene.plot) for item in candidates]).astype(np.float32)
        detector_inputs.update(np.ascontiguousarray(values).tobytes(order="C"))
        logits = np.asarray(detector_runner(values), dtype=np.float32)
        detector_calls += 1
        if logits.shape != (len(candidates), 2 + len(ROLE_ORDER)) or not np.isfinite(logits).all():
            raise RuntimeError("OCR V20 detector returned invalid logits")
        detector_outputs.update(np.ascontiguousarray(logits).tobytes(order="C"))
        detector_probability = _softmax(logits[:, :2])[:, 1]
        if mode == "train":
            positive = np.flatnonzero(targets == 1).tolist()
            negatives = np.flatnonzero(targets == 0).tolist()
            negatives.sort(key=lambda index: (-float(detector_probability[index]), index))
            selected = positive + negatives[:negative_cap_per_scene]
        else:
            selected = [index for index, value in enumerate(detector_probability) if float(value) >= DETECTOR_FLOOR]
        image = Image.fromarray(scene.raster, mode="L")
        for candidate_index in selected:
            candidate = candidates[candidate_index]
            matches = [
                truth_index for truth_index, truth in enumerate(scene.truths)
                if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
            ]
            truth_index = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index].box)) if matches else -1
            role_probabilities = _softmax(logits[candidate_index:candidate_index + 1, 2:])[0]
            detector_features = np.asarray((
                detector_probability[candidate_index],
                np.clip((logits[candidate_index, 1] - logits[candidate_index, 0]) / 8.0, -1.0, 1.0),
            ), dtype=np.float32)
            pending.append((scene_index, candidate_index, truth_index, detector_features, role_probabilities, logits[candidate_index]))
            box = candidate.box
            recognition_tensors.append(recognition_tensor(image.crop((box.left, box.top, box.right, box.bottom)))[0])
    predictions: list[tuple[str, np.ndarray]] = []
    recognition_batches = 0
    for start in range(0, len(recognition_tensors), recognition_batch_size):
        batch = np.stack(recognition_tensors[start:start + recognition_batch_size]).astype(np.float32)
        recognizer_inputs.update(np.ascontiguousarray(batch).tobytes(order="C"))
        output = np.asarray(recognizer_runner(batch), dtype=np.float32)
        recognition_batches += 1
        recognizer_outputs.update(np.ascontiguousarray(output).tobytes(order="C"))
        probabilities = _recognition_probabilities(output)
        if probabilities.shape[0] != len(batch) or probabilities.shape[2] != len(alphabet) + 1:
            raise RuntimeError("OCR V20 recognizer output contract changed")
        predictions.extend(_decode_stats(row, alphabet) for row in probabilities)
    if len(predictions) != len(pending):
        raise RuntimeError("OCR V20 proposal and recognition streams diverged")
    features: list[np.ndarray] = []
    labels: list[int] = []
    records: list[ProposalRecord] = []
    for item, prediction in zip(pending, predictions, strict=True):
        scene_index, candidate_index, truth_index, detector_features, roles, logits = item
        scene = scenes[scene_index]
        candidate = proposals(scene.raster)[candidate_index]
        text, ctc = prediction
        vector = np.concatenate((detector_features, roles, ctc, _geometry(scene, candidate.box), _morphology(scene, candidate.box))).astype(np.float32)
        if vector.shape != (FEATURE_COUNT,) or not np.isfinite(vector).all():
            raise RuntimeError("OCR V20 proposal feature contract changed")
        features.append(vector)
        labels.append(int(truth_index >= 0))
        records.append(ProposalRecord(
            scene_index, candidate_index, truth_index, text,
            ROLE_ORDER[int(np.argmax(logits[2:]))],
        ))
    array = np.stack(features).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    feature_label_stream = sha256()
    feature_label_stream.update(np.ascontiguousarray(array).tobytes(order="C"))
    feature_label_stream.update(label_array.tobytes(order="C"))
    evidence: dict[str, object] = {
        "scene_count": len(scenes), "proposal_count": len(array),
        "positive_proposal_count": int(label_array.sum()),
        "negative_proposal_count": int(len(label_array) - label_array.sum()),
        "detector_inference_calls": detector_calls,
        "recognizer_region_calls": len(recognition_tensors),
        "recognizer_batch_calls": recognition_batches,
        "detector_input_tensor_stream_sha256": detector_inputs.hexdigest(),
        "detector_output_tensor_stream_sha256": detector_outputs.hexdigest(),
        "recognizer_input_tensor_stream_sha256": recognizer_inputs.hexdigest(),
        "recognizer_output_tensor_stream_sha256": recognizer_outputs.hexdigest(),
        "feature_label_stream_sha256": feature_label_stream.hexdigest(),
        "direct_stored_fixture_byte_execution": True,
    }
    return array, label_array, tuple(records), evidence


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[right_index] + 1,
                               previous[right_index - 1] + (left_character != right_character)))
        previous = current
    return previous[-1]


def evaluate_thresholds(
    scenes: tuple[SceneSample, ...], records: tuple[ProposalRecord, ...],
    logits: np.ndarray, thresholds: tuple[float, ...], evidence: dict[str, object],
) -> list[dict[str, object]]:
    if logits.shape != (len(records), 2):
        raise RuntimeError("OCR V20 calibrator output contract changed")
    probabilities = _softmax(logits)[:, 1]
    comparisons: list[dict[str, object]] = []
    truth_count = sum(len(scene.truths) for scene in scenes)
    for threshold in thresholds:
        exact_scenes = true_positives = false_positives = false_negatives = duplicates = 0
        recognition_exact = edits = characters = correct_roles = 0
        per_role = {role: {"correct": 0, "total": 0} for role in ROLE_ORDER}
        for scene_index, scene in enumerate(scenes):
            accepted = [
                record for record, probability in zip(records, probabilities, strict=True)
                if record.scene_index == scene_index and float(probability) >= threshold
            ]
            matched: set[int] = set()
            predicted: dict[int, ProposalRecord] = {}
            scene_fp = scene_dup = scene_roles = 0
            for record in accepted:
                if record.truth_index < 0:
                    scene_fp += 1
                elif record.truth_index in matched:
                    scene_dup += 1
                else:
                    matched.add(record.truth_index)
                    predicted[record.truth_index] = record
                    if record.predicted_role == scene.truths[record.truth_index].role:
                        correct_roles += 1
                        scene_roles += 1
                        per_role[record.predicted_role]["correct"] += 1
            for truth_index, truth in enumerate(scene.truths):
                per_role[truth.role]["total"] += 1
                text = predicted[truth_index].predicted_text if truth_index in predicted else ""
                recognition_exact += int(text == truth.text)
                edits += _edit_distance(text, truth.text)
                characters += len(truth.text)
            scene_fn = len(scene.truths) - len(matched)
            true_positives += len(matched)
            false_positives += scene_fp
            false_negatives += scene_fn
            duplicates += scene_dup
            if scene_fp == scene_fn == scene_dup == 0 and scene_roles == len(scene.truths):
                exact_scenes += 1
        metrics: dict[str, object] = {
            "scene_count": len(scenes), "truth_region_count": truth_count,
            "exact_scene_count": exact_scenes, "true_positives": true_positives,
            "false_positives": false_positives, "false_negatives": false_negatives,
            "duplicate_region_count": duplicates, "prohibited_structure_hits": false_positives,
            "recognition_exact_count": recognition_exact,
            "recognition_exact": recognition_exact / truth_count,
            "character_error_count": edits, "truth_character_count": characters,
            "character_error_rate": edits / characters,
            "role_accuracy": correct_roles / truth_count,
            "per_role_accuracy": {role: value["correct"] / value["total"] for role, value in per_role.items()},
            **evidence,
        }
        comparisons.append({"threshold": threshold, "metrics": metrics})
    return comparisons


def metrics_pass(metrics: dict[str, object]) -> bool:
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


def select_robust_window(comparisons: list[dict[str, object]]) -> tuple[dict[str, object], tuple[float, ...]] | None:
    runs: list[list[int]] = []
    active: list[int] = []
    for index, comparison in enumerate(comparisons):
        if metrics_pass(comparison["metrics"]):
            active.append(index)
        else:
            if active:
                runs.append(active)
            active = []
    if active:
        runs.append(active)
    eligible = [run for run in runs if len(run) >= ROBUST_THRESHOLD_RUN_LENGTH]
    if not eligible:
        return None
    run = max(eligible, key=lambda value: len(value))
    return comparisons[run[len(run) // 2]], tuple(float(comparisons[index]["threshold"]) for index in run)


__all__ = ["ProposalRecord", "evaluate_thresholds", "extract_features", "metrics_pass", "select_robust_window"]

