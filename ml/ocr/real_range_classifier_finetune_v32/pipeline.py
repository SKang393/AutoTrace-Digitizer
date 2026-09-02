# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Canonical aggregate metrics for V32 proposal classification."""

from __future__ import annotations

from hashlib import sha256
from typing import Callable

import numpy as np

from ml.ocr.component_context_detector_v7.dataset import box_iou, encode_proposal, proposals

from .dataset import SceneSample


def _maximum_cardinality(predicted: list[object], truths: tuple[object, ...]) -> int:
    edges = [[index for index, truth in enumerate(truths) if box_iou(item.box, truth) >= 0.5] for item in predicted]
    owners = [-1] * len(truths)

    def visit(predicted_index: int, seen: set[int]) -> bool:
        for truth_index in edges[predicted_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            if owners[truth_index] == -1 or visit(owners[truth_index], seen):
                owners[truth_index] = predicted_index
                return True
        return False

    return sum(int(visit(index, set())) for index in range(len(predicted)))


def evaluate_scenes(scenes: tuple[SceneSample, ...], runner: Callable[[np.ndarray], np.ndarray], threshold: float = 0.82) -> dict[str, object]:
    truth_count = true_positives = false_positives = false_negatives = 0
    input_digest = sha256()
    output_digest = sha256()
    inference_calls = 0
    dimensions: dict[str, list[int]] = {}
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        input_digest.update(values.tobytes(order="C"))
        logits = np.asarray(runner(values), dtype=np.float32)
        output_digest.update(logits.tobytes(order="C"))
        inference_calls += 1
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities = probabilities[:, 1] / probabilities.sum(axis=1)
        accepted = [item for item, probability in zip(candidates, probabilities, strict=True) if float(probability) >= threshold]
        matched = _maximum_cardinality(accepted, scene.truths)
        current_truth = len(scene.truths)
        current_fp = len(accepted) - matched
        current_fn = current_truth - matched
        truth_count += current_truth
        true_positives += matched
        false_positives += current_fp
        false_negatives += current_fn
        row = dimensions.setdefault(f"{scene.raster.shape[1]}x{scene.raster.shape[0]}", [0, 0, 0, 0])
        row[0] += current_truth; row[1] += matched; row[2] += current_fp; row[3] += current_fn
    return {
        "scene_count": len(scenes),
        "truth_region_count": truth_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / max(1, true_positives + false_positives),
        "recall": true_positives / max(1, truth_count),
        "threshold": threshold,
        "inference_calls": inference_calls,
        "input_tensor_stream_sha256": input_digest.hexdigest(),
        "output_tensor_stream_sha256": output_digest.hexdigest(),
        "by_dimension": {
            key: {"truth_region_count": row[0], "true_positives": row[1], "false_positives": row[2], "false_negatives": row[3]}
            for key, row in sorted(dimensions.items())
        },
    }
