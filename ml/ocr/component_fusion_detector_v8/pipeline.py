# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic OCR V8 proposal evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .dataset import SceneSample, box_iou, encode_proposal, proposals
from .protocol import TRUTH_MATCH_IOU_MINIMUM


Runner = Callable[[np.ndarray], np.ndarray]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def evaluate_scenes(scenes: tuple[SceneSample, ...], runner: Runner, threshold: float) -> dict[str, object]:
    case_results: list[dict[str, object]] = []
    true_positives = false_positives = false_negatives = duplicates = 0
    input_hasher = __import__("hashlib").sha256()
    output_hasher = __import__("hashlib").sha256()
    inference_calls = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = np.asarray(runner(values), dtype=np.float32)
        inference_calls += 1
        input_hasher.update(values.tobytes(order="C"))
        output_hasher.update(logits.tobytes(order="C"))
        if logits.shape != (len(candidates), 2) or not np.isfinite(logits).all():
            raise RuntimeError("OCR V8 runner returned an invalid tensor")
        probabilities = _softmax(logits)[:, 1]
        accepted = [item for item, probability in zip(candidates, probabilities, strict=True) if probability >= threshold]
        matched_truths: set[int] = set()
        scene_true_positives = scene_false_positives = scene_duplicates = 0
        for candidate in accepted:
            matches = [index for index, truth in enumerate(scene.truths) if box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM]
            if not matches:
                scene_false_positives += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index]))
            if best in matched_truths:
                scene_duplicates += 1
            else:
                matched_truths.add(best)
                scene_true_positives += 1
        scene_false_negatives = len(scene.truths) - len(matched_truths)
        true_positives += scene_true_positives
        false_positives += scene_false_positives
        false_negatives += scene_false_negatives
        duplicates += scene_duplicates
        exact = scene_false_positives == scene_false_negatives == scene_duplicates == 0
        case_results.append(
            {
                "scene_id": scene.scene_id,
                "truth_region_count": len(scene.truths),
                "proposal_count": len(candidates),
                "accepted_region_count": len(accepted),
                "true_positives": scene_true_positives,
                "false_positives": scene_false_positives,
                "false_negatives": scene_false_negatives,
                "duplicate_region_count": scene_duplicates,
                "prohibited_structure_hits": scene_false_positives,
                "exact": exact,
            }
        )
    return {
        "scene_count": len(scenes),
        "exact_scene_count": sum(int(item["exact"]) for item in case_results),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "duplicate_region_count": duplicates,
        "prohibited_structure_hits": false_positives,
        "direct_execution_inference_calls": inference_calls,
        "direct_execution_input_tensor_stream_sha256": input_hasher.hexdigest(),
        "direct_execution_output_tensor_stream_sha256": output_hasher.hexdigest(),
        "cases": case_results,
    }


__all__ = ["Runner", "evaluate_scenes"]
