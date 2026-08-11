# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Proposal classification metrics for OCR component-region V6."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .dataset import SceneSample, box_iou, encode_proposal, proposals
from .protocol import TRUTH_MATCH_IOU_MINIMUM


Runner = Callable[[np.ndarray], np.ndarray]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def evaluate_scenes(scenes: tuple[SceneSample, ...], runner: Runner, threshold: float) -> dict[str, object]:
    exact_scenes = true_positives = false_positives = false_negatives = duplicates = 0
    inference_calls = 0
    records: list[dict[str, object]] = []
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = np.asarray(runner(values), dtype=np.float32)
        inference_calls += 1
        if logits.shape != (len(candidates), 2) or not np.isfinite(logits).all():
            raise RuntimeError("OCR V6 runner returned an invalid proposal-logit tensor")
        probabilities = _softmax(logits)[:, 1]
        accepted = [item for item, probability in zip(candidates, probabilities, strict=True) if probability >= threshold]
        matched_truths: set[int] = set()
        scene_duplicates = 0
        scene_false = 0
        for candidate in accepted:
            matches = [index for index, truth in enumerate(scene.truths) if box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM]
            if not matches:
                scene_false += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index]))
            if best in matched_truths:
                scene_duplicates += 1
            else:
                matched_truths.add(best)
        scene_missed = len(scene.truths) - len(matched_truths)
        true_positives += len(matched_truths)
        false_positives += scene_false
        false_negatives += scene_missed
        duplicates += scene_duplicates
        is_exact = scene_false == 0 and scene_missed == 0 and scene_duplicates == 0 and len(accepted) == len(scene.truths)
        exact_scenes += int(is_exact)
        records.append(
            {
                "scene_id": scene.scene_id,
                "truth_count": len(scene.truths),
                "proposal_count": len(candidates),
                "accepted_count": len(accepted),
                "true_positives": len(matched_truths),
                "false_positives": scene_false,
                "false_negatives": scene_missed,
                "duplicates": scene_duplicates,
                "exact": is_exact,
            }
        )
    return {
        "scene_count": len(scenes),
        "exact_scene_count": exact_scenes,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "duplicate_region_count": duplicates,
        "prohibited_structure_hits": false_positives,
        "inference_calls": inference_calls,
        "records": records,
    }

