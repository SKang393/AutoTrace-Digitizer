# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P3 selection metrics with the preregistered structural guard."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ml.ocr.component_region_detector_v6.dataset import box_iou
from .dataset import SceneSample, encode_proposal, proposals
from .protocol import ROLE_ORDER, TRUTH_MATCH_IOU_MINIMUM
from .structural_guard import is_rejected_structure


Runner = Callable[[np.ndarray], np.ndarray]


def evaluate_thresholds(
    scenes: tuple[SceneSample, ...], runner: Runner, thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    cached = []
    guarded_proposals = guarded_truth_matches = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        if not candidates:
            raise RuntimeError("OCR V12 P3 scene produced no proposals")
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = np.asarray(runner(values), dtype=np.float32)
        if logits.shape != (len(candidates), 2 + len(ROLE_ORDER)) or not np.isfinite(logits).all():
            raise RuntimeError("OCR V12 P3 runner returned invalid logits")
        shifted = logits[:, :2] - logits[:, :2].max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials[:, 1] / exponentials.sum(axis=1)
        guarded = np.asarray([is_rejected_structure(candidate) for candidate in candidates], dtype=np.bool_)
        guarded_proposals += int(guarded.sum())
        guarded_truth_matches += sum(
            any(box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM for truth in scene.truths)
            for candidate, rejected in zip(candidates, guarded, strict=True)
            if rejected
        )
        cached.append((scene, candidates, probabilities, np.argmax(logits[:, 2:], axis=1), guarded))

    comparisons: list[dict[str, object]] = []
    for threshold in thresholds:
        exact_scenes = true_positives = false_positives = false_negatives = duplicates = 0
        correct_roles = role_truths = guarded_accepted = 0
        per_role = {role: {"correct": 0, "total": 0} for role in ROLE_ORDER}
        for scene, candidates, probabilities, roles, guarded in cached:
            guarded_accepted += sum(
                bool(rejected and probability >= threshold)
                for rejected, probability in zip(guarded, probabilities, strict=True)
            )
            accepted = [
                (index, candidate)
                for index, (candidate, probability, rejected) in enumerate(
                    zip(candidates, probabilities, guarded, strict=True)
                )
                if probability >= threshold and not rejected
            ]
            matched_truths: set[int] = set()
            scene_fp = scene_dup = scene_role_correct = 0
            for candidate_index, candidate in accepted:
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
                role = scene.truths[best].role
                predicted = ROLE_ORDER[int(roles[candidate_index])]
                per_role[role]["total"] += 1
                role_truths += 1
                if predicted == role:
                    per_role[role]["correct"] += 1
                    correct_roles += 1
                    scene_role_correct += 1
            scene_fn = len(scene.truths) - len(matched_truths)
            true_positives += len(matched_truths)
            false_positives += scene_fp
            false_negatives += scene_fn
            duplicates += scene_dup
            if scene_fp == scene_fn == scene_dup == 0 and scene_role_correct == len(scene.truths):
                exact_scenes += 1
        comparisons.append({
            "threshold": threshold,
            "metrics": {
                "scene_count": len(scenes),
                "truth_region_count": sum(len(scene.truths) for scene in scenes),
                "exact_scene_count": exact_scenes,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "duplicate_region_count": duplicates,
                "prohibited_structure_hits": false_positives,
                "role_accuracy": correct_roles / role_truths if role_truths else 0.0,
                "per_role_accuracy": {
                    role: values["correct"] / values["total"] if values["total"] else 0.0
                    for role, values in per_role.items()
                },
                "structural_guard_proposal_count": guarded_proposals,
                "structural_guard_truth_match_count": guarded_truth_matches,
                "structural_guard_accepted_rejection_count": guarded_accepted,
                "direct_execution_inference_calls": len(scenes),
            },
        })
    return comparisons


__all__ = ["Runner", "evaluate_thresholds"]
