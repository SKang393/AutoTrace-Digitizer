# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Selection metrics for OCR V11 proposal and role outputs."""

from __future__ import annotations

from hashlib import sha256
from typing import Callable

import numpy as np

from ml.ocr.component_context_detector_v7.dataset import box_iou
from .dataset import SceneSample, encode_proposal, proposals
from .protocol import ROLE_ORDER, TRUTH_MATCH_IOU_MINIMUM


Runner = Callable[[np.ndarray], np.ndarray]


def evaluate_thresholds(
    scenes: tuple[SceneSample, ...], runner: Runner, thresholds: tuple[float, ...]
) -> list[dict[str, object]]:
    cached: list[tuple[SceneSample, tuple[object, ...], np.ndarray, np.ndarray]] = []
    input_hasher, output_hasher = sha256(), sha256()
    for scene in scenes:
        candidates = proposals(scene.raster)
        if not candidates:
            raise RuntimeError("OCR V11 scene produced no proposals")
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = np.asarray(runner(values), dtype=np.float32)
        if logits.shape != (len(candidates), 2 + len(ROLE_ORDER)) or not np.isfinite(logits).all():
            raise RuntimeError("OCR V11 runner returned an invalid tensor")
        proposal_logits = logits[:, :2]
        shifted = proposal_logits - proposal_logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent[:, 1] / exponent.sum(axis=1)
        roles = np.argmax(logits[:, 2:], axis=1)
        input_hasher.update(values.tobytes(order="C"))
        output_hasher.update(logits.tobytes(order="C"))
        cached.append((scene, candidates, probabilities, roles))
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        cases: list[dict[str, object]] = []
        true_positives = false_positives = false_negatives = duplicates = 0
        correct_roles = role_truths = 0
        per_role = {role: {"correct": 0, "total": 0} for role in ROLE_ORDER}
        for scene, candidates, probabilities, roles in cached:
            accepted = [
                (index, candidate) for index, (candidate, probability) in enumerate(zip(candidates, probabilities, strict=True))
                if probability >= threshold
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
            cases.append({
                "scene_id": scene.scene_id, "truth_region_count": len(scene.truths),
                "accepted_region_count": len(accepted), "true_positives": len(matched_truths),
                "false_positives": scene_fp, "false_negatives": scene_fn,
                "duplicate_region_count": scene_dup, "prohibited_structure_hits": scene_fp,
                "role_correct": scene_role_correct,
                "exact": scene_fp == scene_fn == scene_dup == 0 and scene_role_correct == len(scene.truths),
            })
        role_accuracy = correct_roles / role_truths if role_truths else 0.0
        per_role_accuracy = {
            role: values["correct"] / values["total"] if values["total"] else 0.0
            for role, values in per_role.items()
        }
        results.append({
            "threshold": threshold,
            "metrics": {
                "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
                "exact_scene_count": sum(int(item["exact"]) for item in cases),
                "true_positives": true_positives, "false_positives": false_positives,
                "false_negatives": false_negatives, "duplicate_region_count": duplicates,
                "prohibited_structure_hits": false_positives, "role_accuracy": role_accuracy,
                "per_role_accuracy": per_role_accuracy, "direct_execution_inference_calls": len(scenes),
                "direct_execution_input_tensor_stream_sha256": input_hasher.hexdigest(),
                "direct_execution_output_tensor_stream_sha256": output_hasher.hexdigest(), "cases": cases,
            },
        })
    return results


__all__ = ["Runner", "evaluate_thresholds"]

