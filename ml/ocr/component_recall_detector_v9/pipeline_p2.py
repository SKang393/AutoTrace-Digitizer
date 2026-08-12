# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-inference-per-scene threshold evaluation for V9 training selection."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from .dataset import SceneSample, box_iou, encode_proposal, proposals
from .pipeline import Runner
from .protocol import TRUTH_MATCH_IOU_MINIMUM


def evaluate_thresholds(
    scenes: tuple[SceneSample, ...], runner: Runner, thresholds: tuple[float, ...]
) -> list[dict[str, object]]:
    cached: list[tuple[SceneSample, tuple[object, ...], np.ndarray]] = []
    input_hasher = sha256()
    output_hasher = sha256()
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = np.asarray(runner(values), dtype=np.float32)
        if logits.shape != (len(candidates), 2) or not np.isfinite(logits).all():
            raise RuntimeError("OCR V9 P2 runner returned an invalid tensor")
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent[:, 1] / exponent.sum(axis=1)
        input_hasher.update(values.tobytes(order="C"))
        output_hasher.update(logits.tobytes(order="C"))
        cached.append((scene, candidates, probabilities))
    input_sha256 = input_hasher.hexdigest()
    output_sha256 = output_hasher.hexdigest()
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        cases: list[dict[str, object]] = []
        true_positives = false_positives = false_negatives = duplicates = 0
        for scene, candidates, probabilities in cached:
            accepted = [
                item for item, probability in zip(candidates, probabilities, strict=True)
                if probability >= threshold
            ]
            matched_truths: set[int] = set()
            scene_true_positives = scene_false_positives = scene_duplicates = 0
            for candidate in accepted:
                matches = [
                    index for index, truth in enumerate(scene.truths)
                    if box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM
                ]
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
            cases.append({
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
            })
        results.append({
            "threshold": threshold,
            "metrics": {
                "scene_count": len(scenes),
                "exact_scene_count": sum(int(item["exact"]) for item in cases),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "duplicate_region_count": duplicates,
                "prohibited_structure_hits": false_positives,
                "direct_execution_inference_calls": len(scenes),
                "direct_execution_input_tensor_stream_sha256": input_sha256,
                "direct_execution_output_tensor_stream_sha256": output_sha256,
                "cases": cases,
            },
        })
    return results


__all__ = ["evaluate_thresholds"]
