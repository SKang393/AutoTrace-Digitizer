# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P3 center refinement for quantization-sensitive geometry consensus."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

from ml.markers.center.line_aware_v1.dataset import LineAwareScene
from ml.markers.center.line_aware_v1.pipeline import (
    MATCH_TOLERANCE,
    PROPOSAL_STRIDE,
    MarkerPrediction,
    ProposalBatch,
    _center_is_unmasked,
    _marker_geometry_consensus,
    extract_proposals,
)


REFINEMENT_OFFSETS = (-1.0, 0.0, 1.0)


def _refine_geometry_center(
    scene: LineAwareScene,
    x: float,
    y: float,
    radius: float,
) -> tuple[float, float] | None:
    if not _center_is_unmasked(scene, x, y):
        return None
    if _marker_geometry_consensus(scene, x, y, radius):
        return x, y
    candidates: list[tuple[float, float, float, float, float, float, float]] = []
    for dy in REFINEMENT_OFFSETS:
        for dx in REFINEMENT_OFFSETS:
            refined_x, refined_y = x + dx, y + dy
            if (
                _center_is_unmasked(scene, refined_x, refined_y)
                and _marker_geometry_consensus(scene, refined_x, refined_y, radius)
            ):
                candidates.append(
                    (dx * dx + dy * dy, abs(dy), abs(dx), dy, dx, refined_x, refined_y)
                )
    if not candidates:
        return None
    *_, refined_x, refined_y = min(candidates)
    return refined_x, refined_y


def postprocess_predictions(
    scene: LineAwareScene,
    proposals: ProposalBatch,
    output: np.ndarray,
    *,
    threshold: float,
) -> tuple[MarkerPrediction, ...]:
    if output.shape != (len(proposals.patches), 4):
        raise ValueError("Radial P3 candidate output must be NC [candidate_count,4]")
    candidates: list[MarkerPrediction] = []
    for index in np.flatnonzero(output[:, 0] >= threshold):
        base_x, base_y = proposals.coordinates[index].tolist()
        x = float(base_x + (output[index, 1] * PROPOSAL_STRIDE))
        y = float(base_y + (output[index, 2] * PROPOSAL_STRIDE))
        radius = float(np.clip(output[index, 3], 2.5, 8.0))
        refined = _refine_geometry_center(scene, x, y, radius)
        if refined is None:
            continue
        x, y = refined
        candidates.append(MarkerPrediction(x, y, radius, float(output[index, 0])))
    accepted: list[MarkerPrediction] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.y, item.x)):
        if any(
            math.hypot(candidate.x - current.x, candidate.y - current.y)
            < max(5.0, 1.25 * max(candidate.radius, current.radius))
            for current in accepted
        ):
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: (item.y, item.x, -item.confidence)))


def infer_scene(
    scene: LineAwareScene,
    runner: Callable[[np.ndarray], np.ndarray],
    *,
    threshold: float,
) -> tuple[MarkerPrediction, ...]:
    proposals = extract_proposals(scene.tensor)
    output = runner(proposals.patches.numpy().astype(np.float32, copy=False))
    return postprocess_predictions(scene, proposals, output, threshold=threshold)


def evaluate_scenes(
    scenes: Iterable[LineAwareScene],
    runner: Callable[[np.ndarray], np.ndarray],
    *,
    threshold: float,
) -> dict[str, object]:
    per_scene: list[dict[str, object]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "duplicates": 0, "hits": 0, "exact": 0}
    kinds: dict[str, int] = {}
    for scene in scenes:
        predictions = infer_scene(scene, runner, threshold=threshold)
        edges = sorted(
            (math.hypot(pred.x - tx, pred.y - ty), pi, ti)
            for pi, pred in enumerate(predictions)
            for ti, (tx, ty) in enumerate(scene.centers)
            if math.hypot(pred.x - tx, pred.y - ty) <= MATCH_TOLERANCE
        )
        used_predictions: set[int] = set()
        used_truths: set[int] = set()
        for _, prediction_index, truth_index in edges:
            if prediction_index not in used_predictions and truth_index not in used_truths:
                used_predictions.add(prediction_index)
                used_truths.add(truth_index)
        duplicates = sum(
            1
            for prediction_index, prediction in enumerate(predictions)
            if prediction_index not in used_predictions
            and any(
                math.hypot(prediction.x - truth_x, prediction.y - truth_y) <= MATCH_TOLERANCE
                for truth_x, truth_y in scene.centers
            )
        )
        hits: dict[str, int] = {}
        for prediction in predictions:
            for point in scene.prohibited:
                if math.hypot(prediction.x - point.x, prediction.y - point.y) <= MATCH_TOLERANCE:
                    hits[point.kind] = hits.get(point.kind, 0) + 1
                    kinds[point.kind] = kinds.get(point.kind, 0) + 1
        true_positives = len(used_predictions)
        false_positives = len(predictions) - true_positives
        false_negatives = len(scene.centers) - true_positives
        exact = false_positives == 0 and false_negatives == 0 and duplicates == 0 and not hits
        totals["tp"] += true_positives
        totals["fp"] += false_positives
        totals["fn"] += false_negatives
        totals["duplicates"] += duplicates
        totals["hits"] += sum(hits.values())
        totals["exact"] += int(exact)
        per_scene.append({
            "scene_id": scene.scene_id,
            "truth_count": len(scene.centers),
            "prediction_count": len(predictions),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "duplicate_count": duplicates,
            "prohibited_hits": hits,
            "exact": exact,
        })
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    return {
        "scene_count": len(per_scene),
        "exact_scene_count": totals["exact"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "duplicate_count": totals["duplicates"],
        "prohibited_structure_hits": totals["hits"],
        "prohibited_hits_by_kind": dict(sorted(kinds.items())),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "per_scene": per_scene,
    }


__all__ = ["evaluate_scenes", "infer_scene", "postprocess_predictions"]
