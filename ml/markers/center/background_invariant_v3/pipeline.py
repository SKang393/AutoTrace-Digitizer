# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Median-background invariant inference for the exact radial marker payload."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor

from ml.markers.center.line_aware_v1.dataset import LineAwareScene
from ml.markers.center.line_aware_v1.pipeline import (
    MATCH_TOLERANCE,
    MarkerPrediction,
    ProposalBatch,
    extract_proposals,
)
from ml.markers.center.runtime_consistency_v2.pipeline_p2 import postprocess_predictions


PREPROCESS_REVISION = "patch-ink-median-background-subtraction-v1"
POSTPROCESS_REVISION = "radial-local-consensus-calibration-v2"
MINIMUM_CENTER_SEPARATION = 6.5


def normalize_proposal_patches(patches: Tensor) -> Tensor:
    if patches.ndim != 4 or tuple(patches.shape[1:]) != (3, 33, 33):
        raise ValueError("Background-invariant preprocessing requires NCHW [N,3,33,33]")
    if len(patches) == 0:
        return patches.clone()
    result = patches.clone()
    background = torch.median(result[:, 0].flatten(1), dim=1).values[:, None, None]
    result[:, 0] = torch.clamp(result[:, 0] - background, 0.0, 1.0)
    return result


def extract_background_invariant_proposals(tensor: Tensor) -> ProposalBatch:
    proposals = extract_proposals(tensor)
    return ProposalBatch(normalize_proposal_patches(proposals.patches), proposals.coordinates)


def infer_scene(
    scene: LineAwareScene,
    runner: Callable[[np.ndarray], np.ndarray],
    *,
    threshold: float,
) -> tuple[MarkerPrediction, ...]:
    proposals = extract_background_invariant_proposals(scene.tensor)
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
        per_scene.append(
            {
                "scene_id": scene.scene_id,
                "truth_count": len(scene.centers),
                "prediction_count": len(predictions),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "duplicate_count": duplicates,
                "prohibited_hits": hits,
                "exact": exact,
            }
        )
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


__all__ = [
    "MINIMUM_CENTER_SEPARATION",
    "POSTPROCESS_REVISION",
    "PREPROCESS_REVISION",
    "evaluate_scenes",
    "extract_background_invariant_proposals",
    "infer_scene",
    "normalize_proposal_patches",
]
