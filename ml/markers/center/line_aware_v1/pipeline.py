# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Mask-consensus proposal, postprocessing, and exact-scene evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as functional

from ml.markers.center.line_aware_v1.dataset import LineAwareScene


PATCH_SIZE = 33
PROPOSAL_STRIDE = 4
INK_SUPPORT_WINDOW = 17
INK_SUPPORT_THRESHOLD = 0.11
MASK_REJECTION_THRESHOLD = 0.35
POSITIVE_DISTANCE = 3.0
MATCH_TOLERANCE = 5.0


@dataclass(frozen=True)
class ProposalBatch:
    patches: Tensor
    coordinates: Tensor


@dataclass(frozen=True)
class TrainingExamples:
    patches: Tensor
    labels: Tensor
    offsets: Tensor
    radii: Tensor


@dataclass(frozen=True)
class MarkerPrediction:
    x: float
    y: float
    radius: float
    confidence: float


def _window_max(values: Tensor, x: int, y: int, radius: int) -> float:
    left, right = max(0, x - radius), min(values.shape[1], x + radius + 1)
    top, bottom = max(0, y - radius), min(values.shape[0], y + radius + 1)
    return float(torch.max(values[top:bottom, left:right]))


def extract_proposals(tensor: Tensor) -> ProposalBatch:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Line-aware proposal input requires ink, text, and artifact channels")
    height, width = tensor.shape[1:]
    patches = functional.unfold(
        tensor.unsqueeze(0), kernel_size=PATCH_SIZE, padding=PATCH_SIZE // 2, stride=PROPOSAL_STRIDE
    ).squeeze(0).transpose(0, 1).reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)
    support = functional.max_pool2d(
        tensor[0:1].unsqueeze(0), kernel_size=INK_SUPPORT_WINDOW,
        stride=PROPOSAL_STRIDE, padding=INK_SUPPORT_WINDOW // 2,
    ).flatten()
    grid_width = math.ceil(width / PROPOSAL_STRIDE)
    eligible: list[int] = []
    for index in torch.nonzero(support >= INK_SUPPORT_THRESHOLD, as_tuple=False).flatten().tolist():
        y = (index // grid_width) * PROPOSAL_STRIDE
        x = (index % grid_width) * PROPOSAL_STRIDE
        if _window_max(tensor[1], x, y, 2) >= MASK_REJECTION_THRESHOLD:
            continue
        if _window_max(tensor[2], x, y, 2) >= MASK_REJECTION_THRESHOLD:
            continue
        eligible.append(index)
    indices = torch.tensor(eligible, dtype=torch.long)
    if not eligible:
        return ProposalBatch(patches[:0], torch.empty((0, 2), dtype=torch.float32))
    y = torch.div(indices, grid_width, rounding_mode="floor") * PROPOSAL_STRIDE
    x = torch.remainder(indices, grid_width) * PROPOSAL_STRIDE
    return ProposalBatch(patches.index_select(0, indices), torch.stack((x, y), dim=1).to(torch.float32))


def label_proposals(scene: LineAwareScene, proposals: ProposalBatch) -> TrainingExamples:
    centers = torch.tensor(scene.centers, dtype=torch.float32)
    radii = torch.tensor(scene.radii, dtype=torch.float32)
    distances = torch.cdist(proposals.coordinates, centers)
    nearest_distance, nearest_index = distances.min(dim=1)
    labels = (nearest_distance <= POSITIVE_DISTANCE).to(torch.float32)
    return TrainingExamples(
        proposals.patches,
        labels,
        (centers.index_select(0, nearest_index) - proposals.coordinates) / PROPOSAL_STRIDE,
        radii.index_select(0, nearest_index),
    )


def sample_training_examples(
    scene: LineAwareScene,
    *,
    maximum_negative_per_positive: int,
    generator: torch.Generator,
) -> TrainingExamples:
    proposals = extract_proposals(scene.tensor)
    examples = label_proposals(scene, proposals)
    positive = torch.nonzero(examples.labels > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(examples.labels <= 0.5, as_tuple=False).flatten()
    hard: set[int] = set()
    for item in scene.prohibited:
        point = torch.tensor((item.x, item.y), dtype=torch.float32)
        distance = torch.sqrt(((proposals.coordinates - point) ** 2).sum(dim=1))
        hard.update(torch.nonzero(distance <= 8.0, as_tuple=False).flatten().tolist())
    hard_indices = torch.tensor(sorted(hard.intersection(negative.tolist())), dtype=torch.long)
    remaining = torch.tensor(sorted(set(negative.tolist()) - hard), dtype=torch.long)
    budget = max(len(positive) * maximum_negative_per_positive, len(hard_indices))
    random_budget = max(0, budget - len(hard_indices))
    if len(remaining) > random_budget:
        remaining = remaining.index_select(0, torch.randperm(len(remaining), generator=generator)[:random_budget])
    selected = torch.cat((positive, hard_indices, remaining)).unique(sorted=True)
    return TrainingExamples(
        examples.patches.index_select(0, selected),
        examples.labels.index_select(0, selected),
        examples.offsets.index_select(0, selected),
        examples.radii.index_select(0, selected),
    )


def concatenate_examples(values: Iterable[TrainingExamples]) -> TrainingExamples:
    items = tuple(values)
    return TrainingExamples(
        torch.cat([item.patches for item in items]),
        torch.cat([item.labels for item in items]),
        torch.cat([item.offsets for item in items]),
        torch.cat([item.radii for item in items]),
    )


def _center_is_unmasked(scene: LineAwareScene, x: float, y: float) -> bool:
    ix, iy = int(round(x)), int(round(y))
    return (
        0 <= ix < scene.tensor.shape[2]
        and 0 <= iy < scene.tensor.shape[1]
        and _window_max(scene.tensor[1], ix, iy, 2) < MASK_REJECTION_THRESHOLD
        and _window_max(scene.tensor[2], ix, iy, 2) < MASK_REJECTION_THRESHOLD
    )


def _marker_geometry_consensus(scene: LineAwareScene, x: float, y: float, radius: float) -> bool:
    ix, iy = int(round(x)), int(round(y))
    ink = scene.tensor[0]
    ring_radius = max(3, int(round(radius)))
    points = (
        (ix - ring_radius, iy), (ix + ring_radius, iy),
        (ix, iy - ring_radius), (ix, iy + ring_radius),
        (ix - ring_radius, iy - ring_radius), (ix + ring_radius, iy - ring_radius),
        (ix - ring_radius, iy + ring_radius), (ix + ring_radius, iy + ring_radius),
    )
    support = sum(
        1 for px, py in points
        if 0 <= px < ink.shape[1] and 0 <= py < ink.shape[0] and float(ink[py, px]) >= 0.12
    )
    center_density = float(torch.mean(ink[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]))
    return support >= 3 or center_density >= 0.28


def postprocess_predictions(
    scene: LineAwareScene,
    proposals: ProposalBatch,
    output: np.ndarray,
    *,
    threshold: float,
) -> tuple[MarkerPrediction, ...]:
    if output.shape != (len(proposals.patches), 4):
        raise ValueError("Line-aware candidate output must be NC [candidate_count,4]")
    candidates: list[MarkerPrediction] = []
    for index in np.flatnonzero(output[:, 0] >= threshold):
        base_x, base_y = proposals.coordinates[index].tolist()
        x = float(base_x + (output[index, 1] * PROPOSAL_STRIDE))
        y = float(base_y + (output[index, 2] * PROPOSAL_STRIDE))
        radius = float(np.clip(output[index, 3], 2.5, 8.0))
        if not _center_is_unmasked(scene, x, y) or not _marker_geometry_consensus(scene, x, y, radius):
            continue
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
        for _, pi, ti in edges:
            if pi not in used_predictions and ti not in used_truths:
                used_predictions.add(pi)
                used_truths.add(ti)
        duplicates = sum(
            1 for pi, pred in enumerate(predictions)
            if pi not in used_predictions and any(
                math.hypot(pred.x - tx, pred.y - ty) <= MATCH_TOLERANCE for tx, ty in scene.centers
            )
        )
        hits: dict[str, int] = {}
        for prediction in predictions:
            for point in scene.prohibited:
                if math.hypot(prediction.x - point.x, prediction.y - point.y) <= MATCH_TOLERANCE:
                    hits[point.kind] = hits.get(point.kind, 0) + 1
                    kinds[point.kind] = kinds.get(point.kind, 0) + 1
        tp = len(used_predictions)
        fp, fn = len(predictions) - tp, len(scene.centers) - tp
        exact = fp == 0 and fn == 0 and duplicates == 0 and not hits
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["duplicates"] += duplicates
        totals["hits"] += sum(hits.values())
        totals["exact"] += int(exact)
        per_scene.append({
            "scene_id": scene.scene_id, "truth_count": len(scene.centers),
            "prediction_count": len(predictions), "true_positives": tp,
            "false_positives": fp, "false_negatives": fn, "duplicate_count": duplicates,
            "prohibited_hits": hits, "exact": exact,
        })
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    return {
        "scene_count": len(per_scene), "exact_scene_count": totals["exact"],
        "true_positives": totals["tp"], "false_positives": totals["fp"],
        "false_negatives": totals["fn"], "duplicate_count": totals["duplicates"],
        "prohibited_structure_hits": totals["hits"],
        "prohibited_hits_by_kind": dict(sorted(kinds.items())),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "per_scene": per_scene,
    }


__all__ = [
    "MarkerPrediction", "ProposalBatch", "TrainingExamples", "concatenate_examples",
    "evaluate_scenes", "extract_proposals", "infer_scene", "label_proposals",
    "postprocess_predictions", "sample_training_examples",
]
