# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Candidate proposal, target sampling, postprocessing, and exact-scene metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as functional

from ml.markers.center.candidate_level_v1.dataset import CandidateScene


PATCH_SIZE = 33
PROPOSAL_STRIDE = 4
INK_SUPPORT_WINDOW = 17
INK_SUPPORT_THRESHOLD = 0.12
POSITIVE_DISTANCE = 3.0
MASK_REJECTION_THRESHOLD = 0.35
MASK_SAMPLE_RADIUS = 2
MINIMUM_RADIUS = 2.5
MAXIMUM_RADIUS = 8.0
MINIMUM_SUPPRESSION_DISTANCE = 5.0
RADIUS_SUPPRESSION_SCALE = 1.25
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


def extract_proposals(tensor: Tensor) -> ProposalBatch:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Candidate proposal input must contain ink, text, and artifact channels")
    height, width = tensor.shape[1:]
    patches = functional.unfold(
        tensor.unsqueeze(0),
        kernel_size=PATCH_SIZE,
        padding=PATCH_SIZE // 2,
        stride=PROPOSAL_STRIDE,
    ).squeeze(0).transpose(0, 1).reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)
    support = functional.max_pool2d(
        tensor[0:1].unsqueeze(0),
        kernel_size=INK_SUPPORT_WINDOW,
        stride=PROPOSAL_STRIDE,
        padding=INK_SUPPORT_WINDOW // 2,
    ).flatten()
    grid_width = math.ceil(width / PROPOSAL_STRIDE)
    indices = torch.nonzero(support >= INK_SUPPORT_THRESHOLD, as_tuple=False).flatten()
    y = torch.div(indices, grid_width, rounding_mode="floor") * PROPOSAL_STRIDE
    x = torch.remainder(indices, grid_width) * PROPOSAL_STRIDE
    coordinates = torch.stack((x, y), dim=1).to(dtype=torch.float32)
    return ProposalBatch(patches.index_select(0, indices), coordinates)


def label_proposals(scene: CandidateScene, proposals: ProposalBatch) -> TrainingExamples:
    centers = torch.tensor(scene.centers, dtype=torch.float32)
    radii = torch.tensor(scene.radii, dtype=torch.float32)
    distances = torch.cdist(proposals.coordinates, centers)
    nearest_distance, nearest_index = distances.min(dim=1)
    labels = (nearest_distance <= POSITIVE_DISTANCE).to(dtype=torch.float32)
    offsets = (centers.index_select(0, nearest_index) - proposals.coordinates) / PROPOSAL_STRIDE
    target_radii = radii.index_select(0, nearest_index)
    return TrainingExamples(proposals.patches, labels, offsets, target_radii)


def sample_training_examples(
    scene: CandidateScene,
    *,
    maximum_negative_per_positive: int,
    generator: torch.Generator,
) -> TrainingExamples:
    proposals = extract_proposals(scene.tensor)
    all_examples = label_proposals(scene, proposals)
    positive_indices = torch.nonzero(all_examples.labels > 0.5, as_tuple=False).flatten()
    negative_indices = torch.nonzero(all_examples.labels <= 0.5, as_tuple=False).flatten()
    hard_negative_indices: list[int] = []
    coordinates = proposals.coordinates
    for item in scene.prohibited:
        distance = torch.sqrt(((coordinates - torch.tensor((item.x, item.y))) ** 2).sum(dim=1))
        hard_negative_indices.extend(torch.nonzero(distance <= 6.0, as_tuple=False).flatten().tolist())
    hard_negative = torch.tensor(sorted(set(hard_negative_indices)), dtype=torch.long)
    negative_budget = max(len(positive_indices) * maximum_negative_per_positive, len(hard_negative))
    remaining = torch.tensor(
        sorted(set(negative_indices.tolist()) - set(hard_negative.tolist())), dtype=torch.long
    )
    random_budget = max(0, negative_budget - len(hard_negative))
    if len(remaining) > random_budget:
        order = torch.randperm(len(remaining), generator=generator)[:random_budget]
        remaining = remaining.index_select(0, order)
    selected = torch.cat((positive_indices, hard_negative, remaining)).unique(sorted=True)
    return TrainingExamples(
        all_examples.patches.index_select(0, selected),
        all_examples.labels.index_select(0, selected),
        all_examples.offsets.index_select(0, selected),
        all_examples.radii.index_select(0, selected),
    )


def concatenate_examples(values: Iterable[TrainingExamples]) -> TrainingExamples:
    items = tuple(values)
    return TrainingExamples(
        torch.cat([item.patches for item in items]),
        torch.cat([item.labels for item in items]),
        torch.cat([item.offsets for item in items]),
        torch.cat([item.radii for item in items]),
    )


def _mask_rejected(scene: CandidateScene, x: float, y: float) -> bool:
    ix = int(round(x))
    iy = int(round(y))
    left = max(0, ix - MASK_SAMPLE_RADIUS)
    right = min(scene.tensor.shape[2], ix + MASK_SAMPLE_RADIUS + 1)
    top = max(0, iy - MASK_SAMPLE_RADIUS)
    bottom = min(scene.tensor.shape[1], iy + MASK_SAMPLE_RADIUS + 1)
    return bool(torch.max(scene.tensor[1:3, top:bottom, left:right]) >= MASK_REJECTION_THRESHOLD)


def postprocess_predictions(
    scene: CandidateScene,
    proposals: ProposalBatch,
    output: np.ndarray,
    *,
    threshold: float,
) -> tuple[MarkerPrediction, ...]:
    if output.shape != (len(proposals.patches), 4):
        raise ValueError("Candidate model output must be NC [candidate_count,4]")
    candidates: list[MarkerPrediction] = []
    for index in np.flatnonzero(output[:, 0] >= threshold):
        base_x, base_y = proposals.coordinates[index].tolist()
        x = float(base_x + (output[index, 1] * PROPOSAL_STRIDE))
        y = float(base_y + (output[index, 2] * PROPOSAL_STRIDE))
        radius = float(np.clip(output[index, 3], MINIMUM_RADIUS, MAXIMUM_RADIUS))
        if x < 0 or y < 0 or x >= scene.tensor.shape[2] or y >= scene.tensor.shape[1]:
            continue
        if _mask_rejected(scene, x, y):
            continue
        candidates.append(MarkerPrediction(x, y, radius, float(output[index, 0])))
    accepted: list[MarkerPrediction] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.y, item.x)):
        if any(
            math.hypot(candidate.x - existing.x, candidate.y - existing.y)
            < max(
                MINIMUM_SUPPRESSION_DISTANCE,
                RADIUS_SUPPRESSION_SCALE * max(candidate.radius, existing.radius),
            )
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: (item.y, item.x, -item.confidence)))


def infer_scene(
    scene: CandidateScene,
    runner: Callable[[np.ndarray], np.ndarray],
    *,
    threshold: float,
) -> tuple[MarkerPrediction, ...]:
    proposals = extract_proposals(scene.tensor)
    output = runner(proposals.patches.numpy().astype(np.float32, copy=False))
    return postprocess_predictions(scene, proposals, output, threshold=threshold)


def _maximum_matching(
    predictions: tuple[MarkerPrediction, ...],
    truths: tuple[tuple[float, float], ...],
) -> tuple[int, int]:
    edges = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, (truth_x, truth_y) in enumerate(truths):
            distance = math.hypot(prediction.x - truth_x, prediction.y - truth_y)
            if distance <= MATCH_TOLERANCE:
                edges.append((distance, prediction_index, truth_index))
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    for _, prediction_index, truth_index in sorted(edges):
        if prediction_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
    duplicate_count = sum(
        1
        for prediction in predictions
        if sum(
            math.hypot(prediction.x - truth_x, prediction.y - truth_y) <= MATCH_TOLERANCE
            for truth_x, truth_y in truths
        ) > 0 and prediction not in tuple(predictions[index] for index in used_predictions)
    )
    return len(used_predictions), duplicate_count


def evaluate_scenes(
    scenes: Iterable[CandidateScene],
    runner: Callable[[np.ndarray], np.ndarray],
    *,
    threshold: float,
) -> dict[str, object]:
    per_scene = []
    true_positives = false_positives = false_negatives = duplicates = prohibited_hits = 0
    exact_scenes = 0
    kinds: dict[str, int] = {}
    for scene in scenes:
        predictions = infer_scene(scene, runner, threshold=threshold)
        matched, scene_duplicates = _maximum_matching(predictions, scene.centers)
        scene_false_positives = len(predictions) - matched
        scene_false_negatives = len(scene.centers) - matched
        scene_hits: dict[str, int] = {}
        for prediction in predictions:
            for point in scene.prohibited:
                if math.hypot(prediction.x - point.x, prediction.y - point.y) <= MATCH_TOLERANCE:
                    scene_hits[point.kind] = scene_hits.get(point.kind, 0) + 1
                    kinds[point.kind] = kinds.get(point.kind, 0) + 1
        scene_exact = (
            scene_false_positives == 0
            and scene_false_negatives == 0
            and scene_duplicates == 0
            and not scene_hits
        )
        exact_scenes += int(scene_exact)
        true_positives += matched
        false_positives += scene_false_positives
        false_negatives += scene_false_negatives
        duplicates += scene_duplicates
        prohibited_hits += sum(scene_hits.values())
        per_scene.append(
            {
                "scene_id": scene.scene_id,
                "truth_count": len(scene.centers),
                "prediction_count": len(predictions),
                "true_positives": matched,
                "false_positives": scene_false_positives,
                "false_negatives": scene_false_negatives,
                "duplicate_count": scene_duplicates,
                "prohibited_hits": scene_hits,
                "exact": scene_exact,
            }
        )
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "scene_count": len(per_scene),
        "exact_scene_count": exact_scenes,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "duplicate_count": duplicates,
        "prohibited_structure_hits": prohibited_hits,
        "prohibited_hits_by_kind": dict(sorted(kinds.items())),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_scene": per_scene,
    }


__all__ = [
    "INK_SUPPORT_THRESHOLD",
    "MASK_REJECTION_THRESHOLD",
    "MarkerPrediction",
    "PATCH_SIZE",
    "PROPOSAL_STRIDE",
    "ProposalBatch",
    "TrainingExamples",
    "concatenate_examples",
    "evaluate_scenes",
    "extract_proposals",
    "infer_scene",
    "label_proposals",
    "postprocess_predictions",
    "sample_training_examples",
]
