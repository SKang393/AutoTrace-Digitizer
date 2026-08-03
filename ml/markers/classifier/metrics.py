# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Classification, confidence calibration, and embedding metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ClassificationMetrics:
    macro_f1: float
    accuracy: float
    expected_calibration_error: float
    per_class_f1: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "per_class_f1": list(self.per_class_f1),
        }


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    total = max(1, len(targets))
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            result += float(mask.sum()) / total * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def classification_metrics(probabilities: np.ndarray, targets: np.ndarray, class_count: int) -> ClassificationMetrics:
    predictions = probabilities.argmax(axis=1)
    per_class = []
    for label in range(class_count):
        true_positive = int(np.logical_and(predictions == label, targets == label).sum())
        false_positive = int(np.logical_and(predictions == label, targets != label).sum())
        false_negative = int(np.logical_and(predictions != label, targets == label).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return ClassificationMetrics(
        macro_f1=float(np.mean(per_class)),
        accuracy=float((predictions == targets).mean()),
        expected_calibration_error=expected_calibration_error(probabilities, targets),
        per_class_f1=tuple(float(value) for value in per_class),
    )


def binary_metrics(probabilities: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    truth = targets >= 0.5
    tp = int(np.logical_and(predictions, truth).sum())
    fp = int(np.logical_and(predictions, ~truth).sum())
    fn = int(np.logical_and(~predictions, truth).sum())
    tn = int(np.logical_and(~predictions, ~truth).sum())
    denominator = 2 * tp + fp + fn
    return {
        "f1": 0.0 if denominator == 0 else 2.0 * tp / denominator,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def supervised_embedding_loss(embedding: torch.Tensor, identities: torch.Tensor, margin: float = 0.20) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return embedding.sum() * 0.0
    similarity = embedding @ embedding.T
    diagonal = torch.eye(embedding.shape[0], dtype=torch.bool, device=embedding.device)
    same = identities[:, None].eq(identities[None, :]) & ~diagonal
    different = ~identities[:, None].eq(identities[None, :]) & ~diagonal
    positive = (1.0 - similarity[same]).mean() if same.any() else similarity.sum() * 0.0
    negative = functional.relu(similarity[different] - margin).mean() if different.any() else similarity.sum() * 0.0
    return positive + negative


def embedding_retrieval_accuracy(embeddings: np.ndarray, identities: np.ndarray) -> float:
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    return float((identities[nearest] == identities).mean())


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    candidates = (0.70, 0.85, 1.0, 1.15, 1.35, 1.60)
    return min(candidates, key=lambda value: float(functional.cross_entropy(logits / value, targets)))


__all__ = [
    "ClassificationMetrics",
    "binary_metrics",
    "classification_metrics",
    "embedding_retrieval_accuracy",
    "expected_calibration_error",
    "fit_temperature",
    "supervised_embedding_loss",
]
