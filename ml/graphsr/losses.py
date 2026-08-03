# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Scientific-fidelity losses for GraphSR-x2 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


MAX_ADVERSARIAL_WEIGHT = 0.01


@dataclass(frozen=True)
class LossWeights:
    reconstruction: float = 1.0
    edge: float = 0.20
    marker_center: float = 0.15
    ocr_consistency: float = 0.10
    adversarial: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative")
        if self.adversarial > MAX_ADVERSARIAL_WEIGHT:
            raise ValueError(f"adversarial weight must not exceed {MAX_ADVERSARIAL_WEIGHT}")


def _grayscale(value: Tensor) -> Tensor:
    if value.shape[1] == 1:
        return value
    if value.shape[1] != 3:
        raise ValueError("Loss images must have one grayscale or three RGB channels")
    weights = value.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (value * weights).sum(dim=1, keepdim=True)


def _spatial_gradients(value: Tensor) -> tuple[Tensor, Tensor]:
    gray = _grayscale(value)
    kernel_x = gray.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(2, 3)
    return (
        functional.conv2d(gray, kernel_x, padding=1),
        functional.conv2d(gray, kernel_y, padding=1),
    )


def _normalized_mask(mask: Tensor | None, reference: Tensor, name: str) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 4 or mask.shape[0] != reference.shape[0] or mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError(f"{name} must be shaped [N, 1, output_height, output_width]")
    if mask.shape[1] != 1:
        raise ValueError(f"{name} must have one channel")
    if not mask.is_floating_point():
        mask = mask.float()
    return mask.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)


def charbonnier_loss(prediction: Tensor, target: Tensor, epsilon: float = 1e-3) -> Tensor:
    difference = prediction - target
    return torch.sqrt(difference.square() + epsilon * epsilon).mean()


def edge_consistency_loss(prediction: Tensor, target: Tensor) -> Tensor:
    predicted_x, predicted_y = _spatial_gradients(prediction)
    target_x, target_y = _spatial_gradients(target)
    return 0.5 * (
        functional.l1_loss(predicted_x, target_x) + functional.l1_loss(predicted_y, target_y)
    )


def marker_center_consistency_loss(
    prediction: Tensor,
    target: Tensor,
    marker_center_map: Tensor | None,
) -> Tensor:
    """Penalize movement of local ink mass around labelled marker centers."""

    mask = _normalized_mask(marker_center_map, prediction, "marker_center_map")
    if mask is None or not bool(torch.any(mask > 0)):
        return prediction.sum() * 0.0
    height, width = prediction.shape[-2:]
    predicted_gray = _grayscale(prediction)
    target_gray = _grayscale(target)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=prediction.device, dtype=prediction.dtype),
        torch.arange(width, device=prediction.device, dtype=prediction.dtype),
        indexing="ij",
    )
    pooled = functional.max_pool2d(mask, kernel_size=9, stride=1, padding=4)
    peaks = torch.logical_and(mask >= pooled - 1e-6, mask >= 0.5)
    instance_losses: list[Tensor] = []
    sigma = max(2.0, min(height, width) / 32.0)

    for batch_index in range(prediction.shape[0]):
        candidates = torch.nonzero(peaks[batch_index, 0], as_tuple=False)
        ordered = sorted(
            (
                (-float(mask[batch_index, 0, y, x]), int(y), int(x))
                for y, x in candidates
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        selected: list[tuple[int, int]] = []
        for _negative_value, y, x in ordered:
            if all((x - prior_x) ** 2 + (y - prior_y) ** 2 >= 16 for prior_y, prior_x in selected):
                selected.append((y, x))
            if len(selected) >= 4_096:
                break
        for center_y, center_x in selected:
            region = torch.exp(
                -((xx - center_x).square() + (yy - center_y).square()) / (2.0 * sigma * sigma)
            ).unsqueeze(0)
            predicted_ink = (1.0 - predicted_gray[batch_index]).clamp(0.0, 1.0) * region
            target_ink = (1.0 - target_gray[batch_index]).clamp(0.0, 1.0) * region

            def center(ink: Tensor) -> tuple[Tensor, Tensor]:
                mass = ink.sum().clamp_min(1e-6)
                return (ink * xx).sum() / mass, (ink * yy).sum() / mass

            predicted_x, predicted_y = center(predicted_ink)
            target_x, target_y = center(target_ink)
            instance_losses.append(
                0.5
                * (
                    functional.smooth_l1_loss(predicted_x / width, target_x / width)
                    + functional.smooth_l1_loss(predicted_y / height, target_y / height)
                )
            )
    if not instance_losses:
        return prediction.sum() * 0.0
    return torch.stack(instance_losses).mean()


def ocr_consistency_loss(prediction: Tensor, target: Tensor, ocr_mask: Tensor | None) -> Tensor:
    """Preserve masked text-stroke gradients as a recognizer-independent proxy."""

    mask = _normalized_mask(ocr_mask, prediction, "ocr_mask")
    if mask is None or not bool(torch.any(mask > 0)):
        return prediction.sum() * 0.0
    predicted_x, predicted_y = _spatial_gradients(prediction)
    target_x, target_y = _spatial_gradients(target)
    denominator = mask.sum().clamp_min(1.0)
    horizontal = ((predicted_x - target_x).abs() * mask).sum() / denominator
    vertical = ((predicted_y - target_y).abs() * mask).sum() / denominator
    return 0.5 * (horizontal + vertical)


class GraphSRLoss(nn.Module):
    """Weighted non-adversarial GraphSR objective with named components."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        marker_center_map: Tensor | None = None,
        ocr_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must have the same NCHW shape")
        reconstruction = charbonnier_loss(prediction, target)
        edge = edge_consistency_loss(prediction, target)
        marker_center = marker_center_consistency_loss(prediction, target, marker_center_map)
        ocr_consistency = ocr_consistency_loss(prediction, target, ocr_mask)
        # Adversarial training is intentionally absent. A nonzero configured
        # weight is recorded for experimentation but contributes no hidden loss.
        adversarial = prediction.sum() * 0.0
        total = (
            self.weights.reconstruction * reconstruction
            + self.weights.edge * edge
            + self.weights.marker_center * marker_center
            + self.weights.ocr_consistency * ocr_consistency
            + self.weights.adversarial * adversarial
        )
        return {
            "total": total,
            "reconstruction": reconstruction,
            "edge": edge,
            "marker_center": marker_center,
            "ocr_consistency": ocr_consistency,
            "adversarial": adversarial,
        }


__all__ = [
    "GraphSRLoss",
    "LossWeights",
    "MAX_ADVERSARIAL_WEIGHT",
    "charbonnier_loss",
    "edge_consistency_loss",
    "marker_center_consistency_loss",
    "ocr_consistency_loss",
]
