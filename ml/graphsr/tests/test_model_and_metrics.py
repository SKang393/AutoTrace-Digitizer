# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from ml.graphsr.losses import (
    GraphSRLoss,
    LossWeights,
    MAX_ADVERSARIAL_WEIGHT,
    marker_center_consistency_loss,
)
from ml.graphsr.metrics import METRIC_NAMES, evaluate_quality_metrics, marker_center_metrics
from ml.graphsr.model import GraphSRConfig, GraphSRx2


def _annotations(centers: tuple[tuple[float, float], ...]) -> dict[str, object]:
    markers = tuple({"center": center, "radius": 4.0} for center in centers)
    return {
        "ocr_regions": ((16.0, 84.0, 18.0, 8.0),),
        "marker_centers": markers,
        "axis_lines": ((12.0, 8.0, 12.0, 82.0), (12.0, 82.0, 88.0, 82.0)),
        "open_markers": markers,
    }


def test_compact_model_has_exact_bounded_x2_contract() -> None:
    model = GraphSRx2(GraphSRConfig(channels=8, blocks=1, seed=7)).eval()
    input_tensor = torch.linspace(0.0, 1.0, 3 * 18 * 22).reshape(1, 3, 18, 22)

    with torch.inference_mode():
        output = model(input_tensor)
        interpolation_anchor = functional.interpolate(
            input_tensor,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

    assert output.shape == (1, 3, 36, 44)
    assert torch.isfinite(output).all()
    assert torch.all((0.0 <= output) & (output <= 1.0))
    assert torch.equal(output, interpolation_anchor)
    assert model.contract.scale == 2
    assert model.contract.coordinate_space == "enhanced_pixels"
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_loss_exposes_finite_scientific_components_and_limits_adversarial_use() -> None:
    target = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    target[:, :, 8:25, 16] = 0.0
    prediction = target.clone()
    prediction[:, :, 8:25, 16] = 1.0
    prediction[:, :, 13:18, 19:24] = 0.0
    marker_map = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    marker_map[:, :, 10:24, 10:26] = 1.0
    ocr_mask = torch.zeros_like(marker_map)
    ocr_mask[:, :, 7:26, 14:19] = 1.0

    losses = GraphSRLoss()(prediction, target, marker_map, ocr_mask)

    assert set(losses) == {
        "total",
        "reconstruction",
        "edge",
        "marker_center",
        "ocr_consistency",
        "adversarial",
    }
    assert all(torch.isfinite(value) and float(value) >= 0.0 for value in losses.values())
    assert float(losses["edge"]) > 0.0
    assert float(losses["marker_center"]) > 0.0
    assert float(losses["ocr_consistency"]) > 0.0
    assert float(losses["adversarial"]) == 0.0
    assert LossWeights().adversarial == 0.0
    with pytest.raises(ValueError, match="adversarial"):
        LossWeights(adversarial=MAX_ADVERSARIAL_WEIGHT + 0.001)


def test_structure_metrics_detect_open_marker_closure_and_thin_line_loss(
    chart_fixture: tuple[np.ndarray, tuple[tuple[float, float], ...]],
) -> None:
    reference, centers = chart_fixture
    annotations = _annotations(centers)
    faithful = evaluate_quality_metrics(reference, reference.copy(), annotations)

    damaged = reference.copy()
    damaged[8:83, 12, :] = 255
    damaged[82, 12:89, :] = 255
    for x, y in centers:
        damaged[int(y) - 2 : int(y) + 3, int(x) - 2 : int(x) + 3, :] = 0
    degraded = evaluate_quality_metrics(reference, damaged, annotations)

    assert set(faithful.to_dict()) == set(METRIC_NAMES[:-2])
    assert faithful.marker_center_f1 is None
    assert faithful.numeric_ocr_exact_match is None
    assert faithful.numeric_ocr_proxy_exact_match == 1.0
    assert faithful.shape_fill_classification_f1 is None
    assert faithful.marker_center_mean_error_pixels is not None
    assert faithful.marker_center_mean_error_pixels <= 1.0
    assert faithful.axis_thin_line_recall == 1.0
    assert faithful.axis_line_localization_error_pixels == 0.0
    assert faithful.open_marker_preservation_rate == 1.0
    assert faithful.hallucinated_structure_rate == 0.0
    assert degraded.axis_thin_line_recall == 0.0
    assert degraded.open_marker_preservation_rate == 0.0
    assert degraded.hallucinated_structure_rate > 0.0


def test_marker_f1_requires_predictions_and_penalizes_false_positives(
    chart_fixture: tuple[np.ndarray, tuple[tuple[float, float], ...]],
) -> None:
    image, centers = chart_fixture
    annotations = tuple({"center": center, "radius": 4.0} for center in centers)
    exact_f1, _ = marker_center_metrics(
        image,
        annotations,
        predicted_markers=annotations,
    )
    with_false_positive, _ = marker_center_metrics(
        image,
        annotations,
        predicted_markers=annotations + ({"center": (92.0, 4.0), "radius": 4.0},),
    )
    unavailable, _ = marker_center_metrics(image, annotations)

    assert exact_f1 == 1.0
    assert with_false_positive is not None and with_false_positive < 1.0
    assert unavailable is None


def test_marker_f1_uses_maximum_cardinality_tolerance_matching() -> None:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    expected = (
        {"center": (5.0, 8.0), "radius": 2.0},
        {"center": (7.0, 8.0), "radius": 2.0},
    )
    predicted = (
        {"center": (6.0, 8.0), "radius": 2.0},
        {"center": (4.0, 8.0), "radius": 2.0},
    )

    f1, _ = marker_center_metrics(
        image,
        expected,
        predicted_markers=predicted,
        tolerance_pixels=1.1,
    )

    assert f1 == 1.0


def test_marker_center_loss_cannot_cancel_opposing_marker_movements() -> None:
    target = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    prediction = target.clone()
    for x in (10, 22):
        target[:, :, 15:18, x - 1 : x + 2] = 0.0
    prediction[:, :, 15:18, 7:10] = 0.0
    prediction[:, :, 15:18, 23:26] = 0.0
    marker_map = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    for x in (10, 22):
        marker_map[0, 0] = torch.maximum(
            marker_map[0, 0],
            torch.exp(-((xx - x).float().square() + (yy - 16).float().square()) / 8.0),
        )

    loss = marker_center_consistency_loss(prediction, target, marker_map)

    assert torch.isfinite(loss)
    assert float(loss) > 0.0
