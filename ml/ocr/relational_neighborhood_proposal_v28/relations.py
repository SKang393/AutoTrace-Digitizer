# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Truth-independent ordered proposal-pair geometry for OCR V28."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from ml.ocr.margin_calibrator_v20.pipeline import ProposalRecord

from .dataset import SceneSample, proposals
from .protocol import RELATION_FEATURE_COUNT


def _overlap(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _axis_gap(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0.0


def proposal_relation_features(
    scene: SceneSample,
    records: tuple[ProposalRecord, ...],
) -> np.ndarray:
    """Encode one complete scene proposal set without consulting truth labels."""
    if not records:
        raise RuntimeError("OCR V28 relation stream requires at least one proposal")
    if len({record.scene_index for record in records}) != 1:
        raise RuntimeError("OCR V28 relation stream cannot mix scenes")
    candidates = proposals(scene.raster)
    boxes = []
    for record in records:
        if record.candidate_index < 0 or record.candidate_index >= len(candidates):
            raise RuntimeError("OCR V28 relation record candidate index is invalid")
        boxes.append(candidates[record.candidate_index].box)
    count = len(boxes)
    height, width = scene.raster.shape
    plot_width = max(1.0, float(scene.plot.width))
    plot_height = max(1.0, float(scene.plot.height))
    result = np.zeros((count, count, RELATION_FEATURE_COUNT), dtype=np.float32)
    for first_index, first in enumerate(boxes):
        first_width = max(1.0, float(first.width))
        first_height = max(1.0, float(first.height))
        first_center_x = (float(first.left) + float(first.right)) / 2.0
        first_center_y = (float(first.top) + float(first.bottom)) / 2.0
        for second_index, second in enumerate(boxes):
            second_width = max(1.0, float(second.width))
            second_height = max(1.0, float(second.height))
            second_center_x = (float(second.left) + float(second.right)) / 2.0
            second_center_y = (float(second.top) + float(second.bottom)) / 2.0
            delta_x = (second_center_x - first_center_x) / max(1.0, float(width))
            delta_y = (second_center_y - first_center_y) / max(1.0, float(height))
            overlap_x = _overlap(first.left, first.right, second.left, second.right)
            overlap_y = _overlap(first.top, first.bottom, second.top, second.bottom)
            overlap_area = overlap_x * overlap_y
            union = first_width * first_height + second_width * second_height - overlap_area
            x_fraction = overlap_x / min(first_width, second_width)
            y_fraction = overlap_y / min(first_height, second_height)
            first_inside = (
                first_center_x >= scene.plot.left
                and first_center_x <= scene.plot.right
                and first_center_y >= scene.plot.top
                and first_center_y <= scene.plot.bottom
            )
            second_inside = (
                second_center_x >= scene.plot.left
                and second_center_x <= scene.plot.right
                and second_center_y >= scene.plot.top
                and second_center_y <= scene.plot.bottom
            )
            result[first_index, second_index] = np.asarray((
                np.clip(delta_x, -1.0, 1.0),
                np.clip(delta_y, -1.0, 1.0),
                min(1.0, abs(delta_x)),
                min(1.0, abs(delta_y)),
                np.clip(np.log(second_width / first_width) / 4.0, -1.0, 1.0),
                np.clip(np.log(second_height / first_height) / 4.0, -1.0, 1.0),
                np.clip(x_fraction, 0.0, 1.0),
                np.clip(y_fraction, 0.0, 1.0),
                np.clip(overlap_area / max(1.0, union), 0.0, 1.0),
                min(1.0, _axis_gap(first.left, first.right, second.left, second.right) / width),
                min(1.0, _axis_gap(first.top, first.bottom, second.top, second.bottom) / height),
                float(y_fraction >= 0.5),
                float(x_fraction >= 0.5),
                np.clip((second_center_x - scene.plot.left) / plot_width, -1.0, 1.0),
                np.clip((second_center_y - scene.plot.top) / plot_height, -1.0, 1.0),
                float(first_center_x < scene.plot.left and second_center_x < scene.plot.left),
                float(first_center_x > scene.plot.right and second_center_x > scene.plot.right),
                float(first_inside and second_inside),
                float(first_index == second_index),
            ), dtype=np.float32)
    if result.shape != (count, count, RELATION_FEATURE_COUNT):
        raise RuntimeError("OCR V28 relation tensor shape changed")
    if not np.isfinite(result).all() or float(result.min()) < -1.0 or float(result.max()) > 1.0:
        raise RuntimeError("OCR V28 relation tensor is invalid")
    return result


def scene_relation_stream(
    scenes: tuple[SceneSample, ...],
    records: tuple[ProposalRecord, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[slice, ...], str]:
    """Return scene-local relation tensors, stable record slices, and a stream hash."""
    tensors: list[np.ndarray] = []
    slices: list[slice] = []
    digest = sha256()
    start = 0
    for scene_index, scene in enumerate(scenes):
        stop = start
        while stop < len(records) and records[stop].scene_index == scene_index:
            stop += 1
        if stop == start:
            raise RuntimeError(f"OCR V28 scene {scene_index} has no selected proposals")
        if any(record.scene_index < scene_index for record in records[stop:]):
            raise RuntimeError("OCR V28 proposal records are not scene ordered")
        scene_records = records[start:stop]
        tensor = proposal_relation_features(scene, scene_records)
        digest.update(np.ascontiguousarray(tensor).tobytes(order="C"))
        tensors.append(tensor)
        slices.append(slice(start, stop))
        start = stop
    if start != len(records):
        raise RuntimeError("OCR V28 proposal records reference an unknown scene")
    return tuple(tensors), tuple(slices), digest.hexdigest()


__all__ = ["proposal_relation_features", "scene_relation_stream"]

