# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic binary projection and morphology features for OCR V27."""

from __future__ import annotations

import numpy as np

from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    STRUCTURE_FEATURE_COUNT,
)


INK_THRESHOLD = 0.35


def _span_fraction(active: np.ndarray) -> np.ndarray:
    indices = np.arange(active.shape[1], dtype=np.int64)[None, :]
    first = np.where(active, indices, active.shape[1]).min(axis=1)
    last = np.where(active, indices, -1).max(axis=1)
    span = np.where(last >= first, last - first + 1, 0)
    return span.astype(np.float64) / float(active.shape[1])


def _channel_features(values: np.ndarray) -> np.ndarray:
    binary = values >= INK_THRESHOLD
    rows = binary.mean(axis=2)
    columns = binary.mean(axis=1)
    active_rows = rows > 0.0
    active_columns = columns > 0.0
    row_transitions = np.not_equal(binary[:, 1:, :], binary[:, :-1, :]).mean(
        axis=(1, 2)
    )
    column_transitions = np.not_equal(
        binary[:, :, 1:], binary[:, :, :-1]
    ).mean(axis=(1, 2))
    edge = np.concatenate(
        (binary[:, 0, :], binary[:, -1, :], binary[:, :, 0], binary[:, :, -1]),
        axis=1,
    ).mean(axis=1)
    center = binary[:, 8:24, 32:96].mean(axis=(1, 2))
    corners = np.concatenate(
        (
            binary[:, :8, :16].reshape(len(binary), -1),
            binary[:, :8, -16:].reshape(len(binary), -1),
            binary[:, -8:, :16].reshape(len(binary), -1),
            binary[:, -8:, -16:].reshape(len(binary), -1),
        ),
        axis=1,
    ).mean(axis=1)
    return np.stack(
        (
            binary.mean(axis=(1, 2)),
            active_rows.mean(axis=1),
            active_columns.mean(axis=1),
            rows.max(axis=1),
            columns.max(axis=1),
            _span_fraction(active_columns),
            _span_fraction(active_rows),
            row_transitions,
            column_transitions,
            edge,
            center,
            corners,
        ),
        axis=1,
    )


def structure_features(crops: np.ndarray) -> np.ndarray:
    """Return 24 fixed float64 morphology values for each proposal crop."""
    values = np.asarray(crops)
    if values.ndim != 4 or values.shape[1:] != (
        CROP_CHANNELS,
        CROP_HEIGHT,
        CROP_WIDTH,
    ):
        raise ValueError("Expected proposal crops [N,2,32,128]")
    if not np.isfinite(values).all():
        raise ValueError("Proposal crops contain nonfinite values")
    result = np.concatenate(
        tuple(_channel_features(values[:, channel]) for channel in range(CROP_CHANNELS)),
        axis=1,
    ).astype(np.float64)
    if result.shape != (len(values), STRUCTURE_FEATURE_COUNT):
        raise RuntimeError("OCR V27 structure feature width changed")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise RuntimeError("OCR V27 structure features left normalized range")
    return np.ascontiguousarray(result)


__all__ = ["INK_THRESHOLD", "structure_features"]
