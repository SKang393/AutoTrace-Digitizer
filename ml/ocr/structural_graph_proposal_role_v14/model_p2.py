# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe occupancy repair for structural graph OCR V14 P2."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import StructuralGraphProposalRoleNet
from .protocol import CROP_WIDTH


COLUMN_BIN_COUNT = 18
COLUMN_BIN_BOUNDS = tuple(
    (
        index * CROP_WIDTH // COLUMN_BIN_COUNT,
        ((index + 1) * CROP_WIDTH + COLUMN_BIN_COUNT - 1) // COLUMN_BIN_COUNT,
    )
    for index in range(COLUMN_BIN_COUNT)
)


class StructuralGraphProposalRoleP2Net(StructuralGraphProposalRoleNet):
    """Preserve every P1 parameter while making fixed-width occupancy exportable."""

    @staticmethod
    def _fixed_column_mean(tight: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            tuple(
                tight[:, :, :, start:end].mean(dim=(2, 3), keepdim=True)
                for start, end in COLUMN_BIN_BOUNDS
            ),
            dim=3,
        )

    @staticmethod
    def _fixed_column_peak(tight: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            tuple(
                torch.amax(tight[:, :, :, start:end], dim=(2, 3), keepdim=True)
                for start, end in COLUMN_BIN_BOUNDS
            ),
            dim=3,
        )

    @staticmethod
    def _occupancy(pixels: torch.Tensor) -> torch.Tensor:
        tight = pixels[:, 0:1]
        row_mean = F.adaptive_avg_pool2d(tight, (8, 1)).flatten(1)
        column_mean = StructuralGraphProposalRoleP2Net._fixed_column_mean(tight).flatten(1)
        row_peak = F.adaptive_max_pool2d(tight, (8, 1)).flatten(1)
        column_peak = StructuralGraphProposalRoleP2Net._fixed_column_peak(tight).flatten(1)
        return torch.cat((row_mean, column_mean, row_peak, column_peak), dim=1)


__all__ = ["COLUMN_BIN_BOUNDS", "COLUMN_BIN_COUNT", "StructuralGraphProposalRoleP2Net"]
