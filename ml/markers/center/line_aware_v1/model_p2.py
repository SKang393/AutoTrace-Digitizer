# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe P2 model for the line-aware marker-center defect class."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ml.markers.center.line_aware_v1.model import (
    LineAwarePatchNet,
    LineAwareTensorContract,
)


@dataclass(frozen=True)
class LineAwareP2ModelConfig:
    architecture: str = "line-aware-dual-branch-patch-cnn-v2-export-safe"
    seed: int = 20260901


class LineAwarePatchNetP2(LineAwarePatchNet):
    """P1-compatible network using a fixed, ONNX-exportable spatial pool."""

    contract = LineAwareTensorContract()

    def __init__(self, config: LineAwareP2ModelConfig | None = None) -> None:
        selected = config or LineAwareP2ModelConfig()
        super().__init__()
        self.config = selected
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(selected.seed)
            self.ink_branch = self._branch(1, (16, 24, 32))
            self.mask_branch = self._branch(2, (8, 12, 16))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear((32 + 16) * 3 * 3, 64),
                nn.SiLU(),
                nn.Dropout(0.04),
                nn.Linear(64, 4),
            )
        finally:
            torch.random.set_rng_state(state)

    @staticmethod
    def _branch(input_channels: int, widths: tuple[int, int, int]) -> nn.Sequential:
        a, b, c = widths
        return nn.Sequential(
            nn.Conv2d(input_channels, a, 5, padding=2, bias=False),
            nn.BatchNorm2d(a),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(a, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(b, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            # A 33-pixel patch produces an 8 x 8 feature map. This fixed
            # 4 x 4, stride-two pool preserves P1's 3 x 3 head contract.
            nn.AvgPool2d(kernel_size=4, stride=2),
        )

    def forward_raw(self, value: Tensor) -> Tensor:
        return super().forward_raw(value)

    def export_contract(self) -> dict[str, object]:
        return {
            "architecture": self.config.architecture,
            "model": asdict(self.config),
            "tensor_contract": asdict(self.contract),
        }


__all__ = ["LineAwareP2ModelConfig", "LineAwarePatchNetP2"]
