# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Export-safe P2 candidate model for deterministic hard-negative repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ml.markers.center.candidate_level_v1.model import (
    CandidatePatchNet,
    CandidateTensorContract,
    ConvBlock,
)


@dataclass(frozen=True)
class CandidateP2ModelConfig:
    architecture: str = "candidate-spatial-patch-cnn-v2-export-safe"
    channels: tuple[int, ...] = (16, 24, 32, 48)
    seed: int = 20260822


class CandidatePatchNetP2(nn.Module):
    """P1-compatible tensor contract with an ONNX-exportable spatial pool."""

    contract = CandidateTensorContract()

    def __init__(self, config: CandidateP2ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or CandidateP2ModelConfig()
        generator_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.config.seed)
            c0, c1, c2, c3 = self.config.channels
            self.features = nn.Sequential(
                ConvBlock(3, c0, stride=1),
                ConvBlock(c0, c1, stride=2),
                ConvBlock(c1, c2, stride=2),
                ConvBlock(c2, c3, stride=2),
            )
            # The feature map is exactly 5 x 5 for a 33 x 33 input. A fixed
            # 3 x 3, stride-one average produces the same 3 x 3 head shape
            # without P1's unsupported adaptive 5-to-3 ONNX conversion.
            self.spatial_pool = nn.AvgPool2d(kernel_size=3, stride=1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c3 * 3 * 3, 64),
                nn.SiLU(),
                nn.Dropout(p=0.05),
                nn.Linear(64, 4),
            )
        finally:
            torch.random.set_rng_state(generator_state)

    def forward_raw(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1:] != (3, 33, 33):
            raise ValueError("CandidatePatchNetP2 requires NCHW [N,3,33,33] patches")
        return self.head(self.spatial_pool(self.features(value)))

    def forward(self, value: Tensor) -> Tensor:
        return CandidatePatchNet.activate(self.forward_raw(value))

    def export_contract(self) -> dict[str, object]:
        return {
            "architecture": self.config.architecture,
            "model": asdict(self.config),
            "tensor_contract": asdict(self.contract),
        }


__all__ = ["CandidateP2ModelConfig", "CandidatePatchNetP2"]
