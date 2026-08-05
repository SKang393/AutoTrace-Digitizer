# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Whole-crop spatial-query semantic-slot recognizer for Candidate 3."""

from __future__ import annotations

import math

import torch
from torch import nn

from .protocol import (
    BLANK_CLASS_INDEX,
    CLASS_COUNT,
    MAX_TOKENS,
    ROLE_COUNT,
    SLOT_TIME_INDICES,
    TIME_STEPS,
)


class SpatialQuerySlotRecognizer(nn.Module):
    """Preserve ordered local evidence and decode it with learned semantic queries."""

    EMBEDDING_DIM = 160
    HEAD_COUNT = 4
    COLUMN_COUNT = 32

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.column_projection = nn.Linear(96 * 4, self.EMBEDDING_DIM)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.COLUMN_COUNT, self.EMBEDDING_DIM)
        )
        self.self_qkv = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM * 3)
        self.self_output = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM)
        self.self_norm = nn.LayerNorm(self.EMBEDDING_DIM)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM * 2),
            nn.GELU(),
            nn.Linear(self.EMBEDDING_DIM * 2, self.EMBEDDING_DIM),
        )
        self.feed_norm = nn.LayerNorm(self.EMBEDDING_DIM)
        self.slot_queries = nn.Parameter(torch.empty(MAX_TOKENS, self.EMBEDDING_DIM))
        self.cross_query = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM)
        self.cross_key = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM)
        self.cross_value = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM)
        self.cross_output = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM)
        self.slot_norm = nn.LayerNorm(self.EMBEDDING_DIM)
        self.slot_classifier = nn.Linear(self.EMBEDDING_DIM, CLASS_COUNT)
        self.role_classifier = nn.Linear(self.EMBEDDING_DIM, ROLE_COUNT)
        self._reset_parameters()

        projector = torch.zeros(TIME_STEPS, MAX_TOKENS, dtype=torch.float32)
        for slot, time_index in enumerate(SLOT_TIME_INDICES):
            projector[time_index, slot] = 1.0
        base_logits = torch.full((TIME_STEPS, CLASS_COUNT), -12.0)
        base_logits[:, BLANK_CLASS_INDEX] = 12.0
        for time_index in SLOT_TIME_INDICES:
            base_logits[time_index] = 0.0
        self.register_buffer("slot_projector", projector, persistent=True)
        self.register_buffer("base_time_logits", base_logits, persistent=True)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, count, _ = values.shape
        head_width = self.EMBEDDING_DIM // self.HEAD_COUNT
        return values.reshape(batch, count, self.HEAD_COUNT, head_width).permute(
            0, 2, 1, 3
        )

    def _self_attention(self, columns: torch.Tensor) -> torch.Tensor:
        batch, count, _ = columns.shape
        head_width = self.EMBEDDING_DIM // self.HEAD_COUNT
        qkv = self.self_qkv(columns).reshape(
            batch, count, 3, self.HEAD_COUNT, head_width
        )
        queries = qkv[:, :, 0].permute(0, 2, 1, 3)
        keys = qkv[:, :, 1].permute(0, 2, 1, 3)
        values = qkv[:, :, 2].permute(0, 2, 1, 3)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(head_width)
        attended = torch.matmul(torch.softmax(scores, dim=-1), values)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch, count, self.EMBEDDING_DIM
        )
        columns = self.self_norm(columns + self.self_output(attended))
        return self.feed_norm(columns + self.feed_forward(columns))

    def semantic_logits(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = inputs - inputs.mean(dim=(2, 3), keepdim=True)
        variance = (centered * centered).mean(dim=(2, 3), keepdim=True)
        standardized = centered / torch.sqrt(variance + 1e-6)
        features = self.encoder(standardized)
        columns = features.permute(0, 3, 1, 2).reshape(
            features.shape[0], self.COLUMN_COUNT, 96 * 4
        )
        columns = self._self_attention(
            self.column_projection(columns) + self.position_embedding
        )

        queries = self.slot_queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        query_heads = self._split_heads(self.cross_query(queries))
        key_heads = self._split_heads(self.cross_key(columns))
        value_heads = self._split_heads(self.cross_value(columns))
        head_width = self.EMBEDDING_DIM // self.HEAD_COUNT
        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) / math.sqrt(
            head_width
        )
        contexts = torch.matmul(torch.softmax(scores, dim=-1), value_heads)
        contexts = contexts.permute(0, 2, 1, 3).reshape(
            features.shape[0], MAX_TOKENS, self.EMBEDDING_DIM
        )
        slot_features = self.slot_norm(queries + self.cross_output(contexts))
        slot_logits = self.slot_classifier(slot_features)
        role_logits = self.role_classifier(columns.mean(dim=1))
        return slot_logits, role_logits

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot_logits, role_logits = self.semantic_logits(inputs)
        time_logits = self.base_time_logits.unsqueeze(0) + torch.einsum(
            "ts,nsc->ntc", self.slot_projector, slot_logits
        )
        return time_logits, role_logits
