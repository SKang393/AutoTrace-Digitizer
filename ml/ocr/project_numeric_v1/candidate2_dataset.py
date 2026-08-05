# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Candidate 2 procedural renderer change and newly sealed split."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from .candidate2_protocol import (
    PROTOCOL_ID,
    SEALED_DEGRADATION_FAMILY,
    SEALED_RENDERER_FAMILY,
    TRAIN_DEGRADATION_FAMILY,
    TRAIN_RENDERER_FAMILY,
)
from .dataset import (
    NumericSample,
    _EXCLUSION_KINDS,
    _label_for,
    _render_negative,
    _render_train,
    _rng_for,
    build_split,
)
from .protocol import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MAX_TOKENS,
    ROLE_NONNUMERIC,
    ROLE_NUMERIC_TEXT,
    SEALED_TEST_NEGATIVE_COUNT,
    SEALED_TEST_POSITIVE_COUNT,
    SEED,
    TRAIN_NEGATIVE_COUNT,
    TRAIN_POSITIVE_COUNT,
)

_SEALED_OUTLINES: dict[str, tuple[str, ...]] = {
    "0": ("0111110", "1100011", "1100111", "1101011", "1110011", "1100011", "1100011", "0111110", "0000000"),
    "1": ("0011000", "0111000", "0011000", "0011000", "0011000", "0011000", "0011000", "1111110", "0000000"),
    "2": ("0111110", "1100011", "0000011", "0000110", "0011000", "0110000", "1100000", "1111111", "0000000"),
    "3": ("1111110", "0000011", "0000011", "0011110", "0000011", "0000011", "0000011", "1111110", "0000000"),
    "4": ("0001100", "0011100", "0111100", "1101100", "1101100", "1111111", "0001100", "0001100", "0000000"),
    "5": ("1111111", "1100000", "1100000", "1111110", "0000011", "0000011", "1100011", "0111110", "0000000"),
    "6": ("0011110", "0110000", "1100000", "1111110", "1100011", "1100011", "1100011", "0111110", "0000000"),
    "7": ("1111111", "0000011", "0000110", "0001100", "0011000", "0110000", "0110000", "0110000", "0000000"),
    "8": ("0111110", "1100011", "1100011", "0111110", "1100011", "1100011", "1100011", "0111110", "0000000"),
    "9": ("0111110", "1100011", "1100011", "0111111", "0000011", "0000011", "0000110", "1111100", "0000000"),
    ".": ("0000000", "0000000", "0000000", "0000000", "0000000", "0000000", "0011000", "0011000", "0000000"),
    "-": ("0000000", "0000000", "0000000", "0000000", "1111111", "0000000", "0000000", "0000000", "0000000"),
    "%": ("1100011", "1100110", "0001100", "0011000", "0110000", "1100000", "1100011", "0000011", "0000000"),
}


def _candidate_rng(split: str, index: int, role_offset: int = 0) -> np.random.Generator:
    material = f"{PROTOCOL_ID}:{SEED}:{split}:{role_offset}:{index}".encode()
    value = int.from_bytes(sha256(material).digest()[:8], "little")
    return np.random.default_rng(value)


def _minimum_filter(image: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return image.copy()
    padded = np.pad(image, radius, mode="edge")
    result = np.full_like(image, 255)
    for y_offset in range(radius * 2 + 1):
        for x_offset in range(radius * 2 + 1):
            result = np.minimum(
                result,
                padded[
                    y_offset : y_offset + image.shape[0],
                    x_offset : x_offset + image.shape[1],
                ],
            )
    return result


def _pixelate(image: np.ndarray, cell: int) -> np.ndarray:
    if cell == 1:
        return image.copy()
    result = image.copy()
    for top in range(0, image.shape[0], cell):
        for left in range(0, image.shape[1], cell):
            block = image[top : top + cell, left : left + cell]
            result[top : top + cell, left : left + cell] = int(np.mean(block))
    return result


def _shift(image: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    result = np.full_like(image, 255)
    source_left = max(0, -x_offset)
    source_right = min(INPUT_WIDTH, INPUT_WIDTH - x_offset)
    source_top = max(0, -y_offset)
    source_bottom = min(INPUT_HEIGHT, INPUT_HEIGHT - y_offset)
    target_left = source_left + x_offset
    target_right = source_right + x_offset
    target_top = source_top + y_offset
    target_bottom = source_bottom + y_offset
    result[target_top:target_bottom, target_left:target_right] = image[
        source_top:source_bottom, source_left:source_right
    ]
    return result


def _render_candidate2_train(text: str, case: str, rng: np.random.Generator) -> np.ndarray:
    base = _render_train(text, case, rng)
    image = _pixelate(base, int(rng.choice(np.asarray((1, 1, 2, 3)))))
    image = _minimum_filter(image, int(rng.choice(np.asarray((0, 1, 1, 2)))))
    image = _shift(image, int(rng.integers(-4, 5)), int(rng.integers(-2, 3)))
    threshold = int(rng.integers(145, 226))
    ink = int(rng.integers(8, 76))
    background = int(rng.integers(239, 256))
    image = np.where(image < threshold, ink, background).astype(np.uint8)
    if int(rng.integers(0, 3)) == 0:
        row = int(rng.integers(2, INPUT_HEIGHT - 2))
        image[row, :] = np.maximum(image[row, :], int(rng.integers(175, 236)))
    dropout_count = int(rng.integers(0, 7))
    ink_pixels = np.argwhere(image < 128)
    if len(ink_pixels):
        for _ in range(dropout_count):
            y, x = ink_pixels[int(rng.integers(0, len(ink_pixels)))]
            radius = int(rng.integers(0, 2))
            image[
                max(0, y - radius) : min(INPUT_HEIGHT, y + radius + 1),
                max(0, x - radius) : min(INPUT_WIDTH, x + radius + 1),
            ] = background
    return image


def _render_candidate2_sealed(text: str, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full(
        (INPUT_HEIGHT, INPUT_WIDTH), int(rng.integers(243, 256)), dtype=np.uint8
    )
    scale = 2 if len(text) >= 5 else int(rng.choice(np.asarray((2, 2, 3))))
    glyph_width = 7 * scale
    spacing = max(1, scale - 1)
    total = len(text) * glyph_width + max(0, len(text) - 1) * spacing
    origin_x = max(1, (INPUT_WIDTH - total) // 2 + int(rng.integers(-2, 3)))
    origin_y = max(1, (INPUT_HEIGHT - 9 * scale) // 2 + int(rng.integers(-1, 2)))
    ink = int(rng.integers(18, 88))
    for glyph_index, character in enumerate(text):
        left = origin_x + glyph_index * (glyph_width + spacing)
        for row_index, row in enumerate(_SEALED_OUTLINES[character]):
            for column_index, bit in enumerate(row):
                if bit == "1":
                    top = origin_y + row_index * scale
                    x = left + column_index * scale
                    canvas[top : top + scale, x : x + scale] = ink
    if int(rng.integers(0, 4)) == 0:
        column = int(rng.integers(3, INPUT_WIDTH - 3))
        canvas[:, column] = np.maximum(canvas[:, column], int(rng.integers(185, 235)))
    return canvas


def _render_candidate2_negative(rng: np.random.Generator, kind: str) -> np.ndarray:
    source = _render_negative("validation", kind, rng)
    source = _shift(source, int(rng.integers(-5, 6)), int(rng.integers(-2, 3)))
    return _minimum_filter(source, int(rng.integers(0, 2)))


def build_candidate2_split(
    split: str,
    positive_count: int | None = None,
    negative_count: int | None = None,
) -> tuple[NumericSample, ...]:
    if split == "validation":
        return build_split(
            "validation", positive_count=positive_count, negative_count=negative_count
        )
    if split not in {"train", "sealed_test"}:
        raise ValueError(f"Unsupported Candidate 2 split: {split}")
    default_positive = (
        TRAIN_POSITIVE_COUNT if split == "train" else SEALED_TEST_POSITIVE_COUNT
    )
    default_negative = (
        TRAIN_NEGATIVE_COUNT if split == "train" else SEALED_TEST_NEGATIVE_COUNT
    )
    positive_count = default_positive if positive_count is None else positive_count
    negative_count = default_negative if negative_count is None else negative_count
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Every Candidate 2 split requires positive and negative samples.")
    renderer_family = (
        TRAIN_RENDERER_FAMILY if split == "train" else SEALED_RENDERER_FAMILY
    )
    degradation_family = (
        TRAIN_DEGRADATION_FAMILY if split == "train" else SEALED_DEGRADATION_FAMILY
    )
    samples: list[NumericSample] = []
    for index in range(positive_count):
        label_rng = _rng_for("train", index) if split == "train" else _candidate_rng(split, index)
        target, case = _label_for(index, label_rng)
        if len(target) > MAX_TOKENS:
            raise ValueError(f"Generated target exceeds frozen token count: {target}")
        render_rng = _candidate_rng(split, index)
        raster = (
            _render_candidate2_train(target, case, render_rng)
            if split == "train"
            else _render_candidate2_sealed(target, render_rng)
        )
        samples.append(
            NumericSample(
                sample_id=f"project-numeric-v1-candidate2-{split}-positive-{index:05d}",
                split=split,  # type: ignore[arg-type]
                target_text=target,
                case=case,
                role=ROLE_NUMERIC_TEXT,
                exclusion_kind=None,
                renderer_family=renderer_family,
                degradation_family=degradation_family,
                raster=raster,
            )
        )
    for index in range(negative_count):
        rng = _candidate_rng(split, index, role_offset=1)
        kind = _EXCLUSION_KINDS[index % len(_EXCLUSION_KINDS)]
        raster = (
            _render_negative("train", kind, rng)
            if split == "train"
            else _render_candidate2_negative(rng, kind)
        )
        samples.append(
            NumericSample(
                sample_id=f"project-numeric-v1-candidate2-{split}-negative-{index:05d}",
                split=split,  # type: ignore[arg-type]
                target_text="",
                case="marker_exclusion",
                role=ROLE_NONNUMERIC,
                exclusion_kind=kind,
                renderer_family=renderer_family,
                degradation_family=degradation_family,
                raster=raster,
            )
        )
    return tuple(samples)
