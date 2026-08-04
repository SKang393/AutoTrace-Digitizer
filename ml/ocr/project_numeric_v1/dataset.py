# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Independent procedural renderer families for project numeric OCR V1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Literal

import numpy as np
import torch

from .protocol import (
    ALPHABET,
    BLANK_CLASS_INDEX,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MAX_TOKENS,
    ROLE_NONNUMERIC,
    ROLE_NUMERIC_TEXT,
    SEED,
    SPLITS,
)

Split = Literal["train", "validation", "sealed_test"]


@dataclass(frozen=True)
class NumericSample:
    sample_id: str
    split: Split
    target_text: str
    case: str
    role: int
    exclusion_kind: str | None
    renderer_family: str
    degradation_family: str
    raster: np.ndarray


_TRAIN_POLYLINES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "0": (((1, 0), (5, 0), (6, 2), (6, 8), (5, 10), (1, 10), (0, 8), (0, 2), (1, 0)),),
    "1": (((1, 2), (3, 0), (3, 10)), ((1, 10), (5, 10))),
    "2": (((0, 2), (1, 0), (5, 0), (6, 2), (0, 10), (6, 10)),),
    "3": (((0, 0), (5, 0), (6, 2), (4, 5), (6, 8), (5, 10), (0, 10)),),
    "4": (((5, 10), (5, 0)), ((5, 6), (0, 6), (4, 0))),
    "5": (((6, 0), (1, 0), (0, 5), (5, 5), (6, 7), (5, 10), (0, 10)),),
    "6": (((6, 1), (5, 0), (1, 0), (0, 3), (0, 8), (1, 10), (5, 10), (6, 8), (6, 6), (5, 5), (0, 5)),),
    "7": (((0, 0), (6, 0), (2, 10)),),
    "8": (((1, 5), (0, 3), (1, 0), (5, 0), (6, 3), (5, 5), (1, 5), (0, 7), (1, 10), (5, 10), (6, 7), (5, 5)),),
    "9": (((6, 5), (1, 5), (0, 3), (1, 0), (5, 0), (6, 2), (6, 8), (5, 10), (1, 10)),),
    ".": (((3, 9), (3, 10)),),
    "-": (((0, 5), (6, 5)),),
    "%": (((0, 10), (6, 0)), ((1, 1), (2, 1), (2, 3), (1, 3), (1, 1)), ((4, 7), (5, 7), (5, 9), (4, 9), (4, 7))),
}

_VALIDATION_BITMAPS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "%": ("11001", "11010", "00100", "00100", "01000", "10110", "00110"),
}

_SEALED_SEGMENTS: dict[str, frozenset[str]] = {
    "0": frozenset("abcedf"),
    "1": frozenset("bc"),
    "2": frozenset("abdeg"),
    "3": frozenset("abcdg"),
    "4": frozenset("bcfg"),
    "5": frozenset("acdfg"),
    "6": frozenset("acdefg"),
    "7": frozenset("abc"),
    "8": frozenset("abcdefg"),
    "9": frozenset("abcdfg"),
}

_EXCLUSION_KINDS = (
    "filled_circle",
    "open_circle",
    "axis_or_tick",
    "divider",
    "bracket",
    "arrow",
    "legend_box",
    "line_intersection",
    "filled_square",
)


def _rng_for(split: Split, index: int, role_offset: int = 0) -> np.random.Generator:
    registration = next(item for item in SPLITS if item.split == split)
    material = f"{SEED}:{registration.seed_offset}:{split}:{role_offset}:{index}".encode()
    value = int.from_bytes(sha256(material).digest()[:8], "little")
    return np.random.default_rng(value)


def _label_for(index: int, rng: np.random.Generator) -> tuple[str, str]:
    case = index % 8
    if case == 0:
        return str(int(rng.integers(0, 10_001))), "integer"
    if case == 1:
        return f"{int(rng.integers(0, 1001))}.{int(rng.integers(0, 10))}", "decimal"
    if case == 2:
        return f"-{int(rng.integers(0, 1001))}", "negative"
    if case == 3:
        return f"{int(rng.integers(0, 101))}%", "percentage"
    if case == 4:
        return f"-{int(rng.integers(0, 101))}.{int(rng.integers(0, 10))}", "negative_decimal"
    if case == 5:
        return f"{int(rng.integers(0, 101))}.{int(rng.integers(0, 10))}%", "decimal_percentage"
    if case == 6:
        return str(rng.choice(np.asarray(("0", "10", "20", "50", "100", "0.0")))), "o_zero_ambiguity"
    return str(rng.choice(np.asarray(("1", "10", "11", "21", "101", "1.1")))), "l_one_ambiguity"


def _new_canvas(rng: np.random.Generator, low: int = 246, high: int = 256) -> np.ndarray:
    return np.full((INPUT_HEIGHT, INPUT_WIDTH), int(rng.integers(low, high)), dtype=np.uint8)


def _paint_square(canvas: np.ndarray, x: int, y: int, radius: int, value: int) -> None:
    left = max(0, x - radius)
    right = min(INPUT_WIDTH, x + radius + 1)
    top = max(0, y - radius)
    bottom = min(INPUT_HEIGHT, y + radius + 1)
    canvas[top:bottom, left:right] = np.minimum(canvas[top:bottom, left:right], value)


def _draw_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    value: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        fraction = step / steps
        x = int(round(x0 + (x1 - x0) * fraction))
        y = int(round(y0 + (y1 - y0) * fraction))
        _paint_square(canvas, x, y, max(0, width - 1), value)


def _render_train(text: str, case: str, rng: np.random.Generator) -> np.ndarray:
    canvas = _new_canvas(rng)
    scale = 2 if len(text) >= 5 else int(rng.choice(np.asarray((2, 2, 3))))
    glyph_width = 7 * scale
    spacing = int(rng.integers(1, 4))
    total = len(text) * glyph_width + max(0, len(text) - 1) * spacing
    origin_x = max(1, (INPUT_WIDTH - total) // 2 + int(rng.integers(-2, 3)))
    origin_y = max(1, (INPUT_HEIGHT - 11 * scale) // 2 + int(rng.integers(-1, 2)))
    ink = int(rng.integers(4, 55))
    width = int(rng.choice(np.asarray((1, 1, 2))))
    for glyph_index, character in enumerate(text):
        glyph_x = origin_x + glyph_index * (glyph_width + spacing)
        for polyline in _TRAIN_POLYLINES[character]:
            points = [
                (glyph_x + point_x * scale, origin_y + point_y * scale)
                for point_x, point_y in polyline
            ]
            for start, end in zip(points, points[1:]):
                _draw_line(canvas, start, end, width, ink)
        if character == "0" and case == "o_zero_ambiguity":
            _draw_line(
                canvas,
                (glyph_x + scale, origin_y + 9 * scale),
                (glyph_x + 5 * scale, origin_y + scale),
                1,
                ink,
            )
        if character == "1" and case == "l_one_ambiguity":
            _draw_line(
                canvas,
                (glyph_x + scale, origin_y + 10 * scale),
                (glyph_x + 6 * scale, origin_y + 10 * scale),
                width,
                ink,
            )
    speckles = int(rng.integers(0, 10))
    for _ in range(speckles):
        canvas[int(rng.integers(0, INPUT_HEIGHT)), int(rng.integers(0, INPUT_WIDTH))] = int(
            rng.integers(120, 235)
        )
    return canvas


def _render_validation(text: str, case: str, rng: np.random.Generator) -> np.ndarray:
    canvas = _new_canvas(rng, 242, 253)
    scale = int(rng.choice(np.asarray((2, 3))))
    glyph_width = 5 * scale
    spacing = scale
    total = len(text) * glyph_width + max(0, len(text) - 1) * spacing
    origin_x = max(1, (INPUT_WIDTH - total) // 2 + int(rng.integers(-1, 2)))
    origin_y = max(1, (INPUT_HEIGHT - 7 * scale) // 2)
    ink = int(rng.integers(20, 80))
    for glyph_index, character in enumerate(text):
        glyph_x = origin_x + glyph_index * (glyph_width + spacing)
        for row_index, row in enumerate(_VALIDATION_BITMAPS[character]):
            for column_index, bit in enumerate(row):
                if bit == "1":
                    top = origin_y + row_index * scale
                    left = glyph_x + column_index * scale
                    canvas[top : top + scale, left : left + scale] = ink
        if character == "0" and case == "o_zero_ambiguity":
            _draw_line(canvas, (glyph_x, origin_y + 6 * scale), (glyph_x + 4 * scale, origin_y), 1, ink)
        if character == "1" and case == "l_one_ambiguity":
            canvas[origin_y + 6 * scale - 1 : origin_y + 7 * scale, glyph_x : glyph_x + 5 * scale] = ink
    if int(rng.integers(0, 3)) == 0:
        row = int(rng.integers(3, INPUT_HEIGHT - 3))
        canvas[row, :] = np.minimum(canvas[row, :], int(rng.integers(190, 235)))
    return canvas


def _sealed_segment_lines(origin_x: int, origin_y: int) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    return {
        "a": ((origin_x + 2, origin_y), (origin_x + 8, origin_y)),
        "b": ((origin_x + 9, origin_y + 1), (origin_x + 9, origin_y + 8)),
        "c": ((origin_x + 9, origin_y + 10), (origin_x + 9, origin_y + 17)),
        "d": ((origin_x + 2, origin_y + 18), (origin_x + 8, origin_y + 18)),
        "e": ((origin_x + 1, origin_y + 10), (origin_x + 1, origin_y + 17)),
        "f": ((origin_x + 1, origin_y + 1), (origin_x + 1, origin_y + 8)),
        "g": ((origin_x + 2, origin_y + 9), (origin_x + 8, origin_y + 9)),
    }


def _render_sealed(text: str, case: str, rng: np.random.Generator) -> np.ndarray:
    canvas = _new_canvas(rng, 248, 256)
    cell_width = 12
    total = len(text) * cell_width
    origin_x = max(1, (INPUT_WIDTH - total) // 2 + int(rng.integers(-1, 2)))
    origin_y = 6 + int(rng.integers(-1, 2))
    ink = int(rng.integers(0, 45))
    width = int(rng.choice(np.asarray((1, 2))))
    for glyph_index, character in enumerate(text):
        glyph_x = origin_x + glyph_index * cell_width
        if character in _SEALED_SEGMENTS:
            lines = _sealed_segment_lines(glyph_x, origin_y)
            for segment in _SEALED_SEGMENTS[character]:
                _draw_line(canvas, *lines[segment], width, ink)
            if character == "0" and case == "o_zero_ambiguity":
                _draw_line(canvas, (glyph_x + 2, origin_y + 16), (glyph_x + 8, origin_y + 2), 1, ink)
            if character == "1" and case == "l_one_ambiguity":
                _draw_line(canvas, (glyph_x + 2, origin_y + 18), (glyph_x + 10, origin_y + 18), width, ink)
        elif character == "-":
            _draw_line(canvas, (glyph_x + 1, origin_y + 9), (glyph_x + 9, origin_y + 9), width, ink)
        elif character == ".":
            _paint_square(canvas, glyph_x + 5, origin_y + 17, width, ink)
        elif character == "%":
            _draw_line(canvas, (glyph_x + 1, origin_y + 17), (glyph_x + 9, origin_y + 1), width, ink)
            _paint_square(canvas, glyph_x + 2, origin_y + 3, width, ink)
            _paint_square(canvas, glyph_x + 8, origin_y + 15, width, ink)
    if int(rng.integers(0, 4)) == 0:
        column = int(rng.integers(2, INPUT_WIDTH - 2))
        canvas[:, column] = np.maximum(canvas[:, column], 235)
    return canvas


def _render_negative(split: Split, kind: str, rng: np.random.Generator) -> np.ndarray:
    canvas = _new_canvas(rng)
    ink = int(rng.integers(0, 65))
    width = 1 if split == "validation" else 2
    center_x = INPUT_WIDTH // 2 + int(rng.integers(-8, 9))
    center_y = INPUT_HEIGHT // 2 + int(rng.integers(-3, 4))
    if kind in {"filled_circle", "open_circle"}:
        radius = int(rng.integers(3, 7))
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                distance = (x - center_x) ** 2 + (y - center_y) ** 2
                if kind == "filled_circle" and distance <= radius * radius:
                    _paint_square(canvas, x, y, 0, ink)
                if kind == "open_circle" and abs(distance - radius * radius) <= radius:
                    _paint_square(canvas, x, y, 0, ink)
    elif kind in {"axis_or_tick", "divider"}:
        if kind == "axis_or_tick":
            _draw_line(canvas, (18, center_y), (110, center_y), width, ink)
            _draw_line(canvas, (center_x, center_y - 5), (center_x, center_y + 5), width, ink)
        else:
            _draw_line(canvas, (center_x, 2), (center_x, 29), width, ink)
    elif kind == "bracket":
        _draw_line(canvas, (center_x - 12, 7), (center_x - 12, 25), width, ink)
        _draw_line(canvas, (center_x - 12, 7), (center_x + 12, 7), width, ink)
        _draw_line(canvas, (center_x - 12, 25), (center_x + 12, 25), width, ink)
    elif kind == "arrow":
        _draw_line(canvas, (center_x - 20, center_y), (center_x + 18, center_y), width, ink)
        _draw_line(canvas, (center_x + 18, center_y), (center_x + 10, center_y - 6), width, ink)
        _draw_line(canvas, (center_x + 18, center_y), (center_x + 10, center_y + 6), width, ink)
    elif kind == "legend_box":
        for start, end in (
            ((center_x - 22, 6), (center_x + 22, 6)),
            ((center_x + 22, 6), (center_x + 22, 26)),
            ((center_x + 22, 26), (center_x - 22, 26)),
            ((center_x - 22, 26), (center_x - 22, 6)),
        ):
            _draw_line(canvas, start, end, width, ink)
    elif kind == "line_intersection":
        _draw_line(canvas, (center_x - 25, center_y - 9), (center_x + 25, center_y + 9), width, ink)
        _draw_line(canvas, (center_x - 25, center_y + 9), (center_x + 25, center_y - 9), width, ink)
    else:
        canvas[center_y - 5 : center_y + 6, center_x - 5 : center_x + 6] = ink
    return canvas


def build_split(
    split: Split,
    positive_count: int | None = None,
    negative_count: int | None = None,
) -> tuple[NumericSample, ...]:
    registration = next(item for item in SPLITS if item.split == split)
    positive_count = registration.positive_count if positive_count is None else positive_count
    negative_count = registration.negative_count if negative_count is None else negative_count
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Every procedural split requires positive and negative samples.")
    positives = []
    for index in range(positive_count):
        rng = _rng_for(split, index)
        target, case = _label_for(index, rng)
        if len(target) > MAX_TOKENS:
            raise ValueError(f"Generated target exceeds frozen token count: {target}")
        renderer = {
            "train": _render_train,
            "validation": _render_validation,
            "sealed_test": _render_sealed,
        }[split]
        positives.append(
            NumericSample(
                sample_id=f"project-numeric-v1-{split}-positive-{index:05d}",
                split=split,
                target_text=target,
                case=case,
                role=ROLE_NUMERIC_TEXT,
                exclusion_kind=None,
                renderer_family=registration.renderer_family,
                degradation_family=registration.degradation_family,
                raster=renderer(target, case, rng),
            )
        )
    negatives = []
    for index in range(negative_count):
        rng = _rng_for(split, index, role_offset=1)
        kind = _EXCLUSION_KINDS[index % len(_EXCLUSION_KINDS)]
        negatives.append(
            NumericSample(
                sample_id=f"project-numeric-v1-{split}-negative-{index:05d}",
                split=split,
                target_text="",
                case="marker_exclusion",
                role=ROLE_NONNUMERIC,
                exclusion_kind=kind,
                renderer_family=registration.renderer_family,
                degradation_family=registration.degradation_family,
                raster=_render_negative(split, kind, rng),
            )
        )
    return tuple(positives + negatives)


def encode_slots(text: str) -> tuple[int, ...]:
    encoded = [ALPHABET.index(character) + 1 for character in text]
    return tuple(encoded + [BLANK_CLASS_INDEX] * (MAX_TOKENS - len(encoded)))


def prepare_inputs(samples: tuple[NumericSample, ...] | list[NumericSample]) -> torch.Tensor:
    array = np.stack([sample.raster for sample in samples]).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(1)


def split_fingerprint(samples: tuple[NumericSample, ...]) -> str:
    records = []
    for sample in samples:
        records.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "target_text": sample.target_text,
                "case": sample.case,
                "role": sample.role,
                "exclusion_kind": sample.exclusion_kind,
                "renderer_family": sample.renderer_family,
                "degradation_family": sample.degradation_family,
                "raster_sha256": sha256(sample.raster.tobytes()).hexdigest(),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def split_metadata(samples: tuple[NumericSample, ...]) -> dict[str, object]:
    positives = [sample for sample in samples if sample.role == ROLE_NUMERIC_TEXT]
    negatives = [sample for sample in samples if sample.role == ROLE_NONNUMERIC]
    return {
        "split": samples[0].split,
        "count": len(samples),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "cases": sorted({sample.case for sample in positives}),
        "exclusion_kinds": sorted({sample.exclusion_kind for sample in negatives}),
        "renderer_families": sorted({sample.renderer_family for sample in samples}),
        "degradation_families": sorted({sample.degradation_family for sample in samples}),
        "fingerprint_sha256": split_fingerprint(samples),
    }
