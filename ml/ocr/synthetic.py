# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""In-memory synthetic graph labels with renderer-family holdouts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random
from typing import Literal

Raster = tuple[tuple[int, ...], ...]
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class RenderFamily:
    """A composite family used as the leakage-prevention boundary."""

    renderer: str
    font: str
    degradation: str


@dataclass(frozen=True)
class SyntheticLabelSample:
    sample_id: str
    split: Split
    display_text: str
    target_text: str
    case: str
    raster: Raster
    orientation_degrees: int
    family: RenderFamily


@dataclass(frozen=True)
class SyntheticCorpus:
    seed: int
    train: tuple[SyntheticLabelSample, ...]
    validation: tuple[SyntheticLabelSample, ...]
    test: tuple[SyntheticLabelSample, ...]

    def all_samples(self) -> tuple[SyntheticLabelSample, ...]:
        return self.train + self.validation + self.test


_GLYPHS: dict[str, tuple[str, ...]] = {
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
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
}

_ALIASES = {"O": "0", "l": "1"}

_FAMILIES: dict[Split, RenderFamily] = {
    "train": RenderFamily("pixel-grid", "square-5x7", "clean-print"),
    "validation": RenderFamily("soft-raster", "wide-6x7", "faded-copy"),
    "test": RenderFamily("micro-raster", "condensed-5x7", "adverse-scan"),
}

_TRAIN_CASES = (
    ("0123456789", "0123456789", "digit_inventory", 0),
    ("0.5", "0.5", "decimal", 0),
    ("-12.5", "-12.5", "negative", 0),
    ("25%", "25%", "percent", 0),
    ("O", "0", "o_zero_ambiguity", 0),
    ("l", "1", "l_one_ambiguity", 0),
    ("100", "100", "plain", 0),
    ("24", "24", "plain", 0),
)

_VALIDATION_CASES = (
    ("10", "10", "faded_digits", 0),
    ("2.5", "2.5", "decimal", 0),
    ("-20", "-20", "negative", 0),
    ("50%", "50%", "percent", 0),
    ("1O0", "100", "o_zero_ambiguity", 0),
    ("l0", "10", "l_one_ambiguity", 0),
)

_TEST_CASES = (
    ("7", "7", "tiny_digits", 0),
    ("100", "100", "faded_digits", 0),
    ("O", "0", "o_zero_ambiguity", 0),
    ("l", "1", "l_one_ambiguity", 0),
    ("0.25", "0.25", "decimal", 0),
    ("75%", "75%", "percent", 0),
    ("-5", "-5", "negative", 0),
    ("-20", "-20", "rotated_label", 90),
    ("100", "100", "rotated_label", 270),
)


def _sample_rng(seed: int, split: Split, index: int) -> random.Random:
    digest = sha256(f"{seed}:{split}:{index}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _font_glyph(character: str, font: str) -> tuple[str, ...]:
    canonical = _ALIASES.get(character, character)
    rows = _GLYPHS[canonical]
    if font == "square-5x7":
        return rows
    if font == "wide-6x7":
        # Duplicate the center column to vary width while preserving topology.
        return tuple(row[0:3] + row[2:5] for row in rows)
    if font == "condensed-5x7":
        # Remove one off-center column while preserving the digit spine.
        return tuple(row[0:3] + row[4:5] for row in rows)
    raise ValueError(f"Unknown font family: {font}")


def _rotate_clockwise(matrix: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*matrix[::-1], strict=True)]


def _render(
    text: str,
    family: RenderFamily,
    case: str,
    orientation_degrees: int,
    rng: random.Random,
) -> Raster:
    scale = 1 if family.renderer == "micro-raster" or case == "tiny_digits" else 2
    gap = 2 if scale == 1 else 4
    glyphs = [_font_glyph(character, family.font) for character in text]
    glyph_height = len(glyphs[0])
    binary: list[list[int]] = []
    for row_index in range(glyph_height):
        row: list[int] = []
        for glyph_index, glyph in enumerate(glyphs):
            if glyph_index:
                row.extend([0] * gap)
            row.extend(int(bit) for bit in glyph[row_index])
        expanded: list[int] = []
        for value in row:
            expanded.extend([value] * scale)
        binary.extend([expanded.copy() for _ in range(scale)])

    border = max(1, scale)
    width = len(binary[0]) + 2 * border
    binary = (
        [[0] * width for _ in range(border)]
        + [[0] * border + row + [0] * border for row in binary]
        + [[0] * width for _ in range(border)]
    )

    if orientation_degrees not in {0, 90, 180, 270}:
        raise ValueError("orientation_degrees must be a right angle")
    for _ in range(orientation_degrees // 90):
        binary = _rotate_clockwise(binary)

    if family.degradation == "clean-print":
        ink, paper, noise = 20, 250, 0
    elif family.degradation == "faded-copy":
        ink, paper, noise = 150, 244, 3
    else:
        ink = 132 if case == "faded_digits" else 72
        paper, noise = 238, 8

    grayscale: list[tuple[int, ...]] = []
    for row in binary:
        converted: list[int] = []
        for bit in row:
            center = ink if bit else paper
            value = center + (rng.randint(-noise, noise) if noise else 0)
            converted.append(max(0, min(255, value)))
        grayscale.append(tuple(converted))
    return tuple(grayscale)


def _make_split(seed: int, split: Split) -> tuple[SyntheticLabelSample, ...]:
    cases = {
        "train": _TRAIN_CASES,
        "validation": _VALIDATION_CASES,
        "test": _TEST_CASES,
    }[split]
    family = _FAMILIES[split]
    samples = []
    for index, (display, target, case, rotation) in enumerate(cases):
        samples.append(
            SyntheticLabelSample(
                sample_id=f"{split}-{index:03d}",
                split=split,
                display_text=display,
                target_text=target,
                case=case,
                raster=_render(
                    display,
                    family,
                    case,
                    rotation,
                    _sample_rng(seed, split, index),
                ),
                orientation_degrees=rotation,
                family=family,
            )
        )
    return tuple(samples)


def build_corpus(seed: int = 20260802) -> SyntheticCorpus:
    """Build the fixed in-memory corpus without writing images or labels."""

    return SyntheticCorpus(
        seed=seed,
        train=_make_split(seed, "train"),
        validation=_make_split(seed, "validation"),
        test=_make_split(seed, "test"),
    )
