# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Fixed procedural corpus with dense glyph-span alignment labels."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Literal

import numpy as np
import torch
from torch.nn import functional

from ml.ocr.synthetic import RenderFamily, Raster, _FAMILIES, _render, _sample_rng

ALPHABET = "0123456789.-%"
BLANK_CLASS_INDEX = 0
INPUT_HEIGHT = 32
INPUT_WIDTH = 128
TIME_STEPS = 32
CLASS_COUNT = len(ALPHABET) + 1
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SequenceSample:
    sample_id: str
    split: Split
    display_text: str
    target_text: str
    case: str
    raster: Raster
    orientation_degrees: int
    family: RenderFamily


@dataclass(frozen=True)
class SequenceCorpus:
    seed: int
    train: tuple[SequenceSample, ...]
    validation: tuple[SequenceSample, ...]
    test: tuple[SequenceSample, ...]

    def all_samples(self) -> tuple[SequenceSample, ...]:
        return self.train + self.validation + self.test


def _label(index: int, rng: random.Random) -> tuple[str, str, str]:
    case = index % 6
    if case == 0:
        target = str(rng.randrange(0, 1001))
        return target, target, "plain"
    if case == 1:
        target = f"{rng.randrange(0, 101)}.{rng.randrange(0, 10)}"
        return target, target, "decimal"
    if case == 2:
        target = f"-{rng.randrange(0, 101)}"
        return target, target, "negative"
    if case == 3:
        target = f"{rng.randrange(0, 101)}%"
        return target, target, "percent"
    if case == 4:
        target = rng.choice(("0", "10", "20", "50", "100"))
        return target.replace("0", "O", 1), target, "o_zero_ambiguity"

    target = rng.choice(("1", "10", "11", "21", "101"))
    return target.replace("1", "l", 1), target, "l_one_ambiguity"


def _make_split(seed: int, split: Split, count: int) -> tuple[SequenceSample, ...]:
    family = _FAMILIES[split]
    samples = []
    for index in range(count):
        rng = _sample_rng(seed, split, index + 20_000)
        display, target, case = _label(index, rng)
        orientation = 0
        render_case = case
        if split != "train" and index % 20 == 18:
            orientation = 90
            render_case = "rotated_label"
        elif split != "train" and index % 20 == 19:
            orientation = 270
            render_case = "rotated_label"
        elif split == "test" and index % 20 == 17:
            render_case = "tiny_digits"
        elif split == "test" and index % 20 == 16:
            render_case = "faded_digits"
        samples.append(
            SequenceSample(
                sample_id=f"sequence-v2-{split}-{index:04d}",
                split=split,
                display_text=display,
                target_text=target,
                case=render_case,
                raster=_render(display, family, render_case, orientation, rng),
                orientation_degrees=orientation,
                family=family,
            )
        )
    return tuple(samples)


def build_corpus(
    seed: int = 20260804,
    train_count: int = 1024,
    validation_count: int = 256,
    test_count: int = 256,
) -> SequenceCorpus:
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Every split must contain at least one sample")
    return SequenceCorpus(
        seed=seed,
        train=_make_split(seed, "train", train_count),
        validation=_make_split(seed, "validation", validation_count),
        test=_make_split(seed, "test", test_count),
    )


def manifest_sha256(corpus: SequenceCorpus) -> str:
    records = []
    for sample in corpus.all_samples():
        raster = bytes(value for row in sample.raster for value in row)
        records.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "display_text": sample.display_text,
                "target_text": sample.target_text,
                "case": sample.case,
                "orientation_degrees": sample.orientation_degrees,
                "family": sample.family.__dict__,
                "raster_sha256": sha256(raster).hexdigest(),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _upright(sample: SequenceSample) -> np.ndarray:
    raster = np.asarray(sample.raster, dtype=np.float32)
    if sample.orientation_degrees:
        raster = np.rot90(raster, k=sample.orientation_degrees // 90).copy()
    return raster


def _spans(raster: np.ndarray) -> list[tuple[int, int]]:
    threshold = (float(raster.min()) + float(raster.max())) / 2.0
    ink = raster <= threshold
    active = ink.any(axis=0).tolist() + [False]
    result = []
    start: int | None = None
    for column, is_active in enumerate(active):
        if is_active and start is None:
            start = column
        elif not is_active and start is not None:
            result.append((start, column))
            start = None
    return result


def prepare(sample: SequenceSample) -> tuple[torch.Tensor, torch.Tensor]:
    raster = _upright(sample)
    spans = _spans(raster)
    if len(spans) != len(sample.target_text):
        raise ValueError(
            f"{sample.sample_id} has {len(spans)} glyph spans for {len(sample.target_text)} targets"
        )

    source = torch.from_numpy(raster / 255.0).unsqueeze(0).unsqueeze(0)
    resized = functional.interpolate(
        source,
        size=(INPUT_HEIGHT, INPUT_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    inputs = (resized.squeeze(0) - 0.5) * 2.0

    aligned = torch.zeros(TIME_STEPS, dtype=torch.long)
    source_width = raster.shape[1]
    for character, (left, right) in zip(sample.target_text, spans, strict=True):
        character_class = ALPHABET.index(character) + 1
        positions = [
            position
            for position in range(TIME_STEPS)
            if left <= ((position + 0.5) * source_width / TIME_STEPS) < right
        ]
        if not positions:
            center = (left + right) / 2.0
            positions = [
                min(
                    range(TIME_STEPS),
                    key=lambda position: abs(((position + 0.5) * source_width / TIME_STEPS) - center),
                )
            ]
        for position in positions:
            if aligned[position] != BLANK_CLASS_INDEX:
                raise ValueError(f"{sample.sample_id} has colliding glyph alignment at {position}")
            aligned[position] = character_class
    return inputs, aligned


def decode(logits: torch.Tensor) -> list[str]:
    classes = logits.argmax(dim=-1).detach().cpu().tolist()
    results = []
    for sequence in classes:
        prior = -1
        output = []
        for class_index in sequence:
            if class_index != BLANK_CLASS_INDEX and class_index != prior:
                output.append(ALPHABET[class_index - 1])
            prior = class_index
        results.append("".join(output))
    return results
