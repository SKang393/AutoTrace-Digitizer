# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Deterministic procedural corpus and compact CTC graph-numeric recognizer."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .synthetic import RenderFamily, Raster, _FAMILIES, _render, _sample_rng

ALPHABET = "0123456789.-%"
BLANK_CLASS_INDEX = 0
INPUT_HEIGHT = 32
INPUT_WIDTH = 128
TIME_STEPS = 32
CLASS_COUNT = len(ALPHABET) + 1
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class CtcSample:
    sample_id: str
    split: Split
    display_text: str
    target_text: str
    case: str
    raster: Raster
    orientation_degrees: int
    family: RenderFamily


@dataclass(frozen=True)
class CtcCorpus:
    seed: int
    train: tuple[CtcSample, ...]
    validation: tuple[CtcSample, ...]
    test: tuple[CtcSample, ...]

    def all_samples(self) -> tuple[CtcSample, ...]:
        return self.train + self.validation + self.test


def _label_for(index: int, rng: random.Random) -> tuple[str, str, str]:
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


def _build_split(seed: int, split: Split, count: int) -> tuple[CtcSample, ...]:
    family = _FAMILIES[split]
    result = []
    for index in range(count):
        rng = _sample_rng(seed, split, index + 10_000)
        display, target, case = _label_for(index, rng)
        orientation = 0
        render_case = case
        if split != "train" and index % 16 == 14:
            orientation = 90
            render_case = "rotated_label"
        elif split != "train" and index % 16 == 15:
            orientation = 270
            render_case = "rotated_label"
        elif split == "test" and index % 16 == 13:
            render_case = "tiny_digits"
        elif split == "test" and index % 16 == 12:
            render_case = "faded_digits"

        result.append(
            CtcSample(
                sample_id=f"ctc-{split}-{index:04d}",
                split=split,
                display_text=display,
                target_text=target,
                case=render_case,
                raster=_render(display, family, render_case, orientation, rng),
                orientation_degrees=orientation,
                family=family,
            )
        )
    return tuple(result)


def build_ctc_corpus(
    seed: int = 20260803,
    train_count: int = 768,
    validation_count: int = 192,
    test_count: int = 192,
) -> CtcCorpus:
    """Build the fixed candidate corpus without writing generated samples."""

    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Every split must contain at least one sample")
    return CtcCorpus(
        seed=seed,
        train=_build_split(seed, "train", train_count),
        validation=_build_split(seed, "validation", validation_count),
        test=_build_split(seed, "test", test_count),
    )


def corpus_manifest_sha256(corpus: CtcCorpus) -> str:
    records = []
    for sample in corpus.all_samples():
        raster_bytes = bytes(pixel for row in sample.raster for pixel in row)
        records.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "display_text": sample.display_text,
                "target_text": sample.target_text,
                "case": sample.case,
                "orientation_degrees": sample.orientation_degrees,
                "family": sample.family.__dict__,
                "raster_sha256": sha256(raster_bytes).hexdigest(),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def prepare_input(sample: CtcSample) -> torch.Tensor:
    raster = np.asarray(sample.raster, dtype=np.float32) / 255.0
    if sample.orientation_degrees:
        raster = np.rot90(raster, k=sample.orientation_degrees // 90).copy()
    tensor = torch.from_numpy(raster).unsqueeze(0).unsqueeze(0)
    resized = functional.interpolate(
        tensor,
        size=(INPUT_HEIGHT, INPUT_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    return (resized.squeeze(0) - 0.5) * 2.0


def encode_target(text: str) -> tuple[int, ...]:
    try:
        return tuple(ALPHABET.index(character) + 1 for character in text)
    except ValueError as error:
        raise ValueError(f"Unsupported graph-numeric character in {text!r}") from error


def decode_logits(logits: torch.Tensor) -> list[str]:
    classes = logits.argmax(dim=-1).detach().cpu().tolist()
    predictions = []
    for sequence in classes:
        prior = -1
        output = []
        for class_index in sequence:
            if class_index != BLANK_CLASS_INDEX and class_index != prior:
                output.append(ALPHABET[class_index - 1])
            prior = class_index
        predictions.append("".join(output))
    return predictions


class CompactGraphNumericCtc(nn.Module):
    """Small runtime-compatible CNN with batch-major CTC logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(32, 40, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(40, CLASS_COUNT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        features = functional.adaptive_avg_pool2d(features, (1, TIME_STEPS))
        sequence = features.squeeze(2).transpose(1, 2)
        return self.classifier(sequence)
