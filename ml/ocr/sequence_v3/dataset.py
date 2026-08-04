# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Fixed procedural corpus and canonical glyph-slot preprocessing."""

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
SLOT_COUNT = TIME_STEPS // 2
SLOT_WIDTH = INPUT_WIDTH // SLOT_COUNT
CLASS_COUNT = len(ALPHABET) + 1
Split = Literal["train", "validation", "test"]
GENERATOR_SCOPE = "shared-procedural-glyph-matrix-and-render-function"
FAMILY_IMPLEMENTATIONS_INDEPENDENT = False
CANONICALIZER_INVERTS_SHARED_GENERATOR_WIDTH_TRANSFORMS = True


@dataclass(frozen=True)
class SlotSample:
    sample_id: str
    split: Split
    display_text: str
    target_text: str
    case: str
    raster: Raster
    orientation_degrees: int
    family: RenderFamily


@dataclass(frozen=True)
class SlotCorpus:
    seed: int
    train: tuple[SlotSample, ...]
    validation: tuple[SlotSample, ...]
    test: tuple[SlotSample, ...]

    def all_samples(self) -> tuple[SlotSample, ...]:
        return self.train + self.validation + self.test


def _label(index: int, rng: random.Random) -> tuple[str, str, str]:
    case = index % 7
    if case == 0:
        target = str(rng.randrange(0, 10001))
        return target, target, "plain"
    if case == 1:
        target = f"{rng.randrange(0, 1001)}.{rng.randrange(0, 100):02d}"
        return target, target, "decimal"
    if case == 2:
        target = f"-{rng.randrange(0, 1001)}"
        return target, target, "negative"
    if case == 3:
        target = f"{rng.randrange(0, 101)}%"
        return target, target, "percent"
    if case == 4:
        target = rng.choice(("0", "10", "20", "50", "100", "1000"))
        return target.replace("0", "O", 1), target, "o_zero_ambiguity"
    if case == 5:
        target = rng.choice(("1", "10", "11", "21", "101", "1001"))
        return target.replace("1", "l", 1), target, "l_one_ambiguity"
    target = rng.choice(("0.25", "-0.5", "12.5%", "100.00", "-99.9"))
    return target, target, "mixed"


def _make_split(seed: int, split: Split, count: int) -> tuple[SlotSample, ...]:
    family = _FAMILIES[split]
    samples = []
    for index in range(count):
        rng = _sample_rng(seed, split, index + 30_000)
        display, target, case = _label(index, rng)
        orientation = 0
        render_case = case
        if split != "train" and index % 24 == 22:
            orientation = 90
            render_case = "rotated_label"
        elif split != "train" and index % 24 == 23:
            orientation = 270
            render_case = "rotated_label"
        elif split == "test" and index % 24 == 21:
            render_case = "tiny_digits"
        elif split == "test" and index % 24 == 20:
            render_case = "faded_digits"
        samples.append(
            SlotSample(
                sample_id=f"sequence-v3-{split}-{index:04d}",
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
    train_count: int = 2048,
    validation_count: int = 512,
    test_count: int = 512,
) -> SlotCorpus:
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Every split must contain at least one sample")
    return SlotCorpus(
        seed=seed,
        train=_make_split(seed, "train", train_count),
        validation=_make_split(seed, "validation", validation_count),
        test=_make_split(seed, "test", test_count),
    )


def manifest_sha256(corpus: SlotCorpus) -> str:
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


def _upright(sample: SlotSample) -> np.ndarray:
    raster = np.asarray(sample.raster, dtype=np.float32)
    if sample.orientation_degrees:
        raster = np.rot90(raster, k=sample.orientation_degrees // 90).copy()
    return raster


def _glyph_spans(raster: np.ndarray) -> list[tuple[int, int]]:
    threshold = (float(raster.min()) + float(raster.max())) / 2.0
    ink = raster <= threshold
    active = ink.any(axis=0).tolist() + [False]
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for column, is_active in enumerate(active):
        if is_active and start is None:
            start = column
        elif not is_active and start is not None:
            spans.append((start, column))
            start = None
    return spans


def _canonical_glyph(raster: np.ndarray, left: int, right: int) -> torch.Tensor:
    threshold = (float(raster.min()) + float(raster.max())) / 2.0
    ink = raster[:, left:right] <= threshold
    active_rows = np.flatnonzero(ink.any(axis=1))
    if active_rows.size == 0:
        raise ValueError("Glyph span contains no ink")
    cropped = ink[active_rows[0] : active_rows[-1] + 1, :].astype(np.float32)
    source = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0)
    raster_scale = max(1, round(cropped.shape[0] / 7))
    topology_width = max(1, round(cropped.shape[1] / raster_scale))
    topology = functional.interpolate(
        source,
        size=(7, topology_width),
        mode="nearest",
    )
    if topology_width == 6:
        columns = topology[0, 0].transpose(0, 1)
        duplicate = next(
            (
                index
                for index in range(1, topology_width)
                if torch.equal(columns[index - 1], columns[index])
            ),
            3,
        )
        topology = torch.cat(
            (topology[:, :, :, :duplicate], topology[:, :, :, duplicate + 1 :]),
            dim=3,
        )
    elif topology_width == 4:
        topology = torch.cat(
            (topology[:, :, :, :3], topology[:, :, :, 2:3], topology[:, :, :, 3:]),
            dim=3,
        )
    elif topology_width != 5:
        topology = functional.interpolate(topology, size=(7, 5), mode="nearest")
    resized = functional.interpolate(
        topology,
        size=(INPUT_HEIGHT - 8, SLOT_WIDTH - 2),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0)


def prepare(sample: SlotSample) -> tuple[torch.Tensor, torch.Tensor]:
    raster = _upright(sample)
    spans = _glyph_spans(raster)
    if len(spans) != len(sample.target_text):
        raise ValueError(
            f"{sample.sample_id} has {len(spans)} glyph spans for "
            f"{len(sample.target_text)} targets"
        )
    if len(spans) > SLOT_COUNT:
        raise ValueError(f"{sample.sample_id} exceeds {SLOT_COUNT} glyph slots")

    packed = torch.zeros((1, INPUT_HEIGHT, INPUT_WIDTH), dtype=torch.float32)
    targets = torch.zeros(TIME_STEPS, dtype=torch.long)
    for slot, (character, (left, right)) in enumerate(
        zip(sample.target_text, spans, strict=True)
    ):
        glyph = _canonical_glyph(raster, left, right)
        x = slot * SLOT_WIDTH + 1
        packed[0, 4 : INPUT_HEIGHT - 4, x : x + SLOT_WIDTH - 2] = glyph
        targets[slot * 2] = ALPHABET.index(character) + 1
    return packed, targets


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
