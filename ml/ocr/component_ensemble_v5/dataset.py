# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Procedural multi-renderer labels and hidden public archive for OCR V5."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.ocr.component_geometric_v4.dataset import _filtered_foreground, _normalize_glyph

from .protocol import (
    ENCODED_GLYPH_WIDTH,
    EXCLUSION_KINDS,
    GEOMETRY_FEATURE_COUNT,
    GLYPH_HEIGHT,
    GLYPH_WIDTH,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CANVAS_HEIGHT = 32
CANVAS_WIDTH = 128
Split = Literal["train", "validation", "sealed_public"]


@dataclass(frozen=True)
class LabelSample:
    sample_id: str
    split: Split
    target_text: str
    display_text: str
    case: str
    role: str
    exclusion_kind: str | None
    renderer_family: str
    degradation_family: str
    raster: np.ndarray


def _rng(split: Split, index: int, role_offset: int = 0) -> np.random.Generator:
    registration = split_registration(split)
    material = f"component-ensemble-v5:{registration.seed_offset}:{split}:{role_offset}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _label_for(index: int, rng: np.random.Generator) -> tuple[str, str, str]:
    case = index % 8
    if case == 0:
        target = str(int(rng.integers(0, 10_001)))
        return target, target, "integer"
    if case == 1:
        target = f"{int(rng.integers(0, 1001))}.{int(rng.integers(0, 10))}"
        return target, target, "decimal"
    if case == 2:
        target = f"-{int(rng.integers(0, 1001))}"
        return target, target, "negative"
    if case == 3:
        target = f"{int(rng.integers(0, 101))}%"
        return target, target, "percentage"
    if case == 4:
        target = f"-{int(rng.integers(0, 101))}.{int(rng.integers(0, 10))}"
        return target, target, "negative_decimal"
    if case == 5:
        target = f"{int(rng.integers(0, 101))}.{int(rng.integers(0, 10))}%"
        return target, target, "decimal_percentage"
    if case == 6:
        target = str(rng.choice(np.asarray(("0", "10", "20", "50", "100", "0.0"))))
        return target.replace("0", "O"), target, "o_zero_ambiguity"
    target = str(rng.choice(np.asarray(("1", "10", "11", "21", "101", "1.1"))))
    return target.replace("1", "l"), target, "l_one_ambiguity"


def _font(split: Split, size: int, rng: np.random.Generator) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    index = int(rng.integers(0, len(registration.font_paths)))
    return ImageFont.truetype(str(REPO_ROOT / registration.font_paths[index]), size=size)


def _degrade(image: Image.Image, split: Split, rng: np.random.Generator, background: int) -> np.ndarray:
    registration = split_registration(split)
    if split == "train":
        action = int(rng.integers(0, 4))
        if action == 1:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.choice((0.25, 0.45, 0.65)))))
        elif action == 2:
            reduced = image.resize(
                (max(1, image.width - image.width // 12), max(1, image.height - image.height // 12)),
                resample=Image.Resampling.BILINEAR,
            )
            image = reduced.resize(image.size, resample=Image.Resampling.BILINEAR)
        elif action == 3:
            fade = int(rng.integers(6, 18))
            image = image.point(lambda value: min(255, value + fade))
    elif split == "sealed_public":
        shear = float(rng.choice((-0.055, -0.035, 0.035, 0.055)))
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, shear, -shear * image.height / 2.0, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=background,
        )
    raster = np.asarray(
        image.resize((CANVAS_WIDTH, CANVAS_HEIGHT), resample=Image.Resampling.LANCZOS),
        dtype=np.uint8,
    ).copy()
    if split == "train":
        for _ in range(int(rng.integers(0, 10))):
            raster[int(rng.integers(0, CANVAS_HEIGHT)), int(rng.integers(0, CANVAS_WIDTH))] = int(
                rng.integers(190, 246)
            )
    elif split == "validation":
        if int(rng.integers(0, 2)) == 0:
            small = Image.fromarray(raster).resize((104, 26), resample=Image.Resampling.BILINEAR)
            raster = np.asarray(
                small.resize((CANVAS_WIDTH, CANVAS_HEIGHT), resample=Image.Resampling.BILINEAR),
                dtype=np.uint8,
            ).copy()
        if int(rng.integers(0, 3)) == 0:
            row = int(rng.integers(2, CANVAS_HEIGHT - 2))
            raster[row, :] = np.minimum(raster[row, :], int(rng.integers(220, 244)))
    else:
        foreground = np.argwhere(raster < 190)
        if len(foreground) and int(rng.integers(0, 2)) == 0:
            start = int(rng.integers(0, max(1, len(foreground) - 3)))
            for y, x in foreground[start : start + int(rng.integers(1, 4))]:
                raster[int(y), int(x)] = int(rng.integers(228, 256))
        if int(rng.integers(0, 3)) == 0:
            fade = int(rng.integers(8, 22))
            raster = np.minimum(255, raster.astype(np.int16) + fade).astype(np.uint8)
    return raster


def render_label(split: Split, display_text: str, rng: np.random.Generator) -> np.ndarray:
    registration = split_registration(split)
    scale = registration.supersample
    background = int(rng.integers(247, 256))
    ink = int(rng.integers(3, 70))
    base_size = int(rng.integers(18, 24)) if len(display_text) <= 4 else int(rng.integers(16, 20))
    font = _font(split, base_size * scale, rng)
    spacing = int(rng.integers(2, 5)) * scale
    boxes = [font.getbbox(character) for character in display_text]
    widths = [max(1, box[2] - box[0]) for box in boxes]
    heights = [max(1, box[3] - box[1]) for box in boxes]
    total_width = sum(widths) + spacing * max(0, len(widths) - 1)
    canvas_width = CANVAS_WIDTH * scale
    canvas_height = CANVAS_HEIGHT * scale
    if total_width > canvas_width - 4 * scale:
        shrink = max(12 * scale, int(base_size * scale * (canvas_width - 8 * scale) / total_width))
        font = _font(split, shrink, rng)
        boxes = [font.getbbox(character) for character in display_text]
        widths = [max(1, box[2] - box[0]) for box in boxes]
        heights = [max(1, box[3] - box[1]) for box in boxes]
        total_width = sum(widths) + spacing * max(0, len(widths) - 1)
    origin_x = max(2 * scale, (canvas_width - total_width) // 2 + int(rng.integers(-2, 3)) * scale)
    image = Image.new("L", (canvas_width, canvas_height), color=background)
    draw = ImageDraw.Draw(image)
    cursor = origin_x
    for character, box, width, height in zip(display_text, boxes, widths, heights, strict=True):
        top = max(scale, (canvas_height - height) // 2 + int(rng.integers(-1, 2)) * scale)
        draw.text((cursor - box[0], top - box[1]), character, fill=ink, font=font)
        cursor += width + spacing
    return _degrade(image, split, rng, background)


def _draw_negative(split: Split, kind: str, rng: np.random.Generator) -> np.ndarray:
    registration = split_registration(split)
    scale = registration.supersample
    background = int(rng.integers(247, 256))
    ink = int(rng.integers(3, 70))
    image = Image.new("L", (CANVAS_WIDTH * scale, CANVAS_HEIGHT * scale), color=background)
    draw = ImageDraw.Draw(image)
    cx = (CANVAS_WIDTH // 2 + int(rng.integers(-10, 11))) * scale
    cy = (CANVAS_HEIGHT // 2 + int(rng.integers(-3, 4))) * scale
    width = int(rng.integers(1, 3)) * scale
    if kind in {"filled_circle", "open_circle"}:
        radius = int(rng.integers(4, 8)) * scale
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(box, fill=ink if kind == "filled_circle" else None, outline=ink, width=width)
    elif kind == "axis_or_tick":
        draw.line((18 * scale, cy, 110 * scale, cy), fill=ink, width=width)
        draw.line((cx, cy - 6 * scale, cx, cy + 6 * scale), fill=ink, width=width)
    elif kind == "divider":
        draw.line((cx, 2 * scale, cx, 29 * scale), fill=ink, width=width)
    elif kind == "bracket":
        draw.line((cx - 18 * scale, 6 * scale, cx - 18 * scale, 26 * scale), fill=ink, width=width)
        draw.line((cx - 18 * scale, 6 * scale, cx + 18 * scale, 6 * scale), fill=ink, width=width)
        draw.line((cx - 18 * scale, 26 * scale, cx + 18 * scale, 26 * scale), fill=ink, width=width)
    elif kind == "arrow":
        draw.line((cx - 24 * scale, cy, cx + 20 * scale, cy), fill=ink, width=width)
        draw.line((cx + 20 * scale, cy, cx + 11 * scale, cy - 7 * scale), fill=ink, width=width)
        draw.line((cx + 20 * scale, cy, cx + 11 * scale, cy + 7 * scale), fill=ink, width=width)
    elif kind == "legend_box":
        draw.rectangle((cx - 24 * scale, 5 * scale, cx + 24 * scale, 27 * scale), outline=ink, width=width)
    elif kind == "line_intersection":
        draw.line((cx - 28 * scale, cy - 10 * scale, cx + 28 * scale, cy + 10 * scale), fill=ink, width=width)
        draw.line((cx - 28 * scale, cy + 10 * scale, cx + 28 * scale, cy - 10 * scale), fill=ink, width=width)
    else:
        draw.rectangle((cx - 6 * scale, cy - 6 * scale, cx + 6 * scale, cy + 6 * scale), fill=ink)
    return _degrade(image, split, rng, background)


def build_split(split: Split) -> tuple[LabelSample, ...]:
    registration = split_registration(split)
    samples: list[LabelSample] = []
    for index in range(registration.positive_count):
        rng = _rng(split, index)
        display, target, case = _label_for(index, rng)
        samples.append(
            LabelSample(
                sample_id=f"component-ensemble-v5-{split}-text-{index:05d}",
                split=split,
                target_text=target,
                display_text=display,
                case=case,
                role="numeric_text",
                exclusion_kind=None,
                renderer_family=registration.renderer_family,
                degradation_family=registration.degradation_family,
                raster=render_label(split, display, rng),
            )
        )
    for index in range(registration.negative_count):
        rng = _rng(split, index, role_offset=1)
        kind = EXCLUSION_KINDS[index % len(EXCLUSION_KINDS)]
        samples.append(
            LabelSample(
                sample_id=f"component-ensemble-v5-{split}-exclusion-{index:05d}",
                split=split,
                target_text="",
                display_text="",
                case="exclusion",
                role="non_numeric",
                exclusion_kind=kind,
                renderer_family=registration.renderer_family,
                degradation_family=registration.degradation_family,
                raster=_draw_negative(split, kind, rng),
            )
        )
    return tuple(samples)


def _encode(raster: np.ndarray, mask: np.ndarray, component_width: int) -> np.ndarray:
    normalized = _normalize_glyph(raster, mask)
    ys, xs = np.nonzero(mask)
    height = int(ys.max() - ys.min() + 1)
    area = max(1, height * component_width)
    foreground_values = raster[mask].astype(np.float32)
    geometry = np.asarray(
        (
            height / CANVAS_HEIGHT,
            component_width / CANVAS_HEIGHT,
            ((float(ys.min()) + float(ys.max())) / 2.0) / (CANVAS_HEIGHT - 1),
            float(mask.sum()) / area,
            float(foreground_values.mean()) / 255.0,
            component_width / max(1, height),
        ),
        dtype=np.float32,
    )
    columns = np.broadcast_to(
        geometry[np.newaxis, np.newaxis, :],
        (1, GLYPH_HEIGHT, GEOMETRY_FEATURE_COUNT),
    ).copy()
    encoded = np.concatenate((normalized, columns), axis=2).astype(np.float32)
    if encoded.shape != (1, GLYPH_HEIGHT, ENCODED_GLYPH_WIDTH):
        raise RuntimeError("OCR V5 encoded glyph shape is invalid")
    return encoded


def isolate_glyphs(raster: np.ndarray) -> tuple[np.ndarray, ...]:
    foreground = _filtered_foreground(raster)
    active = np.any(foreground, axis=0)
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            intervals.append((start, index if value and index == len(active) - 1 else index - 1))
            start = None
    merged: list[tuple[int, int]] = []
    for left, right in intervals:
        if merged and left - merged[-1][1] - 1 <= 1:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    glyphs: list[np.ndarray] = []
    for left, right in merged:
        local_mask = foreground[:, left : right + 1]
        if int(local_mask.sum()) < 2:
            continue
        glyphs.append(_encode(raster[:, left : right + 1], local_mask, right - left + 1))
    return tuple(glyphs)


def split_fingerprint(samples: tuple[LabelSample, ...]) -> str:
    digest = sha256()
    for sample in samples:
        for value in (
            sample.sample_id,
            sample.target_text,
            sample.display_text,
            sample.exclusion_kind or "",
            sample.renderer_family,
            sample.degradation_family,
        ):
            digest.update(value.encode())
            digest.update(b"\0")
        digest.update(sample.raster.tobytes())
    return digest.hexdigest()


def save_sealed_public_archive(samples: tuple[LabelSample, ...], path: Path) -> dict[str, object]:
    if not samples or any(sample.split != "sealed_public" for sample in samples):
        raise ValueError("Only the complete OCR V5 sealed-public split may be archived")
    arrays = {
        "rasters": np.stack([sample.raster for sample in samples]).astype(np.uint8),
        "sample_ids": np.asarray([sample.sample_id for sample in samples]),
        "target_texts": np.asarray([sample.target_text for sample in samples]),
        "display_texts": np.asarray([sample.display_text for sample in samples]),
        "cases": np.asarray([sample.case for sample in samples]),
        "roles": np.asarray([sample.role for sample in samples]),
        "exclusion_kinds": np.asarray([sample.exclusion_kind or "" for sample in samples]),
        "renderer_families": np.asarray([sample.renderer_family for sample in samples]),
        "degradation_families": np.asarray([sample.degradation_family for sample in samples]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, mode="x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            stream = BytesIO()
            np.lib.format.write_array(stream, arrays[name], allow_pickle=False)
            entry = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, stream.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=9)
    return {
        "schema": "graphreader.ocr-component-ensemble-private-manifest.v1",
        "split": "sealed_public",
        "sample_count": len(samples),
        "positive_count": sum(sample.exclusion_kind is None for sample in samples),
        "exclusion_count": sum(sample.exclusion_kind is not None for sample in samples),
        "split_fingerprint": split_fingerprint(samples),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "target_text": sample.target_text,
                "display_text": sample.display_text,
                "case": sample.case,
                "role": sample.role,
                "exclusion_kind": sample.exclusion_kind,
                "raster_sha256": sha256(sample.raster.tobytes()).hexdigest(),
            }
            for sample in samples
        ],
    }


def load_sealed_public_archive(path: Path) -> tuple[LabelSample, ...]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "rasters",
            "sample_ids",
            "target_texts",
            "display_texts",
            "cases",
            "roles",
            "exclusion_kinds",
            "renderer_families",
            "degradation_families",
        }
        if set(archive.files) != required:
            raise RuntimeError("OCR V5 sealed archive inventory is invalid")
        rasters = np.asarray(archive["rasters"], dtype=np.uint8)
        values = {name: np.asarray(archive[name]).astype(str) for name in required - {"rasters"}}
    count = len(rasters)
    if rasters.shape != (count, CANVAS_HEIGHT, CANVAS_WIDTH):
        raise RuntimeError("OCR V5 sealed raster tensor is invalid")
    if any(len(value) != count for value in values.values()):
        raise RuntimeError("OCR V5 sealed metadata lengths are invalid")
    return tuple(
        LabelSample(
            sample_id=values["sample_ids"][index],
            split="sealed_public",
            target_text=values["target_texts"][index],
            display_text=values["display_texts"][index],
            case=values["cases"][index],
            role=values["roles"][index],
            exclusion_kind=values["exclusion_kinds"][index] or None,
            renderer_family=values["renderer_families"][index],
            degradation_family=values["degradation_families"][index],
            raster=rasters[index],
        )
        for index in range(count)
    )


__all__ = [
    "LabelSample",
    "build_split",
    "isolate_glyphs",
    "load_sealed_public_archive",
    "save_sealed_public_archive",
    "split_fingerprint",
]
