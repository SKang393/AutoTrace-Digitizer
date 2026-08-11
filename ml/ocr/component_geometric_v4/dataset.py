# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Procedural labels and deterministic glyph isolation for OCR V4."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .protocol import (
    EXCLUSION_KINDS,
    GLYPH_HEIGHT,
    GLYPH_WIDTH,
    SEED,
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
    material = f"{SEED}:{registration.seed_offset}:{split}:{role_offset}:{index}".encode()
    value = int.from_bytes(sha256(material).digest()[:8], "little")
    return np.random.default_rng(value)


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


def _font(split: Split, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    return ImageFont.truetype(str(REPO_ROOT / registration.font_path), size=size)


def _apply_degradation(image: Image.Image, split: Split, rng: np.random.Generator) -> np.ndarray:
    if split == "train" and int(rng.integers(0, 3)) == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.choice((0.2, 0.35)))))
    if split == "validation" and int(rng.integers(0, 2)) == 0:
        reduced = image.resize((96, 24), resample=Image.Resampling.BILINEAR)
        image = reduced.resize((CANVAS_WIDTH, CANVAS_HEIGHT), resample=Image.Resampling.BILINEAR)
    raster = np.asarray(image, dtype=np.uint8).copy()
    if split == "train":
        for _ in range(int(rng.integers(0, 7))):
            raster[int(rng.integers(0, CANVAS_HEIGHT)), int(rng.integers(0, CANVAS_WIDTH))] = int(
                rng.integers(180, 236)
            )
    elif split == "validation" and int(rng.integers(0, 4)) == 0:
        row = int(rng.integers(2, CANVAS_HEIGHT - 2))
        raster[row, :] = np.minimum(raster[row, :], int(rng.integers(215, 241)))
    elif split == "sealed_public":
        foreground = np.argwhere(raster < 180)
        if len(foreground) and int(rng.integers(0, 3)) == 0:
            count = min(len(foreground), int(rng.integers(1, 5)))
            for selected in rng.choice(len(foreground), size=count, replace=False):
                y, x = foreground[int(selected)]
                raster[y, x] = int(rng.integers(225, 256))
    return raster


def render_label(split: Split, display_text: str, rng: np.random.Generator) -> np.ndarray:
    background = int(rng.integers(246, 256))
    ink = int(rng.integers(5, 76))
    font_size = int(rng.integers(18, 24)) if len(display_text) <= 4 else int(rng.integers(16, 20))
    font = _font(split, font_size)
    spacing = int(rng.integers(2, 5))
    boxes = [font.getbbox(character) for character in display_text]
    widths = [max(1, box[2] - box[0]) for box in boxes]
    heights = [max(1, box[3] - box[1]) for box in boxes]
    total_width = sum(widths) + spacing * max(0, len(widths) - 1)
    if total_width > CANVAS_WIDTH - 4:
        return render_label(split, display_text, np.random.default_rng(int(rng.integers(0, 2**32))))
    origin_x = max(2, (CANVAS_WIDTH - total_width) // 2 + int(rng.integers(-2, 3)))
    image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=background)
    draw = ImageDraw.Draw(image)
    cursor = origin_x
    for character, box, width, height in zip(display_text, boxes, widths, heights, strict=True):
        top = max(1, (CANVAS_HEIGHT - height) // 2 + int(rng.integers(-1, 2)))
        draw.text((cursor - box[0], top - box[1]), character, fill=ink, font=font)
        cursor += width + spacing
    return _apply_degradation(image, split, rng)


def _draw_negative(split: Split, kind: str, rng: np.random.Generator) -> np.ndarray:
    background = int(rng.integers(246, 256))
    ink = int(rng.integers(5, 76))
    image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=background)
    draw = ImageDraw.Draw(image)
    cx = CANVAS_WIDTH // 2 + int(rng.integers(-10, 11))
    cy = CANVAS_HEIGHT // 2 + int(rng.integers(-3, 4))
    width = int(rng.integers(1, 3))
    if kind in {"filled_circle", "open_circle"}:
        radius = int(rng.integers(4, 8))
        bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bounds, fill=ink if kind == "filled_circle" else None, outline=ink, width=width)
    elif kind == "axis_or_tick":
        draw.line((18, cy, 110, cy), fill=ink, width=width)
        draw.line((cx, cy - 6, cx, cy + 6), fill=ink, width=width)
    elif kind == "divider":
        draw.line((cx, 2, cx, 29), fill=ink, width=width)
    elif kind == "bracket":
        draw.line((cx - 18, 6, cx - 18, 26), fill=ink, width=width)
        draw.line((cx - 18, 6, cx + 18, 6), fill=ink, width=width)
        draw.line((cx - 18, 26, cx + 18, 26), fill=ink, width=width)
    elif kind == "arrow":
        draw.line((cx - 24, cy, cx + 20, cy), fill=ink, width=width)
        draw.line((cx + 20, cy, cx + 11, cy - 7), fill=ink, width=width)
        draw.line((cx + 20, cy, cx + 11, cy + 7), fill=ink, width=width)
    elif kind == "legend_box":
        draw.rectangle((cx - 24, 5, cx + 24, 27), outline=ink, width=width)
    elif kind == "line_intersection":
        draw.line((cx - 28, cy - 10, cx + 28, cy + 10), fill=ink, width=width)
        draw.line((cx - 28, cy + 10, cx + 28, cy - 10), fill=ink, width=width)
    else:
        draw.rectangle((cx - 6, cy - 6, cx + 6, cy + 6), fill=ink)
    return _apply_degradation(image, split, rng)


def build_split(split: Split) -> tuple[LabelSample, ...]:
    registration = split_registration(split)
    samples: list[LabelSample] = []
    for index in range(registration.positive_count):
        rng = _rng(split, index)
        display, target, case = _label_for(index, rng)
        samples.append(
            LabelSample(
                sample_id=f"component-geometric-v4-{split}-text-{index:05d}",
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
                sample_id=f"component-geometric-v4-{split}-exclusion-{index:05d}",
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


def _filtered_foreground(raster: np.ndarray) -> np.ndarray:
    if raster.shape != (CANVAS_HEIGHT, CANVAS_WIDTH) or raster.dtype != np.uint8:
        raise ValueError("OCR V4 raster must be uint8 [32,128]")
    low = int(raster.min())
    median = int(np.median(raster))
    threshold = min(232, max(low + 12, (low + median) // 2))
    raw = raster <= threshold
    visited = np.zeros_like(raw, dtype=bool)
    filtered = np.zeros_like(raw, dtype=bool)
    for y in range(CANVAS_HEIGHT):
        for x in range(CANVAS_WIDTH):
            if not raw[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            points: list[tuple[int, int]] = []
            while stack:
                px, py = stack.pop()
                points.append((px, py))
                for ny in range(max(0, py - 1), min(CANVAS_HEIGHT, py + 2)):
                    for nx in range(max(0, px - 1), min(CANVAS_WIDTH, px + 2)):
                        if raw[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
            if len(points) >= 2:
                for px, py in points:
                    filtered[py, px] = True
    return filtered


def _normalize_glyph(raster: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot normalize an empty glyph")
    crop = raster[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    source = Image.fromarray(crop, mode="L")
    scale = min((GLYPH_WIDTH - 4) / max(1, source.width), (GLYPH_HEIGHT - 4) / max(1, source.height))
    width = max(1, int(round(source.width * scale)))
    height = max(1, int(round(source.height * scale)))
    resized = source.resize((width, height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (GLYPH_WIDTH, GLYPH_HEIGHT), color=255)
    canvas.paste(resized, ((GLYPH_WIDTH - width) // 2, (GLYPH_HEIGHT - height) // 2))
    value = 1.0 - np.asarray(canvas, dtype=np.float32) / 255.0
    maximum = float(value.max())
    if maximum > 0:
        value /= maximum
    return value[np.newaxis, :, :].astype(np.float32)


def isolate_glyphs(raster: np.ndarray) -> tuple[np.ndarray, ...]:
    foreground = _filtered_foreground(raster)
    active = np.any(foreground, axis=0)
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index if value and index == len(active) - 1 else index - 1
            intervals.append((start, end))
            start = None
    glyphs: list[np.ndarray] = []
    for left, right in intervals:
        local_mask = foreground[:, left : right + 1]
        if int(local_mask.sum()) < 2:
            continue
        local_raster = raster[:, left : right + 1]
        glyphs.append(_normalize_glyph(local_raster, local_mask))
    return tuple(glyphs)


def split_fingerprint(samples: tuple[LabelSample, ...]) -> str:
    digest = sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode())
        digest.update(b"\0")
        digest.update(sample.target_text.encode())
        digest.update(b"\0")
        digest.update(sample.display_text.encode())
        digest.update(b"\0")
        digest.update((sample.exclusion_kind or "").encode())
        digest.update(b"\0")
        digest.update(sample.raster.tobytes())
    return digest.hexdigest()


def save_sealed_public_archive(samples: tuple[LabelSample, ...], path: Path) -> dict[str, object]:
    """Write a byte-deterministic, truth-bearing archive kept outside Git."""

    if not samples or any(sample.split != "sealed_public" for sample in samples):
        raise ValueError("Only the complete sealed-public split may be archived")
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
        "schema": "graphreader.ocr-component-geometric-private-manifest.v1",
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
            raise RuntimeError("OCR V4 sealed archive inventory is invalid")
        rasters = np.asarray(archive["rasters"], dtype=np.uint8)
        values = {name: np.asarray(archive[name]).astype(str) for name in required - {"rasters"}}
    count = len(rasters)
    if rasters.shape != (count, CANVAS_HEIGHT, CANVAS_WIDTH):
        raise RuntimeError("OCR V4 sealed raster tensor is invalid")
    if any(len(value) != count for value in values.values()):
        raise RuntimeError("OCR V4 sealed metadata lengths are invalid")
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
