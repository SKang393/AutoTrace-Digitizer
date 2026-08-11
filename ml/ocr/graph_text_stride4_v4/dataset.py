# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Disjoint procedural data for stride-4 graph text detection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from ml.ocr.graph_text_ignore_band_v3.dataset import (
    _production_resize,
    _shrink_target,
    _supervision_mask,
)

from .protocol import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PATCH_HEIGHT,
    PATCH_WIDTH,
    TILES_PER_SOURCE,
    TRAIN_SOURCE_COUNT,
    VALIDATION_EXCLUSION_COUNT,
    VALIDATION_TEXT_COUNT,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FONT_PATHS = (
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)
GENERIC_TEXT = (
    "0",
    "24",
    "-8",
    "4.25",
    "75%",
    "Baseline",
    "Intervention",
    "Maintenance",
    "Probe",
    "Jordan",
    "Case F",
    "Phase C",
    "Session 18",
    "Participant D",
    "Follow up",
    "Condition B",
)


@dataclass(frozen=True)
class TrainingTile:
    tile_id: str
    source_index: int
    kind: str
    left: int
    top: int
    bgr: np.ndarray
    target: np.ndarray
    supervision_mask: np.ndarray
    renderer_family: str
    degradation_family: str


@dataclass(frozen=True)
class EvaluationFrame:
    case_id: str
    kind: str
    source_png: bytes
    source_sha256: str
    detector_bgr: bytes
    detector_bgr_sha256: str
    truth_bbox: tuple[float, float, float, float] | None
    renderer_family: str
    degradation_family: str
    structure_family: str


@dataclass(frozen=True)
class _RenderedSource:
    image: Image.Image
    detector_bgr: np.ndarray
    target: np.ndarray
    truth_bbox: tuple[float, float, float, float] | None
    kind: str
    renderer_family: str
    degradation_family: str
    structure_family: str
    structure_center: tuple[int, int]


def _rng(split: str, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"graph-text-stride4-v4:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(rng: np.random.Generator, size: int) -> ImageFont.FreeTypeFont:
    path = REPO_ROOT / FONT_PATHS[int(rng.integers(0, len(FONT_PATHS)))]
    return ImageFont.truetype(str(path), size=size)


def _add_mask(
    masks: list[tuple[int, int, int, int]],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    masks.append((max(0, left), max(0, top), min(FRAME_WIDTH, right), min(FRAME_HEIGHT, bottom)))


def _draw_structures(
    draw: ImageDraw.ImageDraw,
    split: str,
    index: int,
) -> tuple[str, list[tuple[int, int, int, int]], tuple[int, int]]:
    registration = split_registration(split)
    masks: list[tuple[int, int, int, int]] = []
    families = ("chevron_grid", "double_brace", "legend_rail", "crossing_fan")
    split_index = ("train", "validation", "sealed_public").index(split)
    family = families[(index * 5 + (0, 1, 3)[split_index]) % len(families)]
    axis_x = (49, 61, 55)[split_index] + ((index * 7) % 6)
    axis_y = (166, 161, 174)[split_index] - ((index * 3) % 5)
    ink = (24, 27, 31)

    draw.line((axis_x, 17, axis_x, axis_y), fill=ink, width=2)
    draw.line((axis_x, axis_y, 371, axis_y), fill=ink, width=2)
    _add_mask(masks, axis_x - 5, 13, axis_x + 6, axis_y + 7)
    _add_mask(masks, axis_x - 5, axis_y - 5, 375, axis_y + 7)
    for tick in range(7):
        x = axis_x + 27 + tick * 44
        y = axis_y - 22 - ((tick * 31 + index * 9 + split_index * 11) % 108)
        draw.line((x, axis_y - 5, x, axis_y + 5), fill=ink, width=1)
        draw.line((axis_x - 5, y, axis_x + 5, y), fill=ink, width=1)
        _add_mask(masks, x - 2, axis_y - 7, x + 3, axis_y + 8)
        _add_mask(masks, axis_x - 7, y - 2, axis_x + 8, y + 3)

    points = (
        (axis_x + 25, axis_y - 34),
        (axis_x + 69, axis_y - 78),
        (axis_x + 116, axis_y - 51),
        (axis_x + 166, axis_y - 112),
        (axis_x + 224, axis_y - 70),
        (axis_x + 291, axis_y - 121),
    )
    draw.line(points, fill=ink, width=2, joint="curve")
    for marker_index, (x, y) in enumerate(points):
        radius = 3 + ((marker_index + index + split_index) % 3)
        if (marker_index + index) % 2:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, fill=(250, 252, 251), width=2)
        else:
            draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), fill=ink)

    divider_x = axis_x + 143 + ((index * 11) % 23)
    draw.line((divider_x, 15, divider_x, axis_y), fill=(37, 39, 42), width=2)
    _add_mask(masks, divider_x - 4, 11, divider_x + 5, axis_y + 5)

    if family == "chevron_grid":
        for offset in range(0, 64, 16):
            draw.line((272 + offset, 32, 279 + offset, 39, 272 + offset, 46), fill=ink, width=2)
        draw.line((255, 26, 361, 112), fill=ink, width=1)
        draw.line((259, 112, 357, 28), fill=ink, width=1)
        center = (307, 69)
    elif family == "double_brace":
        draw.arc((249, 20, 303, 105), start=75, end=285, fill=ink, width=2)
        draw.arc((282, 36, 348, 126), start=75, end=285, fill=ink, width=2)
        draw.line((303, 25, 361, 25), fill=ink, width=2)
        draw.line((322, 119, 369, 119), fill=ink, width=2)
        center = (308, 72)
    elif family == "legend_rail":
        draw.rounded_rectangle((246, 22, 371, 89), radius=9, outline=ink, width=2)
        draw.ellipse((261, 38, 273, 50), outline=ink, width=2)
        draw.rectangle((261, 62, 273, 74), fill=ink)
        draw.line((287, 44, 357, 44), fill=ink, width=2)
        draw.line((287, 68, 348, 68), fill=ink, width=2)
        center = (307, 56)
    else:
        spokes = ((250, 111), (276, 42), (304, 106), (336, 37), (365, 96))
        draw.line(spokes, fill=ink, width=2, joint="curve")
        draw.line((251, 38, 364, 113), fill=ink, width=1)
        draw.polygon(((364, 113), (346, 109), (356, 96)), fill=ink)
        center = (307, 78)
    return f"{registration.renderer_family}:{family}", masks, center


def _degrade(image: Image.Image, split: str, index: int) -> tuple[Image.Image, str]:
    if split == "train":
        family = ("train_contrast", "train_blockdrop", "train_resample", "train_scanline")[index % 4]
        if family == "train_contrast":
            return ImageEnhance.Contrast(image).enhance(0.84), family
        if family == "train_blockdrop":
            array = np.asarray(image, dtype=np.uint8).copy()
            y = 8 + ((index * 13) % 168)
            x = 17 + ((index * 23) % 346)
            array[y : y + 2, x : x + 7, :] = 246
            return Image.fromarray(array, "RGB"), family
        if family == "train_resample":
            reduced = image.resize((347, 177), Image.Resampling.BILINEAR)
            return reduced.resize(image.size, Image.Resampling.LANCZOS), family
        array = np.asarray(image, dtype=np.int16).copy()
        array[7::17, :, :] = np.clip(array[7::17, :, :] - 5, 0, 255)
        return Image.fromarray(array.astype(np.uint8), "RGB"), family
    if split == "validation":
        family = ("validation_defocus", "validation_ringing", "validation_channel", "validation_dropout")[index % 4]
        if family == "validation_defocus":
            return image.filter(ImageFilter.GaussianBlur(0.48)), family
        if family == "validation_ringing":
            reduced = image.resize((353, 173), Image.Resampling.BICUBIC)
            return reduced.resize(image.size, Image.Resampling.LANCZOS), family
        if family == "validation_channel":
            array = np.asarray(image, dtype=np.int16).copy()
            array[:, :, 0] = np.clip(array[:, :, 0] - 6, 0, 255)
            array[:, :, 1] = np.clip(array[:, :, 1] + 4, 0, 255)
            return Image.fromarray(array.astype(np.uint8), "RGB"), family
        array = np.asarray(image, dtype=np.uint8).copy()
        y = 5 + ((index * 19) % 174)
        x = 9 + ((index * 37) % 354)
        array[y : y + 2, x : x + 9, :] = 249
        return Image.fromarray(array, "RGB"), family
    family = ("sealed_speckle", "sealed_median", "sealed_poster", "sealed_microcontrast")[index % 4]
    if family == "sealed_speckle":
        array = np.asarray(image, dtype=np.uint8).copy()
        for ordinal in range(5):
            y = (index * 17 + ordinal * 31 + 13) % FRAME_HEIGHT
            x = (index * 29 + ordinal * 47 + 21) % FRAME_WIDTH
            array[y, x, :] = 238
        return Image.fromarray(array, "RGB"), family
    if family == "sealed_median":
        return image.filter(ImageFilter.MedianFilter(3)), family
    if family == "sealed_poster":
        array = np.asarray(image, dtype=np.uint8)
        return Image.fromarray(((array // 5) * 5).astype(np.uint8), "RGB"), family
    return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.28))).enhance(0.91), family


def _render_source(split: str, index: int) -> _RenderedSource:
    registration = split_registration(split)
    count = registration.text_count + registration.exclusion_count
    if not 0 <= index < count:
        raise ValueError("Stride-4 source index is out of range")
    rng = _rng(split, index)
    background = tuple(int(value) for value in rng.integers(246, 256, size=3))
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    structure_family, masks, structure_center = _draw_structures(draw, split, index)
    target_image = Image.new("L", (FRAME_WIDTH, FRAME_HEIGHT), 0)
    target_draw = ImageDraw.Draw(target_image)
    kind = "text" if index < registration.text_count else "exclusion"
    truth_bbox: tuple[float, float, float, float] | None = None
    if kind == "text":
        multiplier = {"train": 7, "validation": 9, "sealed_public": 11}[split]
        text = GENERIC_TEXT[(index * multiplier + (3, 5, 7)[("train", "validation", "sealed_public").index(split)]) % len(GENERIC_TEXT)]
        size = 14 + ((index * (5 if split == "train" else 7)) % 17)
        if len(text) > 9:
            size = min(size, 20)
        font = _font(rng, size)
        raw = draw.textbbox((0, 0), text, font=font)
        width = raw[2] - raw[0]
        height = raw[3] - raw[1]
        anchors = {
            "train": ((75, 8), (221, 116), (11, 91), (126, 6), (208, 139), (69, 136), (279, 7)),
            "validation": ((86, 12), (231, 119), (16, 83), (142, 9), (198, 137), (81, 132), (267, 12)),
            "sealed_public": ((68, 15), (241, 108), (18, 98), (151, 8), (193, 141), (94, 126), (263, 16)),
        }[split]
        anchor_offset = (0, 2, 4)[("train", "validation", "sealed_public").index(split)]
        x, y = anchors[(index * 5 + anchor_offset) % len(anchors)]
        x = min(max(4, x + int(rng.integers(-6, 7))), FRAME_WIDTH - width - 5)
        y = min(max(3, y + int(rng.integers(-4, 5))), FRAME_HEIGHT - height - 5)
        ink = tuple(int(value) for value in rng.integers(4, 47, size=3))
        draw.text((x, y), text, font=font, fill=ink)
        box = draw.textbbox((x, y), text, font=font)
        truth_bbox = tuple(float(value) for value in box)
        target_draw.rectangle(box, fill=255)
    image, degradation = _degrade(image, split, index)
    rgb = np.asarray(image, dtype=np.uint8)
    detector_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    for left, top, right, bottom in masks:
        detector_bgr[top:bottom, left:right, :] = 255
    return _RenderedSource(
        image=image,
        detector_bgr=detector_bgr,
        target=np.asarray(target_image, dtype=np.uint8).copy(),
        truth_bbox=truth_bbox,
        kind=kind,
        renderer_family=registration.renderer_family,
        degradation_family=f"{registration.degradation_family}:{degradation}",
        structure_family=structure_family,
        structure_center=structure_center,
    )


def _clamped_origin(center: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    return (
        min(max(0, center[0] - PATCH_WIDTH // 2), width - PATCH_WIDTH),
        min(max(0, center[1] - PATCH_HEIGHT // 2), height - PATCH_HEIGHT),
    )


def _tile_origins(source_index: int, kind: str, full_target: np.ndarray) -> tuple[tuple[int, int], ...]:
    height, width = full_target.shape
    grid = (
        (0, 0),
        ((width - PATCH_WIDTH) // 2, 0),
        (width - PATCH_WIDTH, 0),
        (0, height - PATCH_HEIGHT),
        ((width - PATCH_WIDTH) // 2, height - PATCH_HEIGHT),
        (width - PATCH_WIDTH, height - PATCH_HEIGHT),
    )
    origins: list[tuple[int, int]] = []
    if kind == "text":
        ys, xs = np.nonzero(full_target)
        origins.append(_clamped_origin((int(round((xs.min() + xs.max()) / 2)), int(round((ys.min() + ys.max()) / 2))), width, height))
    start = (source_index * 5 + (1 if kind == "text" else 2)) % len(grid)
    for offset in range(len(grid)):
        origin = grid[(start + offset * 5) % len(grid)]
        if origin not in origins:
            origins.append(origin)
        if len(origins) == TILES_PER_SOURCE:
            break
    if len(origins) != TILES_PER_SOURCE:
        raise RuntimeError("Stride-4 tile composition changed")
    return tuple(origins)


def render_training_tiles(source_index: int) -> tuple[TrainingTile, ...]:
    if not 0 <= source_index < TRAIN_SOURCE_COUNT:
        raise ValueError("Stride-4 training source index is out of range")
    source = _render_source("train", source_index)
    bgr = _production_resize(source.detector_bgr, cv2.INTER_LINEAR)
    full_target = _production_resize(source.target, cv2.INTER_NEAREST)
    source_positive = _shrink_target(source.target)
    target = _production_resize(source_positive, cv2.INTER_NEAREST)
    supervision = _production_resize(_supervision_mask(source.target, source_positive), cv2.INTER_NEAREST)
    tiles: list[TrainingTile] = []
    for ordinal, (left, top) in enumerate(_tile_origins(source_index, source.kind, full_target)):
        right, bottom = left + PATCH_WIDTH, top + PATCH_HEIGHT
        tiles.append(TrainingTile(
            tile_id=f"graph-text-stride4-v4-p1-{source_index:05d}-{ordinal}",
            source_index=source_index,
            kind=source.kind,
            left=left,
            top=top,
            bgr=np.ascontiguousarray(bgr[top:bottom, left:right, :]),
            target=np.ascontiguousarray(target[top:bottom, left:right]),
            supervision_mask=np.ascontiguousarray(supervision[top:bottom, left:right]),
            renderer_family=source.renderer_family,
            degradation_family=source.degradation_family,
        ))
    return tuple(tiles)


def build_training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [tile for index in range(TRAIN_SOURCE_COUNT) for tile in render_training_tiles(index)]
    return (
        np.stack([sample.bgr for sample in samples]).astype(np.uint8),
        np.stack([sample.target for sample in samples])[:, None, :, :].astype(np.uint8),
        np.stack([sample.supervision_mask for sample in samples])[:, None, :, :].astype(np.uint8),
    )


def _evaluation_frame(split: str, index: int) -> EvaluationFrame:
    source = _render_source(split, index)
    stream = BytesIO()
    source.image.save(stream, format="PNG", optimize=False, compress_level=9)
    png = stream.getvalue()
    detector = source.detector_bgr.tobytes(order="C")
    return EvaluationFrame(
        case_id=f"graph-text-stride4-v4-{split}-{source.kind}-{index:04d}",
        kind=source.kind,
        source_png=png,
        source_sha256=sha256(png).hexdigest(),
        detector_bgr=detector,
        detector_bgr_sha256=sha256(detector).hexdigest(),
        truth_bbox=source.truth_bbox,
        renderer_family=source.renderer_family,
        degradation_family=source.degradation_family,
        structure_family=source.structure_family,
    )


def build_validation_split() -> tuple[EvaluationFrame, ...]:
    count = VALIDATION_TEXT_COUNT + VALIDATION_EXCLUSION_COUNT
    return tuple(_evaluation_frame("validation", index) for index in range(count))


def split_fingerprint(samples: tuple[TrainingTile | EvaluationFrame, ...]) -> str:
    records: list[dict[str, object]] = []
    for sample in samples:
        if isinstance(sample, TrainingTile):
            records.append({
                "tile_id": sample.tile_id,
                "source_index": sample.source_index,
                "kind": sample.kind,
                "left": sample.left,
                "top": sample.top,
                "bgr_sha256": sha256(sample.bgr.tobytes(order="C")).hexdigest(),
                "target_sha256": sha256(sample.target.tobytes(order="C")).hexdigest(),
                "supervision_mask_sha256": sha256(sample.supervision_mask.tobytes(order="C")).hexdigest(),
                "renderer_family": sample.renderer_family,
                "degradation_family": sample.degradation_family,
            })
        else:
            records.append({
                "case_id": sample.case_id,
                "kind": sample.kind,
                "source_sha256": sample.source_sha256,
                "detector_bgr_sha256": sample.detector_bgr_sha256,
                "truth_bbox": list(sample.truth_bbox) if sample.truth_bbox is not None else None,
                "renderer_family": sample.renderer_family,
                "degradation_family": sample.degradation_family,
                "structure_family": sample.structure_family,
            })
    return sha256((json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def training_split_fingerprint() -> str:
    return split_fingerprint(tuple(tile for index in range(TRAIN_SOURCE_COUNT) for tile in render_training_tiles(index)))


__all__ = [
    "EvaluationFrame",
    "FONT_PATHS",
    "GENERIC_TEXT",
    "TrainingTile",
    "build_training_arrays",
    "build_validation_split",
    "render_training_tiles",
    "split_fingerprint",
    "training_split_fingerprint",
]

