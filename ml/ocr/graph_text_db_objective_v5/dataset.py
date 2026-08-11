# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Disjoint procedural data and DB supervision for graph text detector V5."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .protocol import (
    DB_SHRINK_RATIO,
    DB_THRESHOLD_MAXIMUM,
    DB_THRESHOLD_MINIMUM,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IGNORE_BAND_EXPANSION_PIXELS,
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
    "5",
    "16",
    "-12",
    "3.50",
    "60%",
    "Initial",
    "Treatment",
    "Retention",
    "Check",
    "Morgan",
    "Case J",
    "Phase D",
    "Session 21",
    "Participant K",
    "Follow through",
    "Condition C",
)


@dataclass(frozen=True)
class TrainingTile:
    tile_id: str
    source_index: int
    kind: str
    left: int
    top: int
    bgr: np.ndarray
    shrink_target: np.ndarray
    shrink_mask: np.ndarray
    threshold_target: np.ndarray
    threshold_mask: np.ndarray
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
    material = f"graph-text-db-objective-v5:{registration.seed_offset}:{split}:{index}".encode()
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
    families = ("nested_capsule", "cantilever_fan", "crossbar_kites", "rail_intersections")
    split_index = ("train", "validation", "sealed_public").index(split)
    family = families[(index * 7 + (0, 2, 3)[split_index]) % len(families)]
    axis_x = (44, 58, 51)[split_index] + ((index * 11) % 7)
    axis_y = (169, 163, 176)[split_index] - ((index * 5) % 6)
    ink = (25, 28, 32)

    draw.line((axis_x, 13, axis_x, axis_y), fill=ink, width=2)
    draw.line((axis_x, axis_y, 374, axis_y), fill=ink, width=2)
    _add_mask(masks, axis_x - 5, 9, axis_x + 6, axis_y + 7)
    _add_mask(masks, axis_x - 5, axis_y - 5, 378, axis_y + 7)
    for tick in range(8):
        x = axis_x + 23 + tick * 39
        y = axis_y - 19 - ((tick * 29 + index * 13 + split_index * 17) % 119)
        draw.line((x, axis_y - 5, x, axis_y + 5), fill=ink, width=1)
        draw.line((axis_x - 5, y, axis_x + 5, y), fill=ink, width=1)
        _add_mask(masks, x - 2, axis_y - 7, x + 3, axis_y + 8)
        _add_mask(masks, axis_x - 7, y - 2, axis_x + 8, y + 3)

    points = tuple(
        (axis_x + 24 + ordinal * 48, axis_y - 28 - ((ordinal * 37 + index * 9) % 104))
        for ordinal in range(7)
    )
    draw.line(points, fill=ink, width=2, joint="curve")
    for ordinal, (x, y) in enumerate(points):
        radius = 3 + ((ordinal + index + split_index) % 3)
        if (ordinal + index + split_index) % 2:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)
        else:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=ink)

    divider_x = axis_x + 122 + ((index * 17) % 49)
    draw.line((divider_x, 11, divider_x, axis_y), fill=(39, 41, 44), width=2)
    _add_mask(masks, divider_x - 4, 7, divider_x + 5, axis_y + 5)

    if family == "nested_capsule":
        draw.rounded_rectangle((247, 20, 371, 92), radius=12, outline=ink, width=2)
        draw.rounded_rectangle((262, 34, 354, 78), radius=8, outline=ink, width=2)
        draw.ellipse((277, 47, 291, 61), outline=ink, width=2)
        center = (309, 57)
    elif family == "cantilever_fan":
        draw.line((244, 108, 278, 32, 313, 105, 347, 27, 374, 96), fill=ink, width=2)
        draw.line((248, 30, 369, 112), fill=ink, width=1)
        draw.polygon(((369, 112), (349, 107), (360, 94)), fill=ink)
        center = (309, 73)
    elif family == "crossbar_kites":
        for ordinal in range(4):
            x = 250 + ordinal * 31
            y = 38 + (ordinal % 2) * 35
            draw.polygon(((x, y - 8), (x + 9, y), (x, y + 8), (x - 9, y)), outline=ink)
            draw.line((x - 13, y, x + 13, y), fill=ink, width=2)
        draw.line((241, 22, 370, 112), fill=ink, width=1)
        center = (309, 68)
    else:
        for ordinal in range(5):
            x = 252 + ordinal * 27
            draw.line((x, 25, x + 22, 104), fill=ink, width=2)
            draw.line((x - 7, 69, x + 29, 69), fill=ink, width=1)
        center = (310, 68)
    return f"{registration.renderer_family}:{family}", masks, center


def _degrade(image: Image.Image, split: str, index: int) -> tuple[Image.Image, str]:
    if split == "train":
        family = ("train_halftone", "train_bloom", "train_resample", "train_pulse")[index % 4]
        if family == "train_halftone":
            array = np.asarray(image, dtype=np.int16).copy()
            array[4::9, 2::7, :] = np.clip(array[4::9, 2::7, :] - 7, 0, 255)
            return Image.fromarray(array.astype(np.uint8), "RGB"), family
        if family == "train_bloom":
            return ImageEnhance.Brightness(image.filter(ImageFilter.GaussianBlur(0.32))).enhance(1.015), family
        if family == "train_resample":
            reduced = image.resize((341, 173), Image.Resampling.BILINEAR)
            return reduced.resize(image.size, Image.Resampling.LANCZOS), family
        array = np.asarray(image, dtype=np.int16).copy()
        array[:, 6::19, :] = np.clip(array[:, 6::19, :] - 4, 0, 255)
        return Image.fromarray(array.astype(np.uint8), "RGB"), family
    if split == "validation":
        family = ("validation_ringing", "validation_defocus", "validation_channel", "validation_notch")[index % 4]
        if family == "validation_ringing":
            reduced = image.resize((349, 175), Image.Resampling.BICUBIC)
            return reduced.resize(image.size, Image.Resampling.LANCZOS), family
        if family == "validation_defocus":
            return image.filter(ImageFilter.GaussianBlur(0.42)), family
        if family == "validation_channel":
            array = np.asarray(image, dtype=np.int16).copy()
            array[:, :, 0] = np.clip(array[:, :, 0] + 5, 0, 255)
            array[:, :, 2] = np.clip(array[:, :, 2] - 7, 0, 255)
            return Image.fromarray(array.astype(np.uint8), "RGB"), family
        array = np.asarray(image, dtype=np.uint8).copy()
        y = 8 + ((index * 23) % 168)
        x = 13 + ((index * 31) % 350)
        array[y : y + 2, x : x + 8, :] = 248
        return Image.fromarray(array, "RGB"), family
    family = ("sealed_quantize", "sealed_speckle", "sealed_pulse", "sealed_soften")[index % 4]
    if family == "sealed_quantize":
        array = np.asarray(image, dtype=np.uint8)
        return Image.fromarray(((array // 6) * 6).astype(np.uint8), "RGB"), family
    if family == "sealed_speckle":
        array = np.asarray(image, dtype=np.uint8).copy()
        for ordinal in range(7):
            y = (index * 19 + ordinal * 23 + 11) % FRAME_HEIGHT
            x = (index * 31 + ordinal * 43 + 17) % FRAME_WIDTH
            array[y, x, :] = 236
        return Image.fromarray(array, "RGB"), family
    if family == "sealed_pulse":
        array = np.asarray(image, dtype=np.int16).copy()
        array[5::16, :, :] = np.clip(array[5::16, :, :] - 5, 0, 255)
        return Image.fromarray(array.astype(np.uint8), "RGB"), family
    return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.24))).enhance(0.93), family


def _render_source(split: str, index: int) -> _RenderedSource:
    registration = split_registration(split)
    count = registration.text_count + registration.exclusion_count
    if not 0 <= index < count:
        raise ValueError("DB-objective source index is out of range")
    rng = _rng(split, index)
    background = tuple(int(value) for value in rng.integers(247, 256, size=3))
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    structure_family, masks, structure_center = _draw_structures(draw, split, index)
    target_image = Image.new("L", (FRAME_WIDTH, FRAME_HEIGHT), 0)
    target_draw = ImageDraw.Draw(target_image)
    kind = "text" if index < registration.text_count else "exclusion"
    truth_bbox: tuple[float, float, float, float] | None = None
    if kind == "text":
        split_index = ("train", "validation", "sealed_public").index(split)
        multiplier = (11, 13, 17)[split_index]
        text = GENERIC_TEXT[(index * multiplier + (1, 5, 9)[split_index]) % len(GENERIC_TEXT)]
        size = 14 + ((index * (7, 9, 11)[split_index]) % 17)
        if len(text) > 9:
            size = min(size, 20)
        font = _font(rng, size)
        raw = draw.textbbox((0, 0), text, font=font)
        width, height = raw[2] - raw[0], raw[3] - raw[1]
        anchors = {
            "train": ((72, 7), (215, 116), (8, 87), (137, 4), (201, 139), (77, 132), (272, 8)),
            "validation": ((91, 10), (226, 117), (15, 80), (151, 7), (194, 138), (84, 128), (263, 11)),
            "sealed_public": ((65, 13), (237, 106), (20, 94), (159, 6), (188, 142), (99, 124), (258, 14)),
        }[split]
        x, y = anchors[(index * 7 + split_index) % len(anchors)]
        x = min(max(4, x + int(rng.integers(-6, 7))), FRAME_WIDTH - width - 5)
        y = min(max(3, y + int(rng.integers(-4, 5))), FRAME_HEIGHT - height - 5)
        ink = tuple(int(value) for value in rng.integers(5, 48, size=3))
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


def _production_resize(values: np.ndarray, interpolation: int) -> np.ndarray:
    ratio = 960.0 / max(FRAME_WIDTH, FRAME_HEIGHT)
    resized_width = int(FRAME_WIDTH * ratio)
    resized_height = int(FRAME_HEIGHT * ratio)
    target_width = ((resized_width + 127) // 128) * 128
    target_height = ((resized_height + 127) // 128) * 128
    return cv2.resize(values, (target_width, target_height), interpolation=interpolation)


def _shrink_target(source_target: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(source_target)
    if len(xs) == 0:
        return np.zeros_like(source_target)
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    width, height = right - left, bottom - top
    distance = (width * height) * (1.0 - DB_SHRINK_RATIO**2) / (2.0 * (width + height))
    result = np.zeros_like(source_target)
    shrunk_left, shrunk_top = int(math.ceil(left + distance)), int(math.ceil(top + distance))
    shrunk_right, shrunk_bottom = int(math.floor(right - distance)), int(math.floor(bottom - distance))
    if shrunk_right <= shrunk_left or shrunk_bottom <= shrunk_top:
        raise RuntimeError("DB-objective shrink target collapsed")
    result[shrunk_top:shrunk_bottom, shrunk_left:shrunk_right] = 255
    return result


def _db_supervision(source_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shrink = _shrink_target(source_target)
    shrink_mask = np.full_like(source_target, 255)
    threshold = np.zeros_like(source_target, dtype=np.float32)
    threshold_mask = np.zeros_like(source_target, dtype=np.uint8)
    if not np.any(source_target):
        return shrink, shrink_mask, threshold, threshold_mask
    kernel_size = 1 + 2 * IGNORE_BAND_EXPANSION_PIXELS
    ignored = cv2.dilate(source_target, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    shrink_mask[ignored > 0] = 0
    shrink_mask[shrink > 0] = 255
    ys, xs = np.nonzero(source_target)
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    width, height = right - left, bottom - top
    distance = max(1.0, (width * height) * (1.0 - DB_SHRINK_RATIO**2) / (2.0 * (width + height)))
    margin = int(math.ceil(distance))
    expanded = np.zeros_like(source_target, dtype=np.uint8)
    expanded[max(0, top - margin) : min(FRAME_HEIGHT, bottom + margin), max(0, left - margin) : min(FRAME_WIDTH, right + margin)] = 255
    threshold_mask[expanded > 0] = 255
    foreground = (source_target > 0).astype(np.uint8)
    inside = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
    boundary_distance = np.maximum(0.0, np.where(foreground > 0, inside, outside) - 1.0)
    response = np.clip(1.0 - (boundary_distance / distance), 0.0, 1.0)
    threshold = DB_THRESHOLD_MINIMUM + (DB_THRESHOLD_MAXIMUM - DB_THRESHOLD_MINIMUM) * response
    threshold[threshold_mask == 0] = 0.0
    return shrink, shrink_mask, threshold.astype(np.float32), threshold_mask


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
    start = (source_index * 7 + (1 if kind == "text" else 3)) % len(grid)
    for offset in range(len(grid)):
        origin = grid[(start + offset * 5) % len(grid)]
        if origin not in origins:
            origins.append(origin)
        if len(origins) == TILES_PER_SOURCE:
            break
    if len(origins) != TILES_PER_SOURCE:
        raise RuntimeError("DB-objective tile composition changed")
    return tuple(origins)


def render_training_tiles(source_index: int) -> tuple[TrainingTile, ...]:
    if not 0 <= source_index < TRAIN_SOURCE_COUNT:
        raise ValueError("DB-objective training source index is out of range")
    source = _render_source("train", source_index)
    bgr = _production_resize(source.detector_bgr, cv2.INTER_LINEAR)
    full_target = _production_resize(source.target, cv2.INTER_NEAREST)
    shrink, shrink_mask, threshold, threshold_mask = _db_supervision(source.target)
    shrink = _production_resize(shrink, cv2.INTER_NEAREST)
    shrink_mask = _production_resize(shrink_mask, cv2.INTER_NEAREST)
    threshold = _production_resize(threshold, cv2.INTER_LINEAR)
    threshold_mask = _production_resize(threshold_mask, cv2.INTER_NEAREST)
    tiles: list[TrainingTile] = []
    for ordinal, (left, top) in enumerate(_tile_origins(source_index, source.kind, full_target)):
        right, bottom = left + PATCH_WIDTH, top + PATCH_HEIGHT
        tiles.append(TrainingTile(
            tile_id=f"graph-text-db-objective-v5-p1-{source_index:05d}-{ordinal}",
            source_index=source_index,
            kind=source.kind,
            left=left,
            top=top,
            bgr=np.ascontiguousarray(bgr[top:bottom, left:right, :]),
            shrink_target=np.ascontiguousarray(shrink[top:bottom, left:right]),
            shrink_mask=np.ascontiguousarray(shrink_mask[top:bottom, left:right]),
            threshold_target=np.ascontiguousarray(
                np.rint(np.clip(threshold[top:bottom, left:right], 0.0, 1.0) * 255.0).astype(np.uint8)
            ),
            threshold_mask=np.ascontiguousarray(threshold_mask[top:bottom, left:right]),
            renderer_family=source.renderer_family,
            degradation_family=source.degradation_family,
        ))
    return tuple(tiles)


def build_training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = [tile for index in range(TRAIN_SOURCE_COUNT) for tile in render_training_tiles(index)]
    return (
        np.stack([sample.bgr for sample in samples]).astype(np.uint8),
        np.stack([sample.shrink_target for sample in samples])[:, None].astype(np.uint8),
        np.stack([sample.shrink_mask for sample in samples])[:, None].astype(np.uint8),
        np.stack([sample.threshold_target for sample in samples])[:, None].astype(np.uint8),
        np.stack([sample.threshold_mask for sample in samples])[:, None].astype(np.uint8),
    )


def _evaluation_frame(split: str, index: int) -> EvaluationFrame:
    source = _render_source(split, index)
    stream = BytesIO()
    source.image.save(stream, format="PNG", optimize=False, compress_level=9)
    png = stream.getvalue()
    detector = source.detector_bgr.tobytes(order="C")
    return EvaluationFrame(
        case_id=f"graph-text-db-objective-v5-{split}-{source.kind}-{index:04d}",
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
    return tuple(_evaluation_frame("validation", index) for index in range(VALIDATION_TEXT_COUNT + VALIDATION_EXCLUSION_COUNT))


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
                "shrink_target_sha256": sha256(sample.shrink_target.tobytes(order="C")).hexdigest(),
                "shrink_mask_sha256": sha256(sample.shrink_mask.tobytes(order="C")).hexdigest(),
                "threshold_target_sha256": sha256(sample.threshold_target.tobytes(order="C")).hexdigest(),
                "threshold_mask_sha256": sha256(sample.threshold_mask.tobytes(order="C")).hexdigest(),
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
    "_db_supervision",
    "_evaluation_frame",
    "build_training_arrays",
    "build_validation_split",
    "render_training_tiles",
    "split_fingerprint",
    "training_split_fingerprint",
]
