# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Disjoint procedural data for the balanced-recall graph text detector."""

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
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PATCH_HEIGHT,
    PATCH_WIDTH,
    TRAIN_SAMPLE_COUNT,
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
    "95%",
    "Acquisition",
    "Intervention",
    "Maintenance",
    "Probe",
    "Taylor",
    "Case B",
    "Phase C",
)


@dataclass(frozen=True)
class TrainingPatch:
    sample_id: str
    kind: str
    bgr: np.ndarray
    target: np.ndarray
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


def _rng(split: str, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"graph-text-balanced-recall-v2:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(rng: np.random.Generator, size: int) -> ImageFont.FreeTypeFont:
    path = REPO_ROOT / FONT_PATHS[int(rng.integers(0, len(FONT_PATHS)))]
    return ImageFont.truetype(str(path), size=size)


def _mask(
    masks: list[tuple[int, int, int, int]],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    masks.append((left, top, right, bottom))


def _draw_structures(
    draw: ImageDraw.ImageDraw,
    split: str,
    index: int,
) -> tuple[str, list[tuple[int, int, int, int]]]:
    registration = split_registration(split)
    masks: list[tuple[int, int, int, int]] = []
    families = (
        "rail_legend_with_open_marker",
        "nested_bracket_with_ticks",
        "fan_intersections_with_arrow",
        "dual_series_compact_markers",
    )
    offset = 0 if split == "train" else 1 if split == "validation" else 3
    family = families[(index * 5 + offset) % len(families)]
    axis_x = (49, 63, 42)[("train", "validation", "sealed_public").index(split)] + (index % 4)
    axis_y = (166, 158, 171)[("train", "validation", "sealed_public").index(split)] - (index % 3)
    ink = (25, 28, 31)
    draw.line((axis_x, 25, axis_x, axis_y), fill=ink, width=2)
    draw.line((axis_x, axis_y, 365, axis_y), fill=ink, width=2)
    _mask(masks, axis_x - 4, 21, axis_x + 5, axis_y + 5)
    _mask(masks, axis_x - 4, axis_y - 4, 368, axis_y + 5)
    for tick in range(7):
        x = axis_x + 25 + tick * 42
        y = axis_y - 18 - ((tick * 17 + index * 11 + offset * 7) % 103)
        draw.line((x, axis_y - 5, x, axis_y + 5), fill=ink, width=1)
        draw.line((axis_x - 5, y, axis_x + 5, y), fill=ink, width=1)
        _mask(masks, x - 2, axis_y - 7, x + 3, axis_y + 8)
        _mask(masks, axis_x - 7, y - 2, axis_x + 8, y + 3)
    points = (
        (axis_x + 37, axis_y - 31),
        (axis_x + 79, axis_y - 74),
        (axis_x + 121, axis_y - 48),
        (axis_x + 165, axis_y - 103),
        (axis_x + 213, axis_y - 63),
        (axis_x + 269, axis_y - 114),
    )
    draw.line(points, fill=(20, 22, 24), width=2, joint="curve")
    for marker_index, (x, y) in enumerate(points):
        radius = 3 + ((marker_index + index) % 3)
        if (marker_index + index) % 2:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, fill=(252, 252, 250), width=2)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=ink)
    divider_x = axis_x + 146 + (index % 13)
    draw.line((divider_x, 24, divider_x, axis_y), fill=(38, 38, 38), width=2)
    _mask(masks, divider_x - 3, 20, divider_x + 4, axis_y + 3)
    if family == "rail_legend_with_open_marker":
        draw.rounded_rectangle((259, 27, 363, 77), radius=5, outline=ink, width=2)
        draw.ellipse((271, 45, 282, 56), outline=ink, fill=(252, 252, 250), width=2)
        draw.line((292, 51, 348, 51), fill=ink, width=2)
    elif family == "nested_bracket_with_ticks":
        draw.line((266, 30, 266, 102), fill=ink, width=2)
        draw.line((266, 30, 302, 30), fill=ink, width=2)
        draw.line((266, 102, 302, 102), fill=ink, width=2)
        draw.line((279, 43, 279, 88), fill=ink, width=1)
    elif family == "fan_intersections_with_arrow":
        draw.line((272, 28, 360, 111), fill=ink, width=2)
        draw.line((274, 111, 358, 31), fill=ink, width=2)
        draw.line((241, 126, 331, 91), fill=ink, width=2)
        draw.polygon(((331, 91), (314, 91), (325, 106)), fill=ink)
    else:
        second = tuple((x, min(axis_y - 8, y + 18)) for x, y in points)
        draw.line(second, fill=(44, 44, 44), width=2, joint="curve")
        for x, y in second:
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline=ink, width=2)
    return f"{registration.renderer_family}:{family}", masks


def _degrade(image: Image.Image, split: str, index: int) -> tuple[Image.Image, str]:
    if split == "train":
        family = ("train_clean", "train_low_contrast", "train_ringing", "train_streak")[index % 4]
        if family == "train_clean":
            return image, family
        if family == "train_low_contrast":
            return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.25))).enhance(0.82), family
        if family == "train_ringing":
            reduced = image.resize((347, 173), Image.Resampling.BILINEAR)
            return reduced.resize(image.size, Image.Resampling.LANCZOS), family
        array = np.asarray(image, dtype=np.int16).copy()
        array[::17, :, :] = np.clip(array[::17, :, :] - 6, 0, 255)
        return Image.fromarray(array.astype(np.uint8), "RGB"), family
    if split == "validation":
        family = ("validation_quantized", "validation_haze", "validation_raster", "validation_soften")[index % 4]
        if family == "validation_quantized":
            array = np.asarray(image, dtype=np.uint8)
            return Image.fromarray(((array // 8) * 8).astype(np.uint8), "RGB"), family
        if family == "validation_haze":
            return ImageEnhance.Contrast(image).enhance(0.74), family
        if family == "validation_raster":
            reduced = image.resize((359, 179), Image.Resampling.BILINEAR)
            return reduced.resize(image.size, Image.Resampling.BICUBIC), family
        return image.filter(ImageFilter.GaussianBlur(0.52)), family
    family = ("sealed_sharpen", "sealed_dropout", "sealed_chroma", "sealed_median")[index % 4]
    if family == "sealed_sharpen":
        return image.filter(ImageFilter.UnsharpMask(radius=1, percent=90, threshold=3)), family
    if family == "sealed_dropout":
        array = np.asarray(image, dtype=np.uint8).copy()
        y = 12 + ((index * 13) % 151)
        x = 74 + ((index * 19) % 269)
        array[y : y + 2, x : x + 7, :] = 246
        return Image.fromarray(array, "RGB"), family
    if family == "sealed_chroma":
        array = np.asarray(image, dtype=np.int16).copy()
        array[:, :, 1] = np.clip(array[:, :, 1] + 4, 0, 255)
        array[:, :, 2] = np.clip(array[:, :, 2] - 3, 0, 255)
        return Image.fromarray(array.astype(np.uint8), "RGB"), family
    return image.filter(ImageFilter.MedianFilter(3)), family


def _render_source(split: str, index: int) -> _RenderedSource:
    registration = split_registration(split)
    count = registration.text_count + registration.exclusion_count
    if not 0 <= index < count:
        raise ValueError("Balanced-recall source index is out of range")
    rng = _rng(split, index)
    image = Image.new(
        "RGB",
        (FRAME_WIDTH, FRAME_HEIGHT),
        tuple(int(value) for value in rng.integers(247, 256, size=3)),
    )
    draw = ImageDraw.Draw(image)
    structure_family, masks = _draw_structures(draw, split, index)
    target = Image.new("L", (FRAME_WIDTH, FRAME_HEIGHT), 0)
    target_draw = ImageDraw.Draw(target)
    kind = "text" if index < registration.text_count else "exclusion"
    truth_bbox: tuple[float, float, float, float] | None = None
    if kind == "text":
        multiplier = 7 if split == "train" else 5 if split == "validation" else 11
        text = GENERIC_TEXT[(index * multiplier + 1) % len(GENERIC_TEXT)]
        size = 16 + ((index * (5 if split == "sealed_public" else 3)) % 15)
        if len(text) > 7:
            size = min(size, 23)
        font = _font(rng, size)
        raw = draw.textbbox((0, 0), text, font=font)
        width = raw[2] - raw[0]
        height = raw[3] - raw[1]
        anchors = {
            "train": ((76, 8), (245, 111), (12, 96), (121, 11), (230, 132), (78, 136)),
            "validation": ((103, 8), (269, 106), (15, 82), (136, 12), (242, 128), (91, 130)),
            "sealed_public": ((71, 15), (257, 99), (18, 109), (151, 8), (223, 137), (96, 125)),
        }[split]
        x, y = anchors[(index * 5 + (2 if split == "sealed_public" else 0)) % len(anchors)]
        x = min(max(5, x + int(rng.integers(-4, 5))), FRAME_WIDTH - width - 6)
        y = min(max(4, y + int(rng.integers(-3, 4))), FRAME_HEIGHT - height - 6)
        draw.text((x, y), text, font=font, fill=tuple(int(value) for value in rng.integers(6, 52, size=3)))
        box = draw.textbbox((x, y), text, font=font)
        truth_bbox = tuple(float(value) for value in box)
        target_draw.rectangle(box, fill=255)
    image, degradation = _degrade(image, split, index)
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    for left, top, right, bottom in masks:
        bgr[top:bottom, left:right, :] = 255
    return _RenderedSource(
        image=image,
        detector_bgr=bgr,
        target=np.asarray(target, dtype=np.uint8).copy(),
        truth_bbox=truth_bbox,
        kind=kind,
        renderer_family=registration.renderer_family,
        degradation_family=f"{registration.degradation_family}:{degradation}",
        structure_family=structure_family,
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
    distance = (width * height) * (1.0 - (DB_SHRINK_RATIO * DB_SHRINK_RATIO)) / (2.0 * (width + height))
    shrunk_left = int(math.ceil(left + distance))
    shrunk_top = int(math.ceil(top + distance))
    shrunk_right = int(math.floor(right - distance))
    shrunk_bottom = int(math.floor(bottom - distance))
    if shrunk_right <= shrunk_left or shrunk_bottom <= shrunk_top:
        raise RuntimeError("Balanced-recall DB shrink target collapsed")
    result = np.zeros_like(source_target)
    result[shrunk_top:shrunk_bottom, shrunk_left:shrunk_right] = 255
    return result


def render_training_patch(index: int) -> TrainingPatch:
    if not 0 <= index < TRAIN_SAMPLE_COUNT:
        raise ValueError("Balanced-recall training index is out of range")
    source = _render_source("train", index)
    bgr = _production_resize(source.detector_bgr, cv2.INTER_LINEAR)
    full_target = _production_resize(source.target, cv2.INTER_NEAREST)
    target = _production_resize(_shrink_target(source.target), cv2.INTER_NEAREST)
    maximum_left = bgr.shape[1] - PATCH_WIDTH
    maximum_top = bgr.shape[0] - PATCH_HEIGHT
    if source.kind == "text":
        ys, xs = np.nonzero(full_target)
        center_x = int(round((float(xs.min()) + float(xs.max())) / 2.0))
        center_y = int(round((float(ys.min()) + float(ys.max())) / 2.0))
        left = min(max(0, center_x - PATCH_WIDTH // 2), maximum_left)
        top = min(max(0, center_y - PATCH_HEIGHT // 2), maximum_top)
        if xs.min() < left or xs.max() >= left + PATCH_WIDTH:
            left = min(max(0, int(xs.max()) - PATCH_WIDTH + 8), maximum_left)
        if ys.min() < top or ys.max() >= top + PATCH_HEIGHT:
            top = min(max(0, int(ys.max()) - PATCH_HEIGHT + 8), maximum_top)
    else:
        family = source.structure_family.rsplit(":", 1)[-1]
        centers = {
            "rail_legend_with_open_marker": (796, 135),
            "nested_bracket_with_ticks": (744, 176),
            "fan_intersections_with_arrow": (770, 276),
            "dual_series_compact_markers": (506, 260),
        }
        center_x, center_y = centers[family]
        left = min(max(0, center_x - PATCH_WIDTH // 2), maximum_left)
        top = min(max(0, center_y - PATCH_HEIGHT // 2), maximum_top)
    right, bottom = left + PATCH_WIDTH, top + PATCH_HEIGHT
    return TrainingPatch(
        sample_id=f"graph-text-balanced-recall-v2-train-{index:05d}",
        kind=source.kind,
        bgr=np.ascontiguousarray(bgr[top:bottom, left:right, :]),
        target=np.ascontiguousarray(target[top:bottom, left:right]),
        renderer_family=source.renderer_family,
        degradation_family=source.degradation_family,
    )


def build_training_arrays() -> tuple[np.ndarray, np.ndarray]:
    samples = [render_training_patch(index) for index in range(TRAIN_SAMPLE_COUNT)]
    return (
        np.stack([sample.bgr for sample in samples]).astype(np.uint8),
        np.stack([sample.target for sample in samples])[:, None, :, :].astype(np.uint8),
    )


def _evaluation_frame(split: str, index: int) -> EvaluationFrame:
    source = _render_source(split, index)
    stream = BytesIO()
    source.image.save(stream, format="PNG", optimize=False, compress_level=9)
    png = stream.getvalue()
    detector = source.detector_bgr.tobytes(order="C")
    return EvaluationFrame(
        case_id=f"graph-text-balanced-recall-v2-{split}-{source.kind}-{index:04d}",
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


def split_fingerprint(samples: tuple[TrainingPatch | EvaluationFrame, ...]) -> str:
    records: list[dict[str, object]] = []
    for sample in samples:
        if isinstance(sample, TrainingPatch):
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "kind": sample.kind,
                    "bgr_sha256": sha256(sample.bgr.tobytes(order="C")).hexdigest(),
                    "target_sha256": sha256(sample.target.tobytes(order="C")).hexdigest(),
                    "renderer_family": sample.renderer_family,
                    "degradation_family": sample.degradation_family,
                }
            )
        else:
            records.append(
                {
                    "case_id": sample.case_id,
                    "kind": sample.kind,
                    "source_sha256": sample.source_sha256,
                    "detector_bgr_sha256": sample.detector_bgr_sha256,
                    "truth_bbox": list(sample.truth_bbox) if sample.truth_bbox is not None else None,
                    "renderer_family": sample.renderer_family,
                    "degradation_family": sample.degradation_family,
                    "structure_family": sample.structure_family,
                }
            )
    return sha256((json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def training_split_fingerprint() -> str:
    return split_fingerprint(tuple(render_training_patch(index) for index in range(TRAIN_SAMPLE_COUNT)))


__all__ = [
    "EvaluationFrame",
    "FONT_PATHS",
    "GENERIC_TEXT",
    "TrainingPatch",
    "build_training_arrays",
    "build_validation_split",
    "render_training_patch",
    "split_fingerprint",
    "training_split_fingerprint",
]
