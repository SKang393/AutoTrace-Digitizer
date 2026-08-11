# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Disjoint procedural data for ignore-band graph text detection."""

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
    IGNORE_BAND_EXPANSION_PIXELS,
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
    "18",
    "-6",
    "3.50",
    "82%",
    "Baseline",
    "Treatment",
    "Follow-up",
    "Probe",
    "Morgan",
    "Case D",
    "Phase E",
    "Week 12",
    "Participant B",
)


@dataclass(frozen=True)
class TrainingPatch:
    sample_id: str
    kind: str
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
    material = f"graph-text-ignore-band-v3:{registration.seed_offset}:{split}:{index}".encode()
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
    masks.append((
        max(0, left),
        max(0, top),
        min(FRAME_WIDTH, right),
        min(FRAME_HEIGHT, bottom),
    ))


def _draw_structures(
    draw: ImageDraw.ImageDraw,
    split: str,
    index: int,
) -> tuple[str, list[tuple[int, int, int, int]], tuple[int, int]]:
    registration = split_registration(split)
    masks: list[tuple[int, int, int, int]] = []
    families = (
        "boxed_square_key",
        "double_hook_bracket",
        "arrow_crosshatch",
        "open_diamond_polyline",
    )
    split_index = ("train", "validation", "sealed_public").index(split)
    offset = (0, 2, 3)[split_index]
    family = families[(index * 7 + offset) % len(families)]
    axis_x = (43, 57, 51)[split_index] + ((index * 3) % 5)
    axis_y = (168, 157, 173)[split_index] - (index % 4)
    ink = (22, 25, 29)

    draw.line((axis_x, 20, axis_x, axis_y), fill=ink, width=2)
    draw.line((axis_x, axis_y, 370, axis_y), fill=ink, width=2)
    _add_mask(masks, axis_x - 5, 16, axis_x + 6, axis_y + 6)
    _add_mask(masks, axis_x - 5, axis_y - 5, 374, axis_y + 6)
    for tick in range(8):
        x = axis_x + 20 + (tick * 39)
        y = axis_y - 16 - ((tick * 23 + index * 13 + offset * 5) % 112)
        draw.line((x, axis_y - 5, x, axis_y + 5), fill=ink, width=1)
        draw.line((axis_x - 5, y, axis_x + 5, y), fill=ink, width=1)
        _add_mask(masks, x - 2, axis_y - 7, x + 3, axis_y + 8)
        _add_mask(masks, axis_x - 7, y - 2, axis_x + 8, y + 3)

    points = (
        (axis_x + 29, axis_y - 29),
        (axis_x + 67, axis_y - 65),
        (axis_x + 109, axis_y - 44),
        (axis_x + 151, axis_y - 101),
        (axis_x + 205, axis_y - 61),
        (axis_x + 273, axis_y - 116),
    )
    draw.line(points, fill=ink, width=2, joint="curve")
    for marker_index, (x, y) in enumerate(points):
        radius = 3 + ((marker_index + index + split_index) % 3)
        if (marker_index + index) % 2:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, fill=(251, 252, 250), width=2)
        else:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=ink)

    divider_x = axis_x + 132 + ((index * 5) % 25)
    draw.line((divider_x, 18, divider_x, axis_y), fill=(34, 35, 37), width=2)
    _add_mask(masks, divider_x - 4, 14, divider_x + 5, axis_y + 4)

    if family == "boxed_square_key":
        draw.rounded_rectangle((250, 24, 369, 76), radius=7, outline=ink, width=2)
        draw.rectangle((264, 43, 275, 54), outline=ink, width=2)
        draw.line((287, 49, 353, 49), fill=ink, width=2)
        center = (305, 50)
    elif family == "double_hook_bracket":
        draw.arc((252, 24, 304, 92), start=80, end=280, fill=ink, width=2)
        draw.arc((274, 41, 333, 113), start=80, end=280, fill=ink, width=2)
        draw.line((294, 31, 337, 31), fill=ink, width=2)
        draw.line((316, 105, 356, 105), fill=ink, width=2)
        center = (304, 69)
    elif family == "arrow_crosshatch":
        draw.line((247, 31, 357, 116), fill=ink, width=2)
        draw.line((249, 116, 354, 34), fill=ink, width=2)
        draw.line((267, 24, 341, 127), fill=ink, width=1)
        draw.polygon(((357, 116), (339, 111), (349, 98)), fill=ink)
        center = (304, 76)
    else:
        polyline = ((246, 105), (273, 69), (303, 97), (332, 52), (363, 82))
        draw.line(polyline, fill=ink, width=2, joint="curve")
        for x, y in polyline:
            draw.polygon(((x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)), outline=ink, fill=(251, 252, 250))
        center = (305, 82)
    return f"{registration.renderer_family}:{family}", masks, center


def _degrade(image: Image.Image, split: str, index: int) -> tuple[Image.Image, str]:
    if split == "train":
        family = ("train_gamma", "train_jpeg", "train_anisotropic", "train_scanline")[index % 4]
        if family == "train_gamma":
            array = np.asarray(image, dtype=np.float32) / 255.0
            return Image.fromarray(np.clip((array ** 1.12) * 255.0, 0, 255).astype(np.uint8), "RGB"), family
        if family == "train_jpeg":
            stream = BytesIO()
            image.save(stream, format="JPEG", quality=78, subsampling=0)
            return Image.open(BytesIO(stream.getvalue())).convert("RGB"), family
        if family == "train_anisotropic":
            reduced = image.resize((331, 181), Image.Resampling.BILINEAR)
            return reduced.resize(image.size, Image.Resampling.LANCZOS), family
        array = np.asarray(image, dtype=np.int16).copy()
        array[5::19, :, :] = np.clip(array[5::19, :, :] - 7, 0, 255)
        return Image.fromarray(array.astype(np.uint8), "RGB"), family
    if split == "validation":
        family = ("validation_channel_shift", "validation_soft_raster", "validation_haze", "validation_impulse")[index % 4]
        if family == "validation_channel_shift":
            array = np.asarray(image, dtype=np.int16).copy()
            array[:, :, 0] = np.clip(array[:, :, 0] + 5, 0, 255)
            array[:, :, 2] = np.clip(array[:, :, 2] - 4, 0, 255)
            return Image.fromarray(array.astype(np.uint8), "RGB"), family
        if family == "validation_soft_raster":
            reduced = image.resize((365, 171), Image.Resampling.BILINEAR)
            return reduced.resize(image.size, Image.Resampling.BICUBIC), family
        if family == "validation_haze":
            return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.35))).enhance(0.78), family
        array = np.asarray(image, dtype=np.uint8).copy()
        y = 11 + ((index * 17) % 153)
        x = 7 + ((index * 29) % 351)
        array[y : y + 1, x : x + 4, :] = 244
        return Image.fromarray(array, "RGB"), family
    family = ("sealed_posterize", "sealed_dropout", "sealed_median", "sealed_microcontrast")[index % 4]
    if family == "sealed_posterize":
        array = np.asarray(image, dtype=np.uint8)
        return Image.fromarray(((array // 6) * 6).astype(np.uint8), "RGB"), family
    if family == "sealed_dropout":
        array = np.asarray(image, dtype=np.uint8).copy()
        y = 9 + ((index * 11) % 161)
        x = 63 + ((index * 31) % 293)
        array[y : y + 2, x : x + 8, :] = 247
        return Image.fromarray(array, "RGB"), family
    if family == "sealed_median":
        return image.filter(ImageFilter.MedianFilter(3)), family
    return ImageEnhance.Contrast(image).enhance(0.88), family


def _render_source(split: str, index: int) -> _RenderedSource:
    registration = split_registration(split)
    count = registration.text_count + registration.exclusion_count
    if not 0 <= index < count:
        raise ValueError("Ignore-band source index is out of range")
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
        multiplier = 9 if split == "train" else 11 if split == "validation" else 13
        text = GENERIC_TEXT[(index * multiplier + 2) % len(GENERIC_TEXT)]
        size = 15 + ((index * (7 if split == "sealed_public" else 5)) % 16)
        if len(text) > 8:
            size = min(size, 21)
        font = _font(rng, size)
        raw = draw.textbbox((0, 0), text, font=font)
        width = raw[2] - raw[0]
        height = raw[3] - raw[1]
        anchors = {
            "train": ((82, 7), (226, 118), (9, 89), (118, 8), (213, 138), (77, 139), (281, 8)),
            "validation": ((91, 9), (225, 121), (12, 76), (131, 7), (207, 136), (72, 129), (276, 10)),
            "sealed_public": ((74, 12), (236, 112), (15, 101), (143, 6), (201, 140), (88, 127), (270, 13)),
        }[split]
        x, y = anchors[(index * 3 + (1 if split == "validation" else 2 if split == "sealed_public" else 0)) % len(anchors)]
        x = min(max(4, x + int(rng.integers(-5, 6))), FRAME_WIDTH - width - 5)
        y = min(max(3, y + int(rng.integers(-4, 5))), FRAME_HEIGHT - height - 5)
        ink = tuple(int(value) for value in rng.integers(5, 49, size=3))
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
    distance = (width * height) * (1.0 - (DB_SHRINK_RATIO * DB_SHRINK_RATIO)) / (2.0 * (width + height))
    shrunk_left = int(math.ceil(left + distance))
    shrunk_top = int(math.ceil(top + distance))
    shrunk_right = int(math.floor(right - distance))
    shrunk_bottom = int(math.floor(bottom - distance))
    if shrunk_right <= shrunk_left or shrunk_bottom <= shrunk_top:
        raise RuntimeError("Ignore-band DB shrink target collapsed")
    result = np.zeros_like(source_target)
    result[shrunk_top:shrunk_bottom, shrunk_left:shrunk_right] = 255
    return result


def _supervision_mask(source_target: np.ndarray, positive_target: np.ndarray) -> np.ndarray:
    if not np.any(source_target):
        return np.full_like(source_target, 255)
    kernel_size = 1 + (2 * IGNORE_BAND_EXPANSION_PIXELS)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    ignored = cv2.dilate(source_target, kernel, iterations=1)
    result = np.full_like(source_target, 255)
    result[ignored > 0] = 0
    result[positive_target > 0] = 255
    return result


def render_training_patch(index: int) -> TrainingPatch:
    if not 0 <= index < TRAIN_SAMPLE_COUNT:
        raise ValueError("Ignore-band training index is out of range")
    source = _render_source("train", index)
    bgr = _production_resize(source.detector_bgr, cv2.INTER_LINEAR)
    full_target = _production_resize(source.target, cv2.INTER_NEAREST)
    source_positive = _shrink_target(source.target)
    target = _production_resize(source_positive, cv2.INTER_NEAREST)
    supervision = _production_resize(_supervision_mask(source.target, source_positive), cv2.INTER_NEAREST)
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
        scale_x = bgr.shape[1] / FRAME_WIDTH
        scale_y = bgr.shape[0] / FRAME_HEIGHT
        center_x = int(round(source.structure_center[0] * scale_x))
        center_y = int(round(source.structure_center[1] * scale_y))
        left = min(max(0, center_x - PATCH_WIDTH // 2), maximum_left)
        top = min(max(0, center_y - PATCH_HEIGHT // 2), maximum_top)
    right, bottom = left + PATCH_WIDTH, top + PATCH_HEIGHT
    return TrainingPatch(
        sample_id=f"graph-text-ignore-band-v3-train-{index:05d}",
        kind=source.kind,
        bgr=np.ascontiguousarray(bgr[top:bottom, left:right, :]),
        target=np.ascontiguousarray(target[top:bottom, left:right]),
        supervision_mask=np.ascontiguousarray(supervision[top:bottom, left:right]),
        renderer_family=source.renderer_family,
        degradation_family=source.degradation_family,
    )


def build_training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [render_training_patch(index) for index in range(TRAIN_SAMPLE_COUNT)]
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
        case_id=f"graph-text-ignore-band-v3-{split}-{source.kind}-{index:04d}",
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
            records.append({
                "sample_id": sample.sample_id,
                "kind": sample.kind,
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
