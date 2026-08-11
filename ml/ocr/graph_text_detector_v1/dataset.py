# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic synthetic selection data for graph text-region detection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .protocol import (
    PATCH_HEIGHT,
    PATCH_WIDTH,
    TRAIN_SAMPLE_COUNT,
    VALIDATION_EXCLUSION_COUNT,
    VALIDATION_TEXT_COUNT,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FRAME_WIDTH = 384
FRAME_HEIGHT = 192
FONT_PATHS = (
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)
GENERIC_TEXT = (
    "0",
    "12",
    "-6",
    "3.5",
    "80%",
    "Baseline",
    "Treatment",
    "Follow-up",
    "Transfer",
    "Jordan",
    "Morgan",
    "Phase A",
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


def _rng(split: str, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"graph-text-detector-v1:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(rng: np.random.Generator, size: int) -> ImageFont.FreeTypeFont:
    path = REPO_ROOT / FONT_PATHS[int(rng.integers(0, len(FONT_PATHS)))]
    return ImageFont.truetype(str(path), size=size)


def _draw_patch_structures(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    ink = tuple(int(value) for value in rng.integers(22, 70, size=3))
    choice = int(rng.integers(0, 7))
    if choice == 0:
        points = [(18, 91), (57, 55), (99, 76), (143, 42), (194, 68), (234, 35)]
        draw.line(points, fill=ink, width=int(rng.integers(1, 4)), joint="curve")
        for point_index, (x, y) in enumerate(points):
            radius = int(rng.integers(3, 7))
            box = (x - radius, y - radius, x + radius, y + radius)
            if (point_index + choice) % 2:
                draw.ellipse(box, outline=ink, fill=(250, 250, 248), width=2)
            else:
                draw.ellipse(box, fill=ink)
    elif choice == 1:
        draw.rounded_rectangle((150, 21, 245, 69), radius=4, outline=ink, width=2)
        draw.ellipse((162, 37, 173, 48), fill=ink)
        draw.line((181, 43, 229, 43), fill=ink, width=2)
    elif choice == 2:
        draw.line((39, 21, 39, 110), fill=ink, width=2)
        draw.line((39, 110, 235, 110), fill=ink, width=2)
        for offset in range(5):
            draw.line((68 + offset * 37, 106, 68 + offset * 37, 115), fill=ink, width=1)
    elif choice == 3:
        draw.line((31, 23, 31, 104), fill=ink, width=2)
        draw.line((31, 23, 76, 23), fill=ink, width=2)
        draw.line((31, 104, 76, 104), fill=ink, width=2)
    elif choice == 4:
        draw.line((42, 91, 201, 30), fill=ink, width=2)
        draw.polygon(((201, 30), (185, 32), (195, 45)), fill=ink)
    elif choice == 5:
        draw.line((44, 25, 215, 101), fill=ink, width=2)
        draw.line((44, 101, 215, 25), fill=ink, width=2)
    else:
        x = int(rng.integers(60, 210))
        draw.line((x, 10, x, 118), fill=ink, width=2)


def _degrade_patch(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    action = int(rng.integers(0, 4))
    if action == 0:
        return image, "train_clean_tint"
    if action == 1:
        return image.filter(ImageFilter.GaussianBlur(float(rng.choice((0.3, 0.55, 0.8))))), "train_ink_bleed"
    if action == 2:
        reduced = image.resize((224, 112), Image.Resampling.BILINEAR)
        return reduced.resize(image.size, Image.Resampling.BICUBIC), "train_resample"
    array = np.asarray(image, dtype=np.int16).copy()
    for _ in range(int(rng.integers(8, 28))):
        y = int(rng.integers(0, PATCH_HEIGHT))
        x = int(rng.integers(0, PATCH_WIDTH))
        array[y, x, :] = int(rng.integers(160, 235))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), "train_speckle"


def render_training_patch(index: int) -> TrainingPatch:
    if not 0 <= index < TRAIN_SAMPLE_COUNT:
        raise ValueError("Training patch index is out of range")
    registration = split_registration("train")
    rng = _rng("train", index)
    background = tuple(int(value) for value in rng.integers(246, 256, size=3))
    image = Image.new("RGB", (PATCH_WIDTH, PATCH_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    _draw_patch_structures(draw, rng)
    target = Image.new("L", (PATCH_WIDTH, PATCH_HEIGHT), 0)
    target_draw = ImageDraw.Draw(target)
    kind = "text" if index < registration.text_count else "exclusion"
    if kind == "text":
        text = GENERIC_TEXT[int(rng.integers(0, len(GENERIC_TEXT)))]
        size = int(rng.integers(25, 47)) if len(text) <= 4 else int(rng.integers(19, 33))
        font = _font(rng, size)
        raw = draw.textbbox((0, 0), text, font=font)
        width = raw[2] - raw[0]
        height = raw[3] - raw[1]
        x = int(rng.integers(8, max(9, PATCH_WIDTH - width - 8)))
        y = int(rng.integers(5, max(6, PATCH_HEIGHT - height - 8)))
        ink = tuple(int(value) for value in rng.integers(8, 62, size=3))
        draw.text((x, y), text, font=font, fill=ink)
        box = draw.textbbox((x, y), text, font=font)
        target_draw.rounded_rectangle(
            (max(0, box[0] - 2), max(0, box[1] - 2), min(PATCH_WIDTH - 1, box[2] + 2), min(PATCH_HEIGHT - 1, box[3] + 2)),
            radius=2,
            fill=255,
        )
    image, degradation = _degrade_patch(image, rng)
    rgb = np.asarray(image, dtype=np.uint8)
    return TrainingPatch(
        sample_id=f"graph-text-detector-v1-train-{index:05d}",
        kind=kind,
        bgr=np.ascontiguousarray(rgb[:, :, ::-1]),
        target=np.asarray(target, dtype=np.uint8).copy(),
        renderer_family=registration.renderer_family,
        degradation_family=degradation,
    )


def build_training_arrays() -> tuple[np.ndarray, np.ndarray]:
    samples = [render_training_patch(index) for index in range(TRAIN_SAMPLE_COUNT)]
    return (
        np.stack([sample.bgr for sample in samples]).astype(np.uint8),
        np.stack([sample.target for sample in samples])[:, None, :, :].astype(np.uint8),
    )


def _mask(masks: list[dict[str, int]], kind: str, left: int, top: int, right: int, bottom: int) -> None:
    masks.append({"kind": kind, "left": left, "top": top, "right": right, "bottom": bottom})


def _draw_frame_structures(
    draw: ImageDraw.ImageDraw,
    index: int,
    *,
    sealed: bool,
) -> tuple[str, list[dict[str, int]]]:
    families = (
        "paired_series_with_open_markers",
        "offset_phase_divider",
        "compact_legend_and_arrow",
        "bracket_and_intersection",
    )
    family = families[(index * (5 if sealed else 3) + (1 if sealed else 0)) % len(families)]
    masks: list[dict[str, int]] = []
    axis_x = 57 + (index % 7)
    axis_y = 160 - (index % 5)
    draw.line((axis_x, 31, axis_x, axis_y), fill=(30, 32, 34), width=2)
    draw.line((axis_x, axis_y, 360, axis_y), fill=(30, 32, 34), width=2)
    _mask(masks, "y_axis", axis_x - 3, 28, axis_x + 4, axis_y + 4)
    _mask(masks, "x_axis", axis_x - 3, axis_y - 3, 363, axis_y + 4)
    for tick in range(6):
        x = axis_x + 32 + tick * 44
        y = axis_y - 22 - ((tick * 19 + index * 7) % 82)
        draw.line((x, axis_y - 5, x, axis_y + 5), fill=(34, 34, 34), width=1)
        draw.line((axis_x - 5, y, axis_x + 5, y), fill=(34, 34, 34), width=1)
        _mask(masks, "x_tick", x - 2, axis_y - 7, x + 3, axis_y + 8)
        _mask(masks, "y_tick", axis_x - 7, y - 2, axis_x + 8, y + 3)
    points = [(103, 119), (145, 81), (190, 105), (234, 63), (282, 88), (328, 51)]
    draw.line(points, fill=(28, 29, 30), width=2, joint="curve")
    for point_index, (x, y) in enumerate(points):
        radius = 4 + ((point_index + index) % 2)
        if (point_index + index) % 2:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(22, 22, 22), fill=(252, 252, 250), width=2)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(22, 22, 22))
    if family == "offset_phase_divider":
        x = 205 + (index % 9)
        draw.line((x, 25, x, axis_y), fill=(31, 31, 31), width=2)
        _mask(masks, "phase_divider", x - 3, 22, x + 4, axis_y + 3)
    elif family == "compact_legend_and_arrow":
        draw.rounded_rectangle((257, 32, 357, 78), radius=4, outline=(31, 31, 31), width=2)
        draw.ellipse((269, 48, 279, 58), fill=(24, 24, 24))
        draw.line((288, 53, 341, 53), fill=(31, 31, 31), width=2)
        draw.line((235, 119, 315, 91), fill=(28, 28, 28), width=2)
        draw.polygon(((315, 91), (299, 91), (309, 105)), fill=(28, 28, 28))
    elif family == "bracket_and_intersection":
        draw.line((252, 36, 252, 91), fill=(28, 28, 28), width=2)
        draw.line((252, 36, 281, 36), fill=(28, 28, 28), width=2)
        draw.line((252, 91, 281, 91), fill=(28, 28, 28), width=2)
        draw.line((297, 31, 355, 90), fill=(28, 28, 28), width=2)
        draw.line((297, 90, 355, 31), fill=(28, 28, 28), width=2)
    return family, masks


def _degrade_frame(image: Image.Image, index: int, *, sealed: bool) -> tuple[Image.Image, str]:
    if sealed:
        family = ("sealed_faint_scan", "sealed_block_dropout", "sealed_chroma_shift", "sealed_soft_median")[index % 4]
        if family == "sealed_faint_scan":
            return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.45))).enhance(0.78), family
        if family == "sealed_block_dropout":
            array = np.asarray(image, dtype=np.uint8).copy()
            y = 14 + ((index * 11) % 143)
            x = 70 + ((index * 17) % 274)
            array[y : y + 2, x : x + 5, :] = 245
            return Image.fromarray(array, "RGB"), family
        if family == "sealed_chroma_shift":
            array = np.asarray(image, dtype=np.int16).copy()
            array[:, :, 0] = np.clip(array[:, :, 0] + 3, 0, 255)
            array[:, :, 2] = np.clip(array[:, :, 2] - 2, 0, 255)
            return Image.fromarray(array.astype(np.uint8), "RGB"), family
        return image.filter(ImageFilter.MedianFilter(3)), family
    family = ("validation_clean", "validation_fax", "validation_gamma", "validation_banding")[index % 4]
    if family == "validation_clean":
        return image, family
    if family == "validation_fax":
        reduced = image.resize((331, 166), Image.Resampling.BILINEAR)
        return reduced.resize(image.size, Image.Resampling.BICUBIC), family
    if family == "validation_gamma":
        return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.28))).enhance(0.86), family
    array = np.asarray(image, dtype=np.uint8).copy()
    array[:, ::13, :] = np.clip(array[:, ::13, :].astype(np.int16) - 5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB"), family


def _render_evaluation_frame(split: str, index: int) -> EvaluationFrame:
    sealed = split == "sealed_public"
    registration = split_registration(split)
    count = registration.text_count + registration.exclusion_count
    if not 0 <= index < count:
        raise ValueError("Evaluation frame index is out of range")
    rng = _rng(split, index)
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), tuple(int(value) for value in rng.integers(248, 256, size=3)))
    draw = ImageDraw.Draw(image)
    structure_family, masks = _draw_frame_structures(draw, index, sealed=sealed)
    kind = "text" if index < registration.text_count else "exclusion"
    truth_bbox: tuple[float, float, float, float] | None = None
    if kind == "text":
        text = GENERIC_TEXT[(index * (7 if sealed else 5) + 2) % len(GENERIC_TEXT)]
        size = 18 + ((index * (5 if sealed else 3)) % 13)
        if len(text) > 6:
            size = min(size, 24)
        font = _font(rng, size)
        raw = draw.textbbox((0, 0), text, font=font)
        width = raw[2] - raw[0]
        height = raw[3] - raw[1]
        anchors = ((86, 12), (279, 103), (11, 88), (143, 13), (256, 126), (83, 132))
        x, y = anchors[(index * 5 + (1 if sealed else 0)) % len(anchors)]
        x = min(x, FRAME_WIDTH - width - 6)
        y = min(y, FRAME_HEIGHT - height - 6)
        draw.text((x, y), text, font=font, fill=tuple(int(value) for value in rng.integers(8, 54, size=3)))
        truth_bbox = tuple(float(value) for value in draw.textbbox((x, y), text, font=font))
    image, degradation = _degrade_frame(image, index, sealed=sealed)
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    png = stream.getvalue()
    with Image.open(BytesIO(png)) as loaded:
        rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    for rectangle in masks:
        bgr[rectangle["top"] : rectangle["bottom"], rectangle["left"] : rectangle["right"], :] = 255
    detector_bgr = bgr.tobytes(order="C")
    case_id = f"graph-text-detector-v1-{split}-{kind}-{index:04d}"
    return EvaluationFrame(
        case_id=case_id,
        kind=kind,
        source_png=png,
        source_sha256=sha256(png).hexdigest(),
        detector_bgr=detector_bgr,
        detector_bgr_sha256=sha256(detector_bgr).hexdigest(),
        truth_bbox=truth_bbox,
        renderer_family=registration.renderer_family,
        degradation_family=degradation,
        structure_family=structure_family,
    )


def build_validation_split() -> tuple[EvaluationFrame, ...]:
    count = VALIDATION_TEXT_COUNT + VALIDATION_EXCLUSION_COUNT
    return tuple(_render_evaluation_frame("validation", index) for index in range(count))


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
                    "truth_bbox": sample.truth_bbox,
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
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "TrainingPatch",
    "build_training_arrays",
    "build_validation_split",
    "render_training_patch",
    "split_fingerprint",
    "training_split_fingerprint",
]
