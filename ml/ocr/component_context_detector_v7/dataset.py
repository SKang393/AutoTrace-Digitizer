# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes and dual-context proposal encoding for OCR V7."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.ocr.component_region_detector_v6.dataset import Box, Component, box_iou, connected_components

from .protocol import (
    CONTEXT_HORIZONTAL_PADDING_HEIGHT_RATIO,
    CONTEXT_MINIMUM_PADDING_PIXELS,
    CONTEXT_VERTICAL_PADDING_HEIGHT_RATIO,
    CROP_HEIGHT,
    CROP_WIDTH,
    ENCODED_WIDTH,
    GEOMETRY_FEATURE_COUNT,
    INPUT_CHANNELS,
    MAXIMUM_COMPONENT_HEIGHT_RATIO_WITHIN_LINE,
    PROPOSAL_THRESHOLD_MAXIMUM,
    PROPOSAL_THRESHOLD_MEAN_RATIO,
    PROPOSAL_THRESHOLD_MINIMUM,
    REVISION,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    SEED,
    MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO,
    MAXIMUM_MERGED_HEIGHT_GROWTH_RATIO,
    MINIMUM_VERTICAL_OVERLAP_RATIO,
    TIGHT_HORIZONTAL_PADDING_PIXELS,
    TIGHT_VERTICAL_PADDING_RATIO,
    TRUTH_MATCH_IOU_MINIMUM,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]
_WORDS = (
    "Baseline",
    "Treatment",
    "Followup",
    "Observer",
    "Measure",
    "Session",
    "Response",
    "Phase",
    "Probe",
    "Level",
    "Maintenance",
    "Intervention",
)
_NUMBERS = ("0", "17", "-8", "3.5", "42%", "100", "-0.7", "9.25", "24", "85")


@dataclass(frozen=True)
class SceneSample:
    scene_id: str
    split: Split
    renderer_family: str
    degradation_family: str
    raster: np.ndarray
    truths: tuple[Box, ...]


def _rng(split: Split, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"{SEED}:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: Split, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    path = REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]
    return ImageFont.truetype(str(path), size=size)


def _draw_text_mask(
    size: tuple[int, int],
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
) -> tuple[Image.Image, Box]:
    mask = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.text(position, text, fill=255, font=font)
    bounds = mask.getbbox()
    if bounds is None:
        raise RuntimeError("Rendered OCR V7 text has no foreground")
    return mask, Box(*bounds)


def _draw_structures(draw: ImageDraw.ImageDraw, rng: np.random.Generator, index: int, ink: int) -> None:
    left = 112 + int(rng.integers(-5, 6))
    top = 52 + int(rng.integers(-3, 4))
    right = 516 + int(rng.integers(-5, 6))
    bottom = 258 + int(rng.integers(-3, 4))
    draw.line((left, top, left, bottom), fill=ink, width=2)
    draw.line((left, bottom, right, bottom), fill=ink, width=2)
    for x in range(left + 34, right - 5, 38):
        draw.line((x, bottom - 5, x, bottom + 5), fill=ink, width=2)
    for y in range(top + 32, bottom - 5, 38):
        draw.line((left - 5, y, left + 5, y), fill=ink, width=2)
    for offset in (0, 1):
        divider_x = left + 145 + offset * 132 + int(rng.integers(-8, 9))
        draw.line((divider_x, top + 2, divider_x, bottom - 2), fill=ink, width=1 + (index + offset) % 2)

    points = []
    for step in range(10):
        x = left + 24 + step * 39
        y = bottom - 38 - int(35 * np.sin((step + index % 5) * 0.61))
        points.append((x, y))
    draw.line(points, fill=ink, width=2)
    for step, (x, y) in enumerate(points):
        radius = 3 + ((step + index) % 3)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=ink,
            fill=ink if (step + index) % 2 else None,
            width=2,
        )

    legend_left = 525 + int(rng.integers(-3, 4))
    legend_top = 196 + int(rng.integers(-5, 6))
    draw.rectangle((legend_left, legend_top, 626, 263), outline=ink, width=2)
    draw.line((legend_left + 10, legend_top + 18, legend_left + 42, legend_top + 18), fill=ink, width=2)
    draw.ellipse((legend_left + 22, legend_top + 14, legend_left + 30, legend_top + 22), outline=ink, width=2)
    draw.line((legend_left + 10, legend_top + 44, legend_left + 42, legend_top + 44), fill=ink, width=2)
    draw.rectangle((legend_left + 22, legend_top + 40, legend_left + 30, legend_top + 48), outline=ink, fill=ink)

    bracket_y = top + 13 + int(rng.integers(0, 12))
    bracket_left = left + 35 + int(rng.integers(0, 42))
    bracket_right = bracket_left + 74 + int(rng.integers(0, 24))
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 18), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 18), fill=ink, width=2)

    arrow_y = top + 72 + int(rng.integers(0, 38))
    arrow_left = right - 114
    arrow_right = right - 34
    draw.line((arrow_left, arrow_y, arrow_right, arrow_y), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 14, arrow_y - 9), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 14, arrow_y + 9), fill=ink, width=2)

    cross_x = left + 58 + int(rng.integers(0, 45))
    cross_y = top + 122 + int(rng.integers(0, 44))
    draw.line((cross_x - 18, cross_y, cross_x + 18, cross_y), fill=ink, width=2)
    draw.line((cross_x, cross_y - 18, cross_x, cross_y + 18), fill=ink, width=2)
    triangle_x = right - 50
    triangle_y = bottom - 40
    draw.polygon(
        ((triangle_x, triangle_y - 8), (triangle_x - 8, triangle_y + 7), (triangle_x + 8, triangle_y + 7)),
        outline=ink,
    )


def _render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(246, 256))
    ink = int(rng.integers(10, 70))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.uint8)
    horizontal_ramp = np.linspace(0, float(rng.integers(0, 7)), SCENE_WIDTH, dtype=np.float32)
    vertical_ramp = np.linspace(0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)
    base = np.clip(base.astype(np.float32) - horizontal_ramp[None, :] - vertical_ramp[:, None], 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="L")
    draw = ImageDraw.Draw(image)
    _draw_structures(draw, rng, index, ink)

    annotation_x = 385 + int(rng.integers(0, 55))
    annotation_y = 115 + int(rng.integers(0, 48))
    draw.rectangle((annotation_x - 5, annotation_y - 3, 625, annotation_y + 30), fill=background)
    labels = (
        _NUMBERS[(index * 3 + 1) % len(_NUMBERS)],
        _NUMBERS[(index * 7 + 4) % len(_NUMBERS)],
        _WORDS[(index * 5 + 2) % len(_WORDS)],
        _WORDS[(index * 7 + 5) % len(_WORDS)],
    )
    y_slot = 72 + int(rng.integers(0, 96))
    x_slot = 166 + int(rng.integers(0, 260))
    heading_slot = 148 + int(rng.integers(0, 170))
    if split == "validation":
        y_slot = 78 + (index * 17) % 88
        x_slot = 180 + (index * 29) % 230
        heading_slot = 172 + (index * 31) % 145
    elif split == "sealed_public":
        y_slot = 82 + (index * 23) % 82
        x_slot = 196 + (index * 37) % 214
        heading_slot = 158 + (index * 43) % 154
    positions = ((18, y_slot), (x_slot, 282), (heading_slot, 8), (annotation_x, annotation_y))
    truths: list[Box] = []
    for label_index, (label, position) in enumerate(zip(labels, positions, strict=True)):
        font = _font(split, index + label_index, int(rng.integers(16, 22)))
        mask, bounds = _draw_text_mask(image.size, label, position, font)
        image.paste(ink, mask=mask)
        truths.append(bounds)

    if split == "train":
        pixels = np.asarray(image, dtype=np.uint8).copy().astype(np.float32)
        gamma = float(rng.uniform(0.92, 1.08))
        pixels = 255.0 * np.power(pixels / 255.0, gamma)
        image = Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8), mode="L")
        if index % 3 == 0:
            image = image.filter(ImageFilter.BoxBlur(radius=0.25))
        pixels = np.asarray(image, dtype=np.uint8).copy()
        for _ in range(int(rng.integers(0, 4))):
            column = int(rng.integers(0, SCENE_WIDTH))
            pixels[:, column] = np.minimum(pixels[:, column], int(rng.integers(228, 244)))
        image = Image.fromarray(pixels, mode="L")
    elif split == "validation":
        reduced = image.resize((592, 292), resample=Image.Resampling.BICUBIC)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.uint8).copy()
        row = 45 + (index * 19) % 230
        pixels[row, :] = np.minimum(pixels[row, :], 236)
        image = Image.fromarray(pixels, mode="L")
    else:
        pixels = np.asarray(image, dtype=np.uint8).copy().astype(np.float32)
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(0.92, 1.03)) + 128.0, 0, 255)
        pixels = (np.rint(pixels).astype(np.uint16) // 6 * 6).clip(0, 255).astype(np.uint8)
        foreground = np.argwhere(pixels < 170)
        if len(foreground) and index % 2 == 0:
            for selected in rng.choice(len(foreground), size=min(3, len(foreground)), replace=False):
                y, x = foreground[int(selected)]
                pixels[int(y), int(x)] = int(rng.integers(240, 251))
        image = Image.fromarray(pixels, mode="L")

    raster = np.asarray(image, dtype=np.uint8).copy()
    for _ in range(int(rng.integers(0, 8))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(188, 239))
    return SceneSample(
        f"component-context-v7-{split}-{index:05d}",
        split,
        registration.renderer_family,
        registration.degradation_family,
        raster,
        tuple(truths),
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(_render_scene(split, index) for index in range(registration.scene_count))


def proposals(gray: np.ndarray) -> tuple[Component, ...]:
    remaining = sorted(connected_components(gray), key=lambda item: (item.top, item.left, item.bottom, item.right))
    lines: list[Component] = []
    while remaining:
        line = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for index in range(len(remaining) - 1, -1, -1):
                candidate = remaining[index]
                minimum_height = max(1, min(line.height, candidate.height))
                maximum_height = max(line.height, candidate.height)
                if maximum_height / minimum_height > MAXIMUM_COMPONENT_HEIGHT_RATIO_WITHIN_LINE:
                    continue
                overlap = max(0, min(line.bottom, candidate.bottom) - max(line.top, candidate.top) + 1)
                overlap_fraction = overlap / minimum_height
                if overlap_fraction < MINIMUM_VERTICAL_OVERLAP_RATIO:
                    continue
                if candidate.left > line.right:
                    gap = candidate.left - line.right - 1
                elif line.left > candidate.right:
                    gap = line.left - candidate.right - 1
                else:
                    gap = 0
                if gap > maximum_height * MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO:
                    continue
                merged = line.merge(candidate)
                if merged.height > maximum_height * MAXIMUM_MERGED_HEIGHT_GROWTH_RATIO:
                    continue
                line = merged
                remaining.pop(index)
                changed = True
        lines.append(line)
    return tuple(sorted(lines, key=lambda item: (item.top, item.left, item.bottom, item.right)))


def proposal_labels(scene: SceneSample, items: tuple[Component, ...] | None = None) -> np.ndarray:
    candidates = proposals(scene.raster) if items is None else items
    labels = [int(any(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for truth in scene.truths)) for candidate in candidates]
    return np.asarray(labels, dtype=np.int64)


def _sample_bilinear(gray: np.ndarray, x: float, y: float) -> float:
    x = float(np.clip(x, 0.0, gray.shape[1] - 1.0))
    y = float(np.clip(y, 0.0, gray.shape[0] - 1.0))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, gray.shape[1] - 1)
    y1 = min(y0 + 1, gray.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    top = gray[y0, x0] * (1.0 - wx) + gray[y0, x1] * wx
    bottom = gray[y1, x0] * (1.0 - wx) + gray[y1, x1] * wx
    return float(top * (1.0 - wy) + bottom * wy)


def _encode_crop(gray: np.ndarray, left: float, top: float, width: float, height: float) -> np.ndarray:
    content_width = int(np.clip(np.ceil(CROP_HEIGHT * width / height), 1, CROP_WIDTH))
    output = np.full((CROP_HEIGHT, CROP_WIDTH), 255.0, dtype=np.float32)
    for target_y in range(CROP_HEIGHT):
        source_y = top + ((target_y + 0.5) / CROP_HEIGHT) * height - 0.5
        for target_x in range(content_width):
            source_x = left + ((target_x + 0.5) / content_width) * width - 0.5
            if 0.0 <= source_x < gray.shape[1] and 0.0 <= source_y < gray.shape[0]:
                output[target_y, target_x] = _sample_bilinear(gray, source_x, source_y)
    return 1.0 - np.rint(output).clip(0, 255).astype(np.float32) / 255.0


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    tight_vertical_padding = max(1.0, proposal.height * TIGHT_VERTICAL_PADDING_RATIO)
    tight_left = proposal.left - TIGHT_HORIZONTAL_PADDING_PIXELS
    tight_top = proposal.top - tight_vertical_padding
    tight_width = proposal.width + 2.0 * TIGHT_HORIZONTAL_PADDING_PIXELS
    tight_height = proposal.height + 2.0 * tight_vertical_padding
    tight = _encode_crop(gray, tight_left, tight_top, tight_width, tight_height)

    context_horizontal_padding = max(CONTEXT_MINIMUM_PADDING_PIXELS, proposal.height * CONTEXT_HORIZONTAL_PADDING_HEIGHT_RATIO)
    context_vertical_padding = max(CONTEXT_MINIMUM_PADDING_PIXELS, proposal.height * CONTEXT_VERTICAL_PADDING_HEIGHT_RATIO)
    context_left = proposal.left - context_horizontal_padding
    context_top = proposal.top - context_vertical_padding
    context_width = proposal.width + 2.0 * context_horizontal_padding
    context_height = proposal.height + 2.0 * context_vertical_padding
    context = _encode_crop(gray, context_left, context_top, context_width, context_height)

    threshold = float(np.clip(round(float(gray.mean()) * PROPOSAL_THRESHOLD_MEAN_RATIO), PROPOSAL_THRESHOLD_MINIMUM, PROPOSAL_THRESHOLD_MAXIMUM))
    x0 = max(0, int(np.floor(context_left)))
    y0 = max(0, int(np.floor(context_top)))
    x1 = min(gray.shape[1], int(np.ceil(context_left + context_width)))
    y1 = min(gray.shape[0], int(np.ceil(context_top + context_height)))
    raw_context = gray[y0:y1, x0:x1] <= threshold
    maximum_row_density = float(raw_context.sum(axis=1).max() / max(1, raw_context.shape[1])) if raw_context.size else 0.0
    maximum_column_density = float(raw_context.sum(axis=0).max() / max(1, raw_context.shape[0])) if raw_context.size else 0.0
    edge_density = 0.0
    if raw_context.size:
        edge_values = np.concatenate((raw_context[0], raw_context[-1], raw_context[:, 0], raw_context[:, -1]))
        edge_density = float(edge_values.mean())
    geometry = np.asarray(
        (
            proposal.width / gray.shape[1],
            proposal.height / gray.shape[0],
            proposal.area / max(1, proposal.width * proposal.height),
            min(1.0, proposal.width / max(1.0, proposal.height * 8.0)),
            min(1.0, proposal.height / max(1.0, proposal.width * 4.0)),
            min(1.0, proposal.count / 16.0),
            threshold / 255.0,
            float(tight.mean()),
            float(context.mean()),
            maximum_row_density,
            maximum_column_density,
            edge_density,
        ),
        dtype=np.float32,
    )
    if len(geometry) != GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V7 geometry feature contract changed")
    encoded = np.zeros((INPUT_CHANNELS, CROP_HEIGHT, ENCODED_WIDTH), dtype=np.float32)
    encoded[0, :, :CROP_WIDTH] = tight
    encoded[1, :, :CROP_WIDTH] = context
    encoded[:, :, CROP_WIDTH:] = geometry[None, None, :]
    return encoded


def proposal_examples(scenes: tuple[SceneSample, ...]) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    for scene in scenes:
        candidates = proposals(scene.raster)
        scene_labels = proposal_labels(scene, candidates)
        values.extend(encode_proposal(scene.raster, proposal) for proposal in candidates)
        labels.extend(int(value) for value in scene_labels)
    if not values or not any(labels) or all(labels):
        raise RuntimeError("OCR V7 split did not produce both proposal classes")
    return np.stack(values).astype(np.float32), np.asarray(labels, dtype=np.int64)


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    records = [
        {
            "scene_id": scene.scene_id,
            "renderer_family": scene.renderer_family,
            "degradation_family": scene.degradation_family,
            "raster_sha256": sha256_bytes(scene.raster.tobytes()),
            "truths": [[box.left, box.top, box.right, box.bottom] for box in scene.truths],
        }
        for scene in scenes
    ]
    return sha256_bytes(canonical_json_bytes(records))


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, int]:
    proposal_count = positive_count = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        labels = proposal_labels(scene, candidates)
        proposal_count += len(candidates)
        positive_count += int(labels.sum())
        for truth in scene.truths:
            matches = sum(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for candidate in candidates)
            if matches != 1:
                raise RuntimeError(f"{scene.scene_id} truth has {matches} deterministic proposals")
    return {
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "proposal_count": proposal_count,
        "positive_proposal_count": positive_count,
        "negative_proposal_count": proposal_count - positive_count,
    }


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)


def save_sealed_public_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    if path.exists():
        raise RuntimeError(f"OCR V7 sealed archive already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    images: list[tuple[str, bytes]] = []
    for scene in scenes:
        name = f"images/{scene.scene_id}.png"
        payload = _png_bytes(scene.raster)
        images.append((name, payload))
        cases.append(
            {
                "scene_id": scene.scene_id,
                "image_path": name,
                "image_sha256": sha256_bytes(payload),
                "renderer_family": scene.renderer_family,
                "degradation_family": scene.degradation_family,
                "truths": [[box.left, box.top, box.right, box.bottom] for box in scene.truths],
            }
        )
    manifest = {
        "schema": "graphreader.ocr-component-context-sealed-fixtures.v1",
        "revision": REVISION,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "scene_count": len(scenes),
        "split_fingerprint": split_fingerprint(scenes),
        "cases": cases,
    }
    with ZipFile(path, "x") as archive:
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
        for name, payload in sorted(images):
            _zip_write(archive, name, payload)
    return {
        "schema": "graphreader.ocr-component-context-private-manifest.v1",
        "revision": REVISION,
        "scene_count": len(scenes),
        "split_fingerprint": split_fingerprint(scenes),
        "fixture_archive_sha256": sha256_file(path),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
    }


def load_sealed_public_archive(path: Path) -> tuple[SceneSample, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256_bytes(payload) != case["image_sha256"]:
                raise RuntimeError("OCR V7 sealed fixture image checksum changed")
            with Image.open(BytesIO(payload)) as image:
                raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            scenes.append(
                SceneSample(
                    case["scene_id"],
                    "sealed_public",
                    case["renderer_family"],
                    case["degradation_family"],
                    raster,
                    tuple(Box(*bounds) for bounds in case["truths"]),
                )
            )
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR V7 sealed fixture fingerprint changed")
    return result


__all__ = [
    "Box",
    "Component",
    "SceneSample",
    "box_iou",
    "build_split",
    "encode_proposal",
    "load_sealed_public_archive",
    "proposal_examples",
    "proposal_labels",
    "proposal_summary",
    "proposals",
    "save_sealed_public_archive",
    "split_fingerprint",
]
