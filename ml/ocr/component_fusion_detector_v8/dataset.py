# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes for the OCR component-fusion V8 experiment."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.ocr.component_context_detector_v7.dataset import (
    Box,
    Component,
    SceneSample,
    box_iou,
    encode_proposal as encode_v7_proposal,
    proposals,
)

from .protocol import (
    REVISION,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE,
    TRUTH_MATCH_IOU_MINIMUM,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = str
_NUMBERS = ("0", "5", "10", "20", "25", "40", "50", "60", "75", "80", "90", "100", "-2", "2.5", "33%")
_WORDS = ("Baseline", "Intervention", "Maintenance", "Followup", "Sessions", "Participant", "Probe", "Phase")


def _rng(split: Split, index: int) -> np.random.Generator:
    return np.random.default_rng(split_registration(split).seed_offset + index * 8_191)


def _font(split: Split, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    path = REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]
    return ImageFont.truetype(str(path), size=size)


def _text_mask(
    size: tuple[int, int], text: str, position: tuple[int, int], font: ImageFont.FreeTypeFont
) -> tuple[Image.Image, Box]:
    mask = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.text(position, text, fill=255, font=font)
    bounds = mask.getbbox()
    if bounds is None:
        raise RuntimeError("Rendered OCR V8 text has no foreground")
    return mask, Box(*bounds)


def _draw_plot_structures(draw: ImageDraw.ImageDraw, rng: np.random.Generator, index: int, ink: int) -> None:
    left = 105 + int(rng.integers(-7, 8))
    top = 48 + int(rng.integers(-4, 5))
    right = 508 + int(rng.integers(-6, 7))
    bottom = 257 + int(rng.integers(-4, 5))
    draw.line((left, top, left, bottom), fill=ink, width=1 + index % 2)
    draw.line((left, bottom, right, bottom), fill=ink, width=2)
    for x in range(left + 27, right - 3, 34 + index % 5):
        draw.line((x, bottom - 5, x, bottom + 5), fill=ink, width=2)
    for y in range(top + 25, bottom - 4, 34 + (index + 2) % 5):
        draw.line((left - 5, y, left + 5, y), fill=ink, width=2)
    for slot in range(2):
        divider_x = left + 135 + slot * 128 + int(rng.integers(-10, 11))
        draw.line((divider_x, top + 1, divider_x, bottom - 1), fill=ink, width=1 + (index + slot) % 2)

    points: list[tuple[int, int]] = []
    for step in range(11):
        x = left + 18 + step * 36
        y = bottom - 40 - int(31 * np.sin((step + index % 7) * 0.57))
        points.append((x, y))
    draw.line(points, fill=ink, width=1 + index % 2)
    for step, (x, y) in enumerate(points):
        radius = 3 + ((step + index) % 4)
        shape = (step + index) % 3
        if shape == 0:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)
        elif shape == 1:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, fill=ink)
        else:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)

    bracket_y = top + 10 + int(rng.integers(0, 18))
    bracket_left = left + 28 + int(rng.integers(0, 45))
    bracket_right = bracket_left + 70 + int(rng.integers(0, 28))
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 17), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 17), fill=ink, width=2)

    arrow_y = top + 72 + int(rng.integers(0, 55))
    arrow_left = right - 112
    arrow_right = right - 25
    draw.line((arrow_left, arrow_y, arrow_right, arrow_y), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 13, arrow_y - 8), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 13, arrow_y + 8), fill=ink, width=2)

    legend_left = 520 + int(rng.integers(-4, 5))
    legend_top = 201 + int(rng.integers(-5, 6))
    draw.rectangle((legend_left, legend_top, 629, 268), outline=ink, width=2)
    draw.line((legend_left + 9, legend_top + 18, legend_left + 43, legend_top + 18), fill=ink, width=2)
    draw.ellipse((legend_left + 22, legend_top + 14, legend_left + 30, legend_top + 22), outline=ink, width=2)
    draw.line((legend_left + 9, legend_top + 45, legend_left + 43, legend_top + 45), fill=ink, width=2)
    draw.rectangle((legend_left + 22, legend_top + 41, legend_left + 30, legend_top + 49), outline=ink, fill=ink)


def _draw_compact_hard_negatives(
    draw: ImageDraw.ImageDraw, rng: np.random.Generator, index: int, ink: int
) -> None:
    anchors = ((548, 52), (591, 92), (548, 137), (596, 169))
    for slot, (base_x, base_y) in enumerate(anchors):
        x = base_x + int(rng.integers(-5, 6))
        y = base_y + int(rng.integers(-4, 5))
        radius = 5 + ((index + slot) % 5)
        kind = (index + slot) % 6
        if kind == 0:
            draw.polygon(((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), outline=ink)
        elif kind == 1:
            draw.polygon(((x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)), outline=ink)
        elif kind == 2:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=1 + slot % 2)
        elif kind == 3:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=ink, width=1 + slot % 2)
        elif kind == 4:
            draw.line((x - radius, y - radius, x, y + radius, x + radius, y - radius), fill=ink, width=2)
        else:
            draw.line((x - radius, y, x + radius, y), fill=ink, width=2)
            draw.line((x, y - radius, x, y + radius), fill=ink, width=2)


def _render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(247, 256))
    ink = int(rng.integers(8, 66))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(0.0, float(rng.integers(0, 6)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    _draw_plot_structures(draw, rng, index, ink)
    _draw_compact_hard_negatives(draw, rng, index, ink)

    labels = (
        _NUMBERS[(index * 5 + 2) % len(_NUMBERS)],
        _NUMBERS[(index * 9 + 7) % len(_NUMBERS)],
        _WORDS[(index * 3 + 1) % len(_WORDS)],
        _WORDS[(index * 5 + 4) % len(_WORDS)],
    )
    y_slot = 70 + (index * 19) % 108
    x_slot = 168 + (index * 31) % 245
    heading_slot = 132 + (index * 37) % 192
    annotation_x = 356 + (index * 29) % 105
    annotation_y = 104 + (index * 23) % 77
    if split == "validation":
        y_slot = 76 + (index * 23) % 96
        x_slot = 178 + (index * 41) % 226
        heading_slot = 148 + (index * 43) % 171
        annotation_x = 372 + (index * 31) % 86
    elif split == "sealed_public":
        y_slot = 82 + (index * 29) % 87
        x_slot = 190 + (index * 47) % 211
        heading_slot = 157 + (index * 53) % 158
        annotation_x = 364 + (index * 37) % 94
    draw.rectangle((330, annotation_y - 4, 635, annotation_y + 31), fill=background)
    positions = ((17, y_slot), (x_slot, 282), (heading_slot, 7), (annotation_x, annotation_y))
    truths: list[Box] = []
    for label_index, (label, position) in enumerate(zip(labels, positions, strict=True)):
        font = _font(split, index + label_index, int(rng.integers(16, 23)))
        mask, bounds = _text_mask(image.size, label, position, font)
        image.paste(ink, mask=mask)
        truths.append(bounds)

    if split == "train":
        if index % 3 == 0:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        pixels = np.asarray(image, dtype=np.uint8).copy().astype(np.float32)
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(0.95, 1.07)) + 128.0, 0, 255)
        pixels -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_WIDTH, dtype=np.float32)[None, :]
        image = Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8), mode="L")
    elif split == "validation":
        reduced = image.resize((624, 312), resample=Image.Resampling.LANCZOS)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.uint8).copy()
        row = 38 + (index * 31) % 246
        pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :] = np.maximum(
            pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :], 170
        )
        image = Image.fromarray(pixels, mode="L")
    else:
        pixels = np.asarray(image, dtype=np.uint8).copy().astype(np.float32)
        pixels = 255.0 * np.power(pixels / 255.0, float(rng.uniform(0.94, 1.06)))
        pixels = (np.rint(pixels).astype(np.uint16) // 5 * 5).clip(0, 255).astype(np.uint8)
        foreground = np.argwhere(pixels < 175)
        if len(foreground) and index % 2 == 0:
            for selected in rng.choice(len(foreground), size=min(4, len(foreground)), replace=False):
                y, x = foreground[int(selected)]
                pixels[int(y), int(x)] = int(rng.integers(238, 252))
        image = Image.fromarray(pixels, mode="L")

    raster = np.asarray(image, dtype=np.uint8).copy()
    for _ in range(int(rng.integers(0, 7))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(188, 241))
    return SceneSample(
        f"component-fusion-v8-{split}-{index:05d}",
        split,
        registration.renderer_family,
        registration.degradation_family,
        raster,
        tuple(truths),
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(_render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    return encode_v7_proposal(gray, proposal)


def proposal_labels(scene: SceneSample, items: tuple[Component, ...] | None = None) -> np.ndarray:
    candidates = proposals(scene.raster) if items is None else items
    return np.asarray(
        [int(any(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for truth in scene.truths)) for candidate in candidates],
        dtype=np.int64,
    )


def proposal_examples(scenes: tuple[SceneSample, ...]) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    for scene in scenes:
        candidates = proposals(scene.raster)
        scene_labels = proposal_labels(scene, candidates)
        selected_indices = list(range(len(candidates)))
        if scene.split == "train":
            positive_indices = [index for index, value in enumerate(scene_labels) if value == 1]
            negative_indices = [index for index, value in enumerate(scene_labels) if value == 0]
            selected_indices = sorted(
                positive_indices + negative_indices[:TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE]
            )
        values.extend(encode_proposal(scene.raster, candidates[index]) for index in selected_indices)
        labels.extend(int(scene_labels[index]) for index in selected_indices)
    if not values or not any(labels) or all(labels):
        raise RuntimeError("OCR V8 split did not produce both proposal classes")
    return np.stack(values).astype(np.float32), np.asarray(labels, dtype=np.int64)


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode("utf-8"))
        digest.update(scene.renderer_family.encode("utf-8"))
        digest.update(scene.degradation_family.encode("utf-8"))
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(f"{truth.left},{truth.top},{truth.right},{truth.bottom}".encode("ascii"))
    return digest.hexdigest()


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, int]:
    proposal_count = 0
    positive_count = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        labels = proposal_labels(scene, candidates)
        proposal_count += len(candidates)
        positive_count += int(labels.sum())
        if any(sum(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for candidate in candidates) != 1 for truth in scene.truths):
            raise RuntimeError(f"OCR V8 proposal completeness failed: {scene.scene_id}")
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
        raise RuntimeError(f"OCR V8 sealed archive already exists: {path}")
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
        "schema": "graphreader.ocr-component-fusion-sealed-fixtures.v1",
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
        "schema": "graphreader.ocr-component-fusion-private-manifest.v1",
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
        if manifest.get("revision") != REVISION:
            raise RuntimeError("OCR V8 sealed fixture revision changed")
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256_bytes(payload) != case["image_sha256"]:
                raise RuntimeError("OCR V8 sealed fixture image checksum changed")
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
        raise RuntimeError("OCR V8 sealed fixture fingerprint changed")
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
