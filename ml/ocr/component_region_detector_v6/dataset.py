# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes and exact deterministic proposal composition for OCR V6."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Callable, Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file

from .protocol import (
    CROP_HEIGHT,
    CROP_HORIZONTAL_PADDING_PIXELS,
    CROP_VERTICAL_PADDING_RATIO,
    CROP_WIDTH,
    ENCODED_WIDTH,
    GEOMETRY_FEATURE_COUNT,
    MAXIMUM_COMPONENT_HEIGHT_RATIO,
    MAXIMUM_COMPONENT_WIDTH_RATIO,
    MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO,
    MINIMUM_COMPONENT_AREA,
    MINIMUM_VERTICAL_OVERLAP_RATIO,
    PROPOSAL_THRESHOLD_MAXIMUM,
    PROPOSAL_THRESHOLD_MEAN_RATIO,
    PROPOSAL_THRESHOLD_MINIMUM,
    REVISION,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    SEED,
    TRUTH_MATCH_IOU_MINIMUM,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]
_WORDS = ("Baseline", "Treatment", "Followup", "Observer", "Measure", "Session", "Response", "Phase", "Probe", "Level")
_NUMBERS = ("0", "17", "-8", "3.5", "42%", "100", "-0.7", "9.25")


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class Component:
    left: int
    top: int
    right: int
    bottom: int
    area: int
    count: int = 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def box(self) -> Box:
        return Box(self.left, self.top, self.right + 1, self.bottom + 1)

    def merge(self, other: "Component") -> "Component":
        return Component(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
            self.area + other.area,
            self.count + other.count,
        )


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
    return ImageFont.truetype(str(REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]), size=size)


def _draw_text_mask(size: tuple[int, int], text: str, position: tuple[int, int], font: ImageFont.FreeTypeFont) -> tuple[Image.Image, Box]:
    mask = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.text(position, text, fill=255, font=font)
    bounds = mask.getbbox()
    if bounds is None:
        raise RuntimeError("Rendered OCR V6 text has no foreground")
    return mask, Box(*bounds)


def _render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(246, 256))
    ink = int(rng.integers(8, 66))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.uint8)
    ramp = np.linspace(0, int(rng.integers(0, 8)), SCENE_WIDTH, dtype=np.float32)
    base = np.clip(base.astype(np.float32) - ramp[None, :], 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="L")
    draw = ImageDraw.Draw(image)

    # Graph structures are deliberately separated from label rows. They still
    # create compact hard-negative proposals after long axes are filtered.
    draw.line((105, 44, 105, 205), fill=ink, width=2)
    draw.line((105, 205, 410, 205), fill=ink, width=2)
    for x in range(135, 391, 32):
        draw.line((x, 201, x, 209), fill=ink, width=2)
    draw.line((265, 48, 265, 202), fill=ink, width=2)
    points = [(130 + step * 34, 175 - int(38 * np.sin((step + index % 4) * 0.7))) for step in range(8)]
    draw.line(points, fill=ink, width=2)
    for step, (x, y) in enumerate(points):
        radius = 4 + ((index + step) % 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, fill=ink if step % 2 else None, width=2)
    draw.rectangle((424, 139, 492, 179), outline=ink, width=2)
    draw.line((164, 54, 225, 54), fill=ink, width=2)
    draw.line((164, 54, 164, 72), fill=ink, width=2)
    draw.line((225, 54, 225, 72), fill=ink, width=2)
    draw.line((318, 88, 378, 88), fill=ink, width=2)
    draw.line((378, 88, 365, 80), fill=ink, width=2)
    draw.line((378, 88, 365, 96), fill=ink, width=2)

    labels = (
        _NUMBERS[(index * 3 + 1) % len(_NUMBERS)],
        _WORDS[(index * 7 + 2) % len(_WORDS)],
        _NUMBERS[(index * 5 + 4) % len(_NUMBERS)],
    )
    if split == "validation":
        positions = ((12, 7), (354, 7), (14, 224))
    elif split == "sealed_public":
        positions = ((344, 7), (12, 7), (362, 224))
    else:
        positions = ((12 + int(rng.integers(0, 36)), 7), (352 + int(rng.integers(0, 32)), 7), (14 + int(rng.integers(0, 50)), 224))
    truths: list[Box] = []
    for label_index, (label, position) in enumerate(zip(labels, positions, strict=True)):
        font = _font(split, index + label_index, int(rng.integers(17, 22)))
        mask, bounds = _draw_text_mask(image.size, label, position, font)
        image.paste(ink, mask=mask)
        truths.append(bounds)

    if split == "train" and index % 3 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
    elif split == "validation" and index % 2 == 0:
        reduced = image.resize((448, 224), resample=Image.Resampling.BILINEAR)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR)
        row = 118 + index % 9
        pixels = np.asarray(image, dtype=np.uint8).copy()
        pixels[row, :] = np.minimum(pixels[row, :], 235)
        image = Image.fromarray(pixels, mode="L")
    elif split == "sealed_public":
        pixels = np.asarray(image, dtype=np.uint8).copy()
        pixels = ((pixels.astype(np.uint16) // 8) * 8).clip(0, 255).astype(np.uint8)
        foreground = np.argwhere(pixels < 170)
        if len(foreground) and index % 2 == 0:
            for selected in rng.choice(len(foreground), size=min(4, len(foreground)), replace=False):
                y, x = foreground[int(selected)]
                pixels[int(y), int(x)] = 246
        image = Image.fromarray(pixels, mode="L")
    raster = np.asarray(image, dtype=np.uint8).copy()
    for _ in range(int(rng.integers(0, 12))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(185, 238))
    return SceneSample(
        f"component-region-v6-{split}-{index:05d}",
        split,
        registration.renderer_family,
        registration.degradation_family,
        raster,
        tuple(truths),
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(_render_scene(split, index) for index in range(registration.scene_count))


def connected_components(gray: np.ndarray) -> tuple[Component, ...]:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("OCR V6 proposal source must be a uint8 Gray8 raster")
    height, width = gray.shape
    threshold = int(np.clip(round(float(gray.mean()) * PROPOSAL_THRESHOLD_MEAN_RATIO), PROPOSAL_THRESHOLD_MINIMUM, PROPOSAL_THRESHOLD_MAXIMUM))
    foreground = gray <= threshold
    visited = np.zeros_like(foreground, dtype=np.bool_)
    result: list[Component] = []
    maximum_width = max(2.0, width * MAXIMUM_COMPONENT_WIDTH_RATIO)
    maximum_height = max(2.0, height * MAXIMUM_COMPONENT_HEIGHT_RATIO)
    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            visited[y, x] = True
            if not foreground[y, x]:
                continue
            queue = [(x, y)]
            cursor = 0
            left = right = x
            top = bottom = y
            area = 0
            while cursor < len(queue):
                current_x, current_y = queue[cursor]
                cursor += 1
                area += 1
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)
                for next_x, next_y in ((current_x - 1, current_y), (current_x + 1, current_y), (current_x, current_y - 1), (current_x, current_y + 1)):
                    if not (0 <= next_x < width and 0 <= next_y < height) or visited[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    if foreground[next_y, next_x]:
                        queue.append((next_x, next_y))
            component = Component(left, top, right, bottom, area)
            if area >= MINIMUM_COMPONENT_AREA and component.width <= maximum_width and component.height <= maximum_height:
                result.append(component)
    return tuple(result)


def group_lines(items: tuple[Component, ...]) -> tuple[Component, ...]:
    remaining = sorted(items, key=lambda item: (item.top, item.left, item.bottom, item.right))
    lines: list[Component] = []
    while remaining:
        line = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for index in range(len(remaining) - 1, -1, -1):
                candidate = remaining[index]
                overlap = max(0, min(line.bottom, candidate.bottom) - max(line.top, candidate.top) + 1)
                overlap_fraction = overlap / max(1, min(line.height, candidate.height))
                if candidate.left > line.right:
                    gap = candidate.left - line.right - 1
                elif line.left > candidate.right:
                    gap = line.left - candidate.right - 1
                else:
                    gap = 0
                maximum_gap = max(line.height, candidate.height) * MAXIMUM_HORIZONTAL_GAP_HEIGHT_RATIO
                if overlap_fraction >= MINIMUM_VERTICAL_OVERLAP_RATIO and gap <= maximum_gap:
                    line = line.merge(candidate)
                    remaining.pop(index)
                    changed = True
        lines.append(line)
    return tuple(sorted(lines, key=lambda item: (item.top, item.left, item.bottom, item.right)))


def proposals(gray: np.ndarray) -> tuple[Component, ...]:
    return group_lines(connected_components(gray))


def box_iou(left: Box, right: Box) -> float:
    x0 = max(left.left, right.left)
    y0 = max(left.top, right.top)
    x1 = min(left.right, right.right)
    y1 = min(left.bottom, right.bottom)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / max(1, union)


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


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    left = proposal.left - CROP_HORIZONTAL_PADDING_PIXELS
    top = proposal.top - max(1.0, proposal.height * CROP_VERTICAL_PADDING_RATIO)
    width = proposal.width + 2.0 * CROP_HORIZONTAL_PADDING_PIXELS
    height = proposal.height + 2.0 * max(1.0, proposal.height * CROP_VERTICAL_PADDING_RATIO)
    content_width = int(np.clip(np.ceil(CROP_HEIGHT * width / height), 1, CROP_WIDTH))
    output = np.full((CROP_HEIGHT, CROP_WIDTH), 255.0, dtype=np.float32)
    for target_y in range(CROP_HEIGHT):
        source_y = top + ((target_y + 0.5) / CROP_HEIGHT) * height - 0.5
        for target_x in range(content_width):
            source_x = left + ((target_x + 0.5) / content_width) * width - 0.5
            if 0.0 <= source_x < gray.shape[1] and 0.0 <= source_y < gray.shape[0]:
                output[target_y, target_x] = _sample_bilinear(gray, source_x, source_y)
    ink = 1.0 - np.rint(output).clip(0, 255).astype(np.float32) / 255.0
    encoded = np.zeros((1, CROP_HEIGHT, ENCODED_WIDTH), dtype=np.float32)
    encoded[0, :, :CROP_WIDTH] = ink
    threshold = float(np.clip(round(float(gray.mean()) * PROPOSAL_THRESHOLD_MEAN_RATIO), PROPOSAL_THRESHOLD_MINIMUM, PROPOSAL_THRESHOLD_MAXIMUM))
    geometry = np.asarray(
        (
            proposal.width / gray.shape[1],
            proposal.height / gray.shape[0],
            proposal.area / max(1, proposal.width * proposal.height),
            min(1.0, proposal.width / max(1.0, proposal.height * 8.0)),
            min(1.0, proposal.height / max(1.0, proposal.width * 4.0)),
            min(1.0, proposal.count / 12.0),
            threshold / 255.0,
            float(ink.mean()),
        ),
        dtype=np.float32,
    )
    if len(geometry) != GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V6 geometry feature contract changed")
    encoded[0, :, CROP_WIDTH:] = geometry[None, :]
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
        raise RuntimeError("OCR V6 split did not produce both proposal classes")
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
        raise RuntimeError(f"OCR V6 sealed archive already exists: {path}")
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
        "schema": "graphreader.ocr-component-region-sealed-fixtures.v1",
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
        "schema": "graphreader.ocr-component-region-private-manifest.v1",
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
                raise RuntimeError("OCR V6 sealed fixture image checksum changed")
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
        raise RuntimeError("OCR V6 sealed fixture fingerprint changed")
    return result
