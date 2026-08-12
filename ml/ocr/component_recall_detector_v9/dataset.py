# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh five-role procedural scenes for OCR detector recall V9."""

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
_PHASES = ("Baseline", "Treatment", "Maintenance", "Followup", "Intervention", "Phase A", "Phase B")
_ANNOTATIONS = ("Probe", "Review", "Change", "Transfer", "Prompt", "Criterion", "Level")
_LEGENDS = ("Target", "Series", "Rate", "Data", "Level", "Plan", "Probe")


def _rng(split: Split, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"component-recall-v9:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: Split, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    path = REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]
    return ImageFont.truetype(str(path), size=size)


def _text_mask(size: tuple[int, int], text: str, position: tuple[int, int], font: ImageFont.FreeTypeFont) -> tuple[Image.Image, Box]:
    mask = Image.new("L", size, color=0)
    ImageDraw.Draw(mask).text(position, text, fill=255, font=font)
    bounds = mask.getbbox()
    if bounds is None:
        raise RuntimeError("Rendered OCR V9 text has no foreground")
    return mask, Box(*bounds)


def _draw_structures(draw: ImageDraw.ImageDraw, rng: np.random.Generator, index: int, ink: int) -> None:
    left, top, right, bottom = 104, 48, 510, 256
    draw.line((left, top, left, bottom), fill=ink, width=1 + index % 2)
    draw.line((left, bottom, right, bottom), fill=ink, width=2)
    for x in range(left + 27, right - 2, 33 + index % 6):
        draw.line((x, bottom - 5, x, bottom + 5), fill=ink, width=2)
    for y in range(top + 24, bottom - 2, 31 + (index + 2) % 6):
        draw.line((left - 5, y, left + 5, y), fill=ink, width=2)
    for slot in range(2):
        x = left + 133 + slot * 132 + int(rng.integers(-11, 12))
        draw.line((x, top + 1, x, bottom - 1), fill=ink, width=1 + (index + slot) % 2)
    points = []
    for step in range(11):
        x = left + 17 + step * 36
        y = bottom - 44 - int(30 * np.sin((step + index % 8) * 0.61))
        points.append((x, y))
    draw.line(points, fill=ink, width=1 + index % 2)
    for step, (x, y) in enumerate(points):
        radius = 3 + ((step + index) % 4)
        kind = (step + index) % 4
        bounds = (x - radius, y - radius, x + radius, y + radius)
        if kind == 0:
            draw.ellipse(bounds, outline=ink, width=2)
        elif kind == 1:
            draw.ellipse(bounds, outline=ink, fill=ink)
        elif kind == 2:
            draw.rectangle(bounds, outline=ink, width=2)
        else:
            draw.polygon(((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), outline=ink)
    bracket_y = top + 10 + int(rng.integers(0, 21))
    bracket_left = left + 25 + int(rng.integers(0, 54))
    bracket_right = bracket_left + 68 + int(rng.integers(0, 34))
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 18), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 18), fill=ink, width=2)
    arrow_y = top + 72 + int(rng.integers(0, 55))
    draw.line((right - 116, arrow_y, right - 24, arrow_y), fill=ink, width=2)
    draw.line((right - 24, arrow_y, right - 38, arrow_y - 8), fill=ink, width=2)
    draw.line((right - 24, arrow_y, right - 38, arrow_y + 8), fill=ink, width=2)
    legend_left = 518 + int(rng.integers(-4, 5))
    legend_top = 198 + int(rng.integers(-5, 6))
    draw.rectangle((legend_left, legend_top, 635, 274), outline=ink, width=2)
    draw.line((legend_left + 8, legend_top + 18, legend_left + 44, legend_top + 18), fill=ink, width=2)
    draw.ellipse((legend_left + 22, legend_top + 14, legend_left + 30, legend_top + 22), outline=ink, width=2)
    draw.line((legend_left + 8, legend_top + 46, legend_left + 44, legend_top + 46), fill=ink, width=2)
    draw.rectangle((legend_left + 22, legend_top + 42, legend_left + 30, legend_top + 50), outline=ink, fill=ink)
    for slot, (base_x, base_y) in enumerate(((548, 52), (593, 91), (550, 138), (600, 169))):
        x = base_x + int(rng.integers(-5, 6))
        y = base_y + int(rng.integers(-4, 5))
        radius = 5 + ((index + slot) % 5)
        kind = (index + slot) % 5
        if kind == 0:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)
        elif kind == 1:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)
        elif kind == 2:
            draw.line((x - radius, y, x + radius, y), fill=ink, width=2)
            draw.line((x, y - radius, x, y + radius), fill=ink, width=2)
        elif kind == 3:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=ink, width=2)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=ink, width=2)
        else:
            draw.polygon(((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), outline=ink)


def _render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(247, 256))
    ink = int(rng.integers(8, 63))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(0.0, float(rng.integers(0, 7)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(0.0, float(rng.integers(0, 6)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    _draw_structures(draw, rng, index, ink)
    labels = (
        _NUMBERS[(index * 5 + 2) % len(_NUMBERS)],
        _NUMBERS[(index * 9 + 7) % len(_NUMBERS)],
        _PHASES[(index * 3 + 1) % len(_PHASES)],
        _ANNOTATIONS[(index * 5 + 3) % len(_ANNOTATIONS)],
        _LEGENDS[(index * 7 + 2) % len(_LEGENDS)],
    )
    y_slot = 68 + (index * 23) % 111
    x_slot = 164 + (index * 37) % 236
    phase_slot = 128 + (index * 41) % 196
    annotation_x = 350 + (index * 31) % 110
    annotation_y = 102 + (index * 19) % 52
    legend_y = 174 + (index * 7) % 11
    if split == "validation":
        y_slot = 73 + (index * 29) % 103
        x_slot = 176 + (index * 43) % 218
        phase_slot = 142 + (index * 47) % 179
        annotation_x = 363 + (index * 37) % 95
    elif split == "sealed_public":
        y_slot = 79 + (index * 31) % 94
        x_slot = 184 + (index * 53) % 208
        phase_slot = 151 + (index * 59) % 166
        annotation_x = 370 + (index * 41) % 88
    draw.rectangle((326, annotation_y - 5, 638, annotation_y + 32), fill=background)
    draw.rectangle((569, 162, 639, 221), fill=background)
    positions = ((17, y_slot), (x_slot, 282), (phase_slot, 7), (annotation_x, annotation_y), (581, legend_y))
    truths: list[Box] = []
    for label_index, (label, position) in enumerate(zip(labels, positions, strict=True)):
        font = _font(split, index + label_index, int(rng.integers(16, 22)))
        mask, bounds = _text_mask(image.size, label, position, font)
        image.paste(ink, mask=mask)
        truths.append(bounds)
    if split == "train":
        if index % 4 == 0:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        pixels = np.asarray(image, dtype=np.float32)
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(0.94, 1.08)) + 128.0, 0, 255)
        row = 42 + (index * 17) % 234
        pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2)] = np.maximum(
            pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2)], 168
        )
        image = Image.fromarray(np.rint(pixels).astype(np.uint8), mode="L")
    elif split == "validation":
        reduced = image.resize((624, 312), resample=Image.Resampling.BOX)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.uint8).copy()
        row = 35 + (index * 37) % 248
        pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2)] = np.maximum(
            pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2)], 170
        )
        image = Image.fromarray(pixels, mode="L")
    else:
        pixels = np.asarray(image, dtype=np.float32)
        pixels = 255.0 * np.power(pixels / 255.0, float(rng.uniform(0.93, 1.07)))
        pixels = (np.rint(pixels).astype(np.uint16) // 4 * 4).clip(0, 255).astype(np.uint8)
        foreground = np.argwhere(pixels < 176)
        if len(foreground):
            for selected in rng.choice(len(foreground), size=min(6, len(foreground)), replace=False):
                y, x = foreground[int(selected)]
                pixels[int(y), int(x)] = int(rng.integers(235, 251))
        image = Image.fromarray(pixels, mode="L")
    raster = np.asarray(image, dtype=np.uint8).copy()
    for _ in range(int(rng.integers(0, 8))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(187, 241))
    return SceneSample(
        f"component-recall-v9-{split}-{index:05d}", split, registration.renderer_family,
        registration.degradation_family, raster, tuple(truths)
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(_render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    return encode_v7_proposal(gray, proposal)


def proposal_labels(scene: SceneSample, items: tuple[Component, ...] | None = None) -> np.ndarray:
    candidates = proposals(scene.raster) if items is None else items
    return np.asarray([
        int(any(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for truth in scene.truths))
        for candidate in candidates
    ], dtype=np.int64)


def proposal_examples(scenes: tuple[SceneSample, ...]) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    for scene in scenes:
        candidates = proposals(scene.raster)
        scene_labels = proposal_labels(scene, candidates)
        selected = list(range(len(candidates)))
        if scene.split == "train":
            positive = [index for index, value in enumerate(scene_labels) if value == 1]
            negative = [index for index, value in enumerate(scene_labels) if value == 0]
            selected = sorted(positive + negative[:TRAINING_NEGATIVE_PROPOSAL_CAP_PER_SCENE])
        values.extend(encode_proposal(scene.raster, candidates[index]) for index in selected)
        labels.extend(int(scene_labels[index]) for index in selected)
    if not values or not any(labels) or all(labels):
        raise RuntimeError("OCR V9 split did not produce both proposal classes")
    return np.stack(values).astype(np.float32), np.asarray(labels, dtype=np.int64)


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(f"{truth.left},{truth.top},{truth.right},{truth.bottom}\n".encode())
    return digest.hexdigest()


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, int]:
    proposal_count = positive_count = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        proposal_count += len(candidates)
        for candidate in candidates:
            positive_count += int(any(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for truth in scene.truths))
        for truth in scene.truths:
            matches = sum(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for candidate in candidates)
            if matches != 1:
                raise RuntimeError(f"{scene.scene_id} truth {truth} has {matches} deterministic proposals")
    truth_count = sum(len(scene.truths) for scene in scenes)
    return {
        "scene_count": len(scenes), "truth_region_count": truth_count, "proposal_count": proposal_count,
        "positive_proposal_count": positive_count, "negative_proposal_count": proposal_count - positive_count,
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
        raise RuntimeError(f"OCR V9 sealed archive already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    images: list[tuple[str, bytes]] = []
    for scene in scenes:
        name = f"images/{scene.scene_id}.png"
        payload = _png_bytes(scene.raster)
        images.append((name, payload))
        cases.append({
            "scene_id": scene.scene_id, "image_path": name, "image_sha256": sha256_bytes(payload),
            "renderer_family": scene.renderer_family, "degradation_family": scene.degradation_family,
            "truths": [[truth.left, truth.top, truth.right, truth.bottom] for truth in scene.truths],
        })
    manifest = {
        "schema": "graphreader.ocr-component-recall-fixtures.v1", "revision": REVISION,
        "split": "sealed_public", "scene_count": len(scenes),
        "split_fingerprint": split_fingerprint(scenes), "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "predecessor_fixture_bytes_reused": False, "cases": cases,
    }
    with ZipFile(path, "x") as archive:
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
        for name, payload in sorted(images):
            _zip_write(archive, name, payload)
    return {
        "schema": "graphreader.ocr-component-recall-private-manifest.v1", "revision": REVISION,
        "scene_count": len(scenes), "split_fingerprint": split_fingerprint(scenes),
        "fixture_archive_sha256": sha256_file(path), "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False, "generalization_label_included": False,
    }


def load_sealed_public_archive(path: Path) -> tuple[SceneSample, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("revision") != REVISION:
            raise RuntimeError("OCR V9 sealed fixture revision changed")
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256_bytes(payload) != case["image_sha256"]:
                raise RuntimeError("OCR V9 sealed fixture image checksum changed")
            with Image.open(BytesIO(payload)) as image:
                raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            scenes.append(SceneSample(
                case["scene_id"], "sealed_public", case["renderer_family"], case["degradation_family"],
                raster, tuple(Box(*bounds) for bounds in case["truths"])
            ))
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR V9 sealed fixture fingerprint changed")
    return result


__all__ = [
    "Box", "Component", "SceneSample", "box_iou", "build_split", "encode_proposal",
    "load_sealed_public_archive", "proposal_examples", "proposal_labels", "proposal_summary",
    "proposals", "save_sealed_public_archive", "split_fingerprint",
]
