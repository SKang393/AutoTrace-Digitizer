# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural graph scenes for OCR production-composition V1."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals

from .protocol import (
    PLOT_BOUNDS,
    REVISION,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    TRUTH_MATCH_IOU_MINIMUM,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
NUMERIC_LABELS = (
    ("0", "0"),
    ("O", "0"),
    ("5", "5"),
    ("10", "10"),
    ("2O", "20"),
    ("25", "25"),
    ("4O", "40"),
    ("50", "50"),
    ("75", "75"),
    ("8O", "80"),
    ("100", "100"),
    ("-2", "-2"),
    ("2.5", "2.5"),
    ("33%", "33%"),
    ("l0", "10"),
    ("l.5", "1.5"),
)
PHASE_WORDS = ("Baseline", "Treatment", "Maintenance", "Followup", "Intervention", "Phase A", "Phase B")
ANNOTATION_WORDS = ("Probe", "Review", "Change", "Transfer", "Prompt", "Criterion", "Level")
LEGEND_WORDS = ("Target", "Series", "Rate", "Data", "Level", "Plan", "Probe")


@dataclass(frozen=True)
class TextTruth:
    display_text: str
    truth_text: str
    role: str
    box: Box


@dataclass(frozen=True)
class CompositionScene:
    scene_id: str
    split: str
    renderer_family: str
    degradation_family: str
    raster: np.ndarray
    truths: tuple[TextTruth, ...]


def _rng(split: str, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"composition-v1:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: str, index: int, size: int) -> ImageFont.FreeTypeFont:
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
        raise RuntimeError("Rendered OCR composition text has no foreground")
    return mask, Box(*bounds)


def _draw_structures(draw: ImageDraw.ImageDraw, rng: np.random.Generator, index: int, ink: int) -> None:
    left, top, right, bottom = PLOT_BOUNDS
    draw.line((left, top, left, bottom), fill=ink, width=1 + index % 2)
    draw.line((left, bottom, right, bottom), fill=ink, width=2)
    for x in range(left + 28, right - 3, 34 + index % 5):
        draw.line((x, bottom - 5, x, bottom + 5), fill=ink, width=2)
    for y in range(top + 26, bottom - 3, 32 + (index + 1) % 5):
        draw.line((left - 5, y, left + 5, y), fill=ink, width=2)
    for slot in range(2):
        divider_x = left + 138 + slot * 126 + int(rng.integers(-8, 9))
        draw.line((divider_x, top + 1, divider_x, bottom - 1), fill=ink, width=1 + (index + slot) % 2)

    points: list[tuple[int, int]] = []
    for step in range(11):
        x = left + 17 + step * 36
        y = bottom - 43 - int(31 * np.sin((step + index % 7) * 0.59))
        points.append((x, y))
    draw.line(points, fill=ink, width=1 + index % 2)
    for step, (x, y) in enumerate(points):
        radius = 3 + ((step + index) % 4)
        kind = (step + index) % 4
        if kind == 0:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)
        elif kind == 1:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, fill=ink)
        elif kind == 2:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)
        else:
            draw.polygon(((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), outline=ink)

    bracket_y = top + 11 + int(rng.integers(0, 18))
    bracket_left = left + 30 + int(rng.integers(0, 52))
    bracket_right = bracket_left + 70 + int(rng.integers(0, 30))
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 17), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 17), fill=ink, width=2)

    arrow_y = top + 74 + int(rng.integers(0, 48))
    arrow_left = right - 114
    arrow_right = right - 25
    draw.line((arrow_left, arrow_y, arrow_right, arrow_y), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 13, arrow_y - 8), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 13, arrow_y + 8), fill=ink, width=2)

    legend_left = 518 + int(rng.integers(-3, 4))
    legend_top = 198 + int(rng.integers(-4, 5))
    draw.rectangle((legend_left, legend_top, 634, 272), outline=ink, width=2)
    draw.line((legend_left + 9, legend_top + 18, legend_left + 42, legend_top + 18), fill=ink, width=2)
    draw.ellipse((legend_left + 21, legend_top + 14, legend_left + 29, legend_top + 22), outline=ink, width=2)
    draw.line((legend_left + 9, legend_top + 45, legend_left + 42, legend_top + 45), fill=ink, width=2)
    draw.rectangle((legend_left + 21, legend_top + 41, legend_left + 29, legend_top + 49), outline=ink, fill=ink)

    for slot, (x, y) in enumerate(((550, 54), (594, 92), (550, 140), (600, 171))):
        x += int(rng.integers(-4, 5))
        y += int(rng.integers(-4, 5))
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


def _render_scene(split: str, index: int) -> CompositionScene:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(247, 256))
    ink = int(rng.integers(8, 62))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(0.0, float(rng.integers(0, 7)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    _draw_structures(draw, rng, index, ink)

    y_display, y_truth = NUMERIC_LABELS[(index * 5 + 2) % len(NUMERIC_LABELS)]
    x_display, x_truth = NUMERIC_LABELS[(index * 9 + 7) % len(NUMERIC_LABELS)]
    phase = PHASE_WORDS[(index * 3 + 1) % len(PHASE_WORDS)]
    annotation = ANNOTATION_WORDS[(index * 5 + 3) % len(ANNOTATION_WORDS)]
    legend = LEGEND_WORDS[(index * 7 + 2) % len(LEGEND_WORDS)]
    y_slot = 70 + (index * 19) % 108
    x_slot = 170 + (index * 31) % 232
    phase_slot = 134 + (index * 37) % 190
    annotation_x = 356 + (index * 29) % 105
    annotation_y = 104 + (index * 17) % 43
    legend_y = 176 + (index * 5) % 8
    if split == "sealed_public":
        y_slot = 81 + (index * 29) % 87
        x_slot = 186 + (index * 47) % 207
        phase_slot = 151 + (index * 53) % 164
        annotation_x = 366 + (index * 37) % 91

    draw.rectangle((330, annotation_y - 4, 635, annotation_y + 31), fill=background)
    draw.rectangle((570, 164, 639, 220), fill=background)
    labels = (
        (y_display, y_truth, "y_tick", (17, y_slot)),
        (x_display, x_truth, "x_tick", (x_slot, 282)),
        (phase, phase, "phase_heading", (phase_slot, 7)),
        (annotation, annotation, "annotation", (annotation_x, annotation_y)),
        (legend, legend, "legend_text", (582, legend_y)),
    )
    truths: list[TextTruth] = []
    for label_index, (display, truth, role, position) in enumerate(labels):
        font = _font(split, index + label_index, int(rng.integers(16, 22)))
        mask, bounds = _text_mask(image.size, display, position, font)
        image.paste(ink, mask=mask)
        truths.append(TextTruth(display, truth, role, bounds))

    if split == "validation":
        reduced = image.resize((608, 304), resample=Image.Resampling.BICUBIC)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.uint8).copy()
        row = 39 + (index * 31) % 242
        pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :] = np.maximum(
            pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :], 174
        )
        image = Image.fromarray(pixels, mode="L")
    else:
        pixels = np.asarray(image, dtype=np.float32)
        pixels = 255.0 * np.power(pixels / 255.0, float(rng.uniform(0.945, 1.055)))
        pixels = (np.rint(pixels).astype(np.uint16) // 5 * 5).clip(0, 255).astype(np.uint8)
        foreground = np.argwhere(pixels < 175)
        if len(foreground) and index % 2 == 0:
            for selected in rng.choice(len(foreground), size=min(4, len(foreground)), replace=False):
                y, x = foreground[int(selected)]
                pixels[int(y), int(x)] = int(rng.integers(238, 252))
        image = Image.fromarray(pixels, mode="L")

    raster = np.asarray(image, dtype=np.uint8).copy()
    for _ in range(int(rng.integers(0, 8))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(188, 240))
    return CompositionScene(
        f"ocr-production-composition-v1-{split}-{index:05d}",
        split,
        registration.renderer_family,
        registration.degradation_family,
        raster,
        tuple(truths),
    )


def build_split(split: str) -> tuple[CompositionScene, ...]:
    registration = split_registration(split)
    return tuple(_render_scene(split, index) for index in range(registration.scene_count))


def split_fingerprint(scenes: tuple[CompositionScene, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(
                f"{truth.display_text}\0{truth.truth_text}\0{truth.role}\0"
                f"{truth.box.left},{truth.box.top},{truth.box.right},{truth.box.bottom}\n".encode()
            )
    return digest.hexdigest()


def proposal_summary(scenes: tuple[CompositionScene, ...]) -> dict[str, int]:
    proposal_count = positive_count = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        proposal_count += len(candidates)
        for candidate in candidates:
            positive_count += int(
                any(
                    box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
                    for truth in scene.truths
                )
            )
        for truth in scene.truths:
            matches = sum(
                box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
                for candidate in candidates
            )
            if matches != 1:
                raise RuntimeError(
                    f"{scene.scene_id} truth {truth.role}:{truth.truth_text} has {matches} deterministic proposals"
                )
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


def save_sealed_archive(scenes: tuple[CompositionScene, ...], path: Path) -> dict[str, object]:
    if path.exists():
        raise RuntimeError(f"OCR composition sealed archive already exists: {path}")
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
                "raster_sha256": sha256_bytes(scene.raster.tobytes(order="C")),
                "renderer_family": scene.renderer_family,
                "degradation_family": scene.degradation_family,
                "truths": [
                    {
                        "display_text": truth.display_text,
                        "truth_text": truth.truth_text,
                        "role": truth.role,
                        "bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom],
                    }
                    for truth in scene.truths
                ],
            }
        )
    manifest = {
        "schema": "graphreader.ocr-production-composition-fixtures.v1",
        "revision": REVISION,
        "split": scenes[0].split,
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "split_fingerprint": split_fingerprint(scenes),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_public_archive_reused": False,
        "cases": cases,
    }
    with ZipFile(path, "x") as archive:
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
        for name, payload in sorted(images):
            _zip_write(archive, name, payload)
    return {
        "schema": "graphreader.ocr-production-composition-private-manifest.v1",
        "revision": REVISION,
        "split": scenes[0].split,
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "split_fingerprint": split_fingerprint(scenes),
        "fixture_archive_sha256": sha256_file(path),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
    }


def load_sealed_archive(path: Path) -> tuple[CompositionScene, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("revision") != REVISION:
            raise RuntimeError("OCR composition fixture revision changed")
        scenes: list[CompositionScene] = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256_bytes(payload) != case["image_sha256"]:
                raise RuntimeError("OCR composition fixture PNG checksum changed")
            with Image.open(BytesIO(payload)) as image:
                raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            if sha256_bytes(raster.tobytes(order="C")) != case["raster_sha256"]:
                raise RuntimeError("OCR composition fixture raster checksum changed")
            scenes.append(
                CompositionScene(
                    case["scene_id"],
                    manifest["split"],
                    case["renderer_family"],
                    case["degradation_family"],
                    raster,
                    tuple(
                        TextTruth(
                            item["display_text"],
                            item["truth_text"],
                            item["role"],
                            Box(*item["bbox"]),
                        )
                        for item in case["truths"]
                    ),
                )
            )
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR composition fixture fingerprint changed")
    return result


__all__ = [
    "CompositionScene",
    "TextTruth",
    "build_split",
    "load_sealed_archive",
    "proposal_summary",
    "save_sealed_archive",
    "split_fingerprint",
]
