# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural graph scenes for relational OCR V21."""

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

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals
from ml.ocr.component_region_detector_v6.dataset import Box, Component
from ml.ocr.layout_conditioned_proposal_role_v15.dataset import encode_proposal as encode_v15_proposal
from .protocol import (
    ENCODED_WIDTH,
    INPUT_CHANNELS,
    ROLE_ORDER,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    TRUTH_MATCH_IOU_MINIMUM,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]

_LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "YTick": ("0", "20", "40", "60", "80", "100", "-5", "2.5", "75%"),
        "XTick": ("1", "3", "6", "9", "12", "18", "24", "30"),
        "AxisTitle": ("Session", "Trial", "Day", "Visit"),
        "PhaseHeading": ("Baseline", "Treatment", "Practice", "Maintenance"),
        "LegendText": ("Target", "Probe", "Rate", "Level"),
        "Participant": ("Alex", "Morgan", "Taylor", "CaseA"),
        "Annotation": ("Prompted", "Revised", "Noted", "Shifted"),
        "Other": ("Weekly", "Outcome", "Summary", "Measure"),
    },
    "validation": {
        "YTick": ("15", "35", "50", "70", "90", "-2", "3.5", "65%"),
        "XTick": ("2", "5", "8", "11", "16", "22", "28", "36"),
        "AxisTitle": ("Observation", "Attempt", "Sample", "Interval"),
        "PhaseHeading": ("Orientation", "Intervention", "Fluency", "Followup"),
        "LegendText": ("Measured", "Average", "Band", "Series"),
        "Participant": ("Jordan", "Riley", "StudentB", "LearnerC"),
        "Annotation": ("Scheduled", "Checked", "Updated", "Moved"),
        "Other": ("Daily", "Record", "Figure", "Result"),
    },
    "sealed_public": {
        "YTick": ("105", "25", "45", "55", "85", "-10", "4.5", "90%"),
        "XTick": ("4", "7", "10", "14", "19", "26", "32", "42"),
        "AxisTitle": ("Sessions", "Checks", "Cycles", "Probes"),
        "PhaseHeading": ("Acquisition", "Coaching", "Retention", "Transfer"),
        "LegendText": ("Observed", "Goal", "Range", "Index"),
        "Participant": ("Avery", "Quinn", "ClientD", "ObserverE"),
        "Annotation": ("Criterion", "Setting", "Reviewed", "Adjusted"),
        "Other": ("Monthly", "Overview", "Panel", "Report"),
    },
}


@dataclass(frozen=True)
class RoleTruth:
    box: Box
    role: str
    text: str


@dataclass(frozen=True)
class SceneSample:
    scene_id: str
    split: Split
    renderer_family: str
    degradation_family: str
    raster: np.ndarray
    plot: Box
    truths: tuple[RoleTruth, ...]


def _rng(split: Split, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"relational-scene-proposal-role-v21:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: Split, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    path = REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]
    return ImageFont.truetype(str(path), size=size)


def _right_aligned_text_x(
    split: Split,
    index: int,
    label_index: int,
    text: str,
    font_size: int,
    right_edge: int,
) -> int:
    """Place axis text with a fixed gap from plot ink before degradation."""
    font = _font(split, index * len(ROLE_ORDER) + label_index, font_size)
    bounds = font.getbbox(text, anchor="lt")
    width = bounds[2] - bounds[0]
    return max(3, right_edge - width)


def _marker(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, ink: int, style: int) -> None:
    bounds = (x - radius, y - radius, x + radius, y + radius)
    if style % 4 == 0:
        draw.ellipse(bounds, fill=ink)
    elif style % 4 == 1:
        draw.ellipse(bounds, outline=ink, width=2)
    elif style % 4 == 2:
        draw.rectangle(bounds, outline=ink, width=2)
    else:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), outline=ink)


def _draw_structures(
    image: Image.Image,
    rng: np.random.Generator,
    plot: Box,
    ink: int,
    index: int,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = plot.left, plot.top, plot.right, plot.bottom
    draw.line((left, top, left, bottom), fill=ink, width=2)
    draw.line((left, bottom, right, bottom), fill=ink, width=2)
    x_ticks = 8 + index % 5
    y_ticks = 5 + index % 4
    for tick in range(x_ticks + 1):
        x = round(left + tick * (right - left) / x_ticks)
        draw.line((x, bottom - 3, x, bottom + 7), fill=ink, width=1 + tick % 2)
    for tick in range(y_ticks + 1):
        y = round(bottom - tick * (bottom - top) / y_ticks)
        draw.line((left - 7, y, left + 3, y), fill=ink, width=1 + tick % 2)
        if tick not in {0, y_ticks} and (tick + index) % 2 == 0:
            draw.line((left + 1, y, right, y), fill=max(ink, 178), width=1)

    divider_count = 1 + index % 3
    for divider in range(divider_count):
        x = round(left + (divider + 1) * (right - left) / (divider_count + 1))
        draw.line((x, top - 7, x, bottom), fill=ink, width=1 + divider % 2)

    for series in range(2 + index % 2):
        point_count = 8 + (index + series) % 7
        points: list[tuple[int, int]] = []
        for point in range(point_count):
            x = round(left + 14 + point * (right - left - 28) / max(1, point_count - 1))
            wave = np.sin((point + series * 1.4 + index * 0.17) * 0.81)
            y = round(top + 38 + series * 39 + (wave + 1.0) * 28 + rng.integers(-6, 7))
            points.append((x, min(bottom - 10, max(top + 10, y))))
        draw.line(points, fill=ink, width=1 + (index + series) % 2)
        for point_index, (x, y) in enumerate(points):
            _marker(draw, x, y, 3 + (point_index + series) % 3, ink, index + point_index + series)

    legend_left = right + 10
    legend_top = top + 35 + index % 17
    legend_right = min(SCENE_WIDTH - 7, legend_left + 135)
    legend_bottom = min(bottom - 4, legend_top + 73)
    draw.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=ink, width=1)
    _marker(draw, legend_left + 14, legend_top + 17, 4, ink, index)
    draw.line((legend_left + 8, legend_top + 47, legend_left + 29, legend_top + 47), fill=ink, width=2)

    bracket_y = top + 13 + index % 19
    bracket_left = left + 29 + (index * 17) % max(31, right - left - 151)
    bracket_right = min(right - 12, bracket_left + 43 + index % 67)
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 12), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 12), fill=ink, width=2)

    arrow_y = top + 71 + (index * 13) % max(31, bottom - top - 91)
    arrow_right = right - 19
    arrow_left = max(left + 93, arrow_right - 63 - index % 23)
    draw.line((arrow_left, arrow_y, arrow_right, arrow_y), fill=ink, width=2)
    draw.polygon(((arrow_right, arrow_y), (arrow_right - 11, arrow_y - 7), (arrow_right - 8, arrow_y), (arrow_right - 11, arrow_y + 7)), fill=ink)

    for decoy in range(3 + index % 4):
        x = left + 21 + (index * 31 + decoy * 67) % max(29, right - left - 42)
        y = top + 27 + (index * 43 + decoy * 47) % max(31, bottom - top - 54)
        radius = 3 + (index + decoy) % 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=1)
        draw.line((x - radius - 3, y, x + radius + 3, y), fill=ink, width=1)


def _draw_text(
    image: Image.Image,
    split: Split,
    index: int,
    background: int,
    ink: int,
    labels: tuple[tuple[str, str, tuple[int, int], int], ...],
) -> tuple[RoleTruth, ...]:
    truths: list[RoleTruth] = []
    for label_index, (text, role, position, font_size) in enumerate(labels):
        font = _font(split, index * len(ROLE_ORDER) + label_index, font_size)
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.text(position, text, font=font, fill=255, anchor="lt")
        bounds = mask.getbbox()
        if bounds is None:
            raise RuntimeError("OCR V21 text renderer produced an empty mask")
        box = Box(*bounds)
        draw = ImageDraw.Draw(image)
        horizontal = 112 if role == "Annotation" else 96 if role == "LegendText" else 5 if role in {"YTick", "XTick"} else 8
        vertical = 11 if role == "Annotation" else 10 if role == "LegendText" else 6
        draw.rectangle((
            max(0, box.left - horizontal),
            max(0, box.top - vertical),
            min(SCENE_WIDTH - 1, box.right + horizontal),
            min(SCENE_HEIGHT - 1, box.bottom + vertical),
        ), fill=background)
        image.paste(ink, mask=mask)
        truths.append(RoleTruth(box, role, text))
    return tuple(truths)


def _degrade(image: Image.Image, split: Split, index: int, rng: np.random.Generator) -> np.ndarray:
    if split == "train":
        if index % 3:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.04, 0.22))))
        width = 626 + (index * 7) % 31
        image = image.resize((width, SCENE_HEIGHT), Image.Resampling.BICUBIC).resize(
            (SCENE_WIDTH, SCENE_HEIGHT), Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
        column_bias = np.sin(np.linspace(0.0, np.pi * 2.0, SCENE_WIDTH, dtype=np.float32)) * float(index % 4)
        pixels += column_bias[None, :]
    elif split == "validation":
        width = 646 + (index * 11) % 29
        height = 312 + (index * 5) % 17
        image = image.resize((width, height), Image.Resampling.LANCZOS).resize(
            (SCENE_WIDTH, SCENE_HEIGHT), Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
        corner = np.linspace(0.0, float(2 + index % 5), SCENE_WIDTH, dtype=np.float32)
        pixels -= corner[None, :]
        row = 31 + (index * 37) % (SCENE_HEIGHT - 62)
        pixels[row:row + 1] = np.clip(pixels[row:row + 1] - 12.0, 0.0, 255.0)
    else:
        width = 618 + (index * 13) % 37
        height = 304 + (index * 7) % 23
        image = image.resize((width, height), Image.Resampling.BOX).resize(
            (SCENE_WIDTH, SCENE_HEIGHT), Image.Resampling.BICUBIC,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
        wave = np.sin(np.linspace(0.0, np.pi, SCENE_HEIGHT, dtype=np.float32)) * float(1 + index % 4)
        pixels -= wave[:, None]
        pixels = np.rint(pixels).astype(np.int16) // 3 * 3
        for speckle in range(index % 7):
            y = (19 + index * 43 + speckle * 71) % SCENE_HEIGHT
            x = (23 + index * 59 + speckle * 83) % SCENE_WIDTH
            pixels[y, x] = 190 + (index + speckle) % 47
    return np.rint(pixels).clip(0, 255).astype(np.uint8)


def render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    if not 0 <= index < registration.scene_count:
        raise IndexError(f"OCR V21 {split} scene index out of range: {index}")
    rng = _rng(split, index)
    background = int(rng.integers(248, 256))
    ink = int(rng.integers(6, 46))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(0.0, float(rng.integers(0, 7)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")

    if split == "train":
        left, top = int(rng.integers(128, 145)), int(rng.integers(47, 69))
        right, bottom = int(rng.integers(462, 492)), int(rng.integers(232, 256))
    elif split == "validation":
        left, top = 129 + (index * 7) % 17, 49 + (index * 11) % 19
        right, bottom = 465 + (index * 13) % 26, 234 + (index * 17) % 21
    else:
        left, top = 127 + (index * 13) % 19, 45 + (index * 17) % 23
        right, bottom = 458 + (index * 19) % 34, 229 + (index * 23) % 27
    plot = Box(left, top, right, bottom)
    _draw_structures(image, rng, plot, ink, index)

    values = _LABELS[split]
    label = lambda role, stride: values[role][(index * stride + stride) % len(values[role])]
    numeric = 21 + index % 3
    small = 17 + index % 3
    medium = 15 + (index * 3) % 4
    large = 17 + (index * 5) % 4
    y_tick = label("YTick", 5)
    labels = (
        (
            y_tick,
            "YTick",
            (
                _right_aligned_text_x(split, index, 0, y_tick, numeric, left - 76),
                top + 37 + index % max(29, bottom - top - 71),
            ),
            numeric,
        ),
        (label("XTick", 7), "XTick", (left + 27 + (index * 11) % max(31, (right - left) // 3), bottom + 13), numeric),
        (label("AxisTitle", 3), "AxisTitle", ((left + right) // 2 - 34, min(299, bottom + 31)), medium),
        (label("PhaseHeading", 3), "PhaseHeading", (left + 71 + (index * 17) % max(29, right - left - 213), max(3, top - 35)), large),
        (label("LegendText", 3), "LegendText", (right + 44, top + 43 + index % 15), small),
        (label("Participant", 3), "Participant", (right + 6, min(298, bottom + 18)), medium),
        (label("Annotation", 3), "Annotation", (right - 161 - index % 31, top + 82 + index % max(29, bottom - top - 119)), small),
        (label("Other", 3), "Other", (7 + index % 31, 4 + (index * 7) % 17), medium),
    )
    truths = _draw_text(image, split, index, background, ink, labels)
    raster = _degrade(image, split, index, rng)
    return SceneSample(
        f"relational-scene-proposal-role-v21-{split}-{index:05d}",
        split,
        registration.renderer_families[index % len(registration.renderer_families)],
        registration.degradation_families[index % len(registration.degradation_families)],
        raster,
        plot,
        truths,
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registration.scene_count))


def label_vocabulary(split: Split) -> dict[str, tuple[str, ...]]:
    """Return an immutable copy of the preregistered split vocabulary."""
    split_registration(split)
    return {role: tuple(values) for role, values in _LABELS[split].items()}


def encode_scene(scene: SceneSample) -> tuple[np.ndarray, tuple[Component, ...], np.ndarray, np.ndarray]:
    candidates = proposals(scene.raster)
    encoded = np.stack([encode_v15_proposal(scene.raster, candidate, scene.plot) for candidate in candidates])
    if encoded.ndim != 4 or encoded.shape[1:] != (INPUT_CHANNELS, 32, ENCODED_WIDTH):
        raise RuntimeError("OCR V21 production proposal tensor contract changed")
    proposal_labels, role_labels = proposal_targets(scene, candidates)
    return encoded.astype(np.float32), candidates, proposal_labels, role_labels


def proposal_targets(
    scene: SceneSample,
    candidates: tuple[Component, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    items = proposals(scene.raster) if candidates is None else candidates
    proposal_labels: list[int] = []
    role_labels: list[int] = []
    for candidate in items:
        matches = [
            (truth_index, box_iou(candidate.box, truth.box))
            for truth_index, truth in enumerate(scene.truths)
            if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
        ]
        if not matches:
            proposal_labels.append(0)
            role_labels.append(-1)
            continue
        best = max(matches, key=lambda item: item[1])[0]
        proposal_labels.append(1)
        role_labels.append(ROLE_ORDER.index(scene.truths[best].role))
    return np.asarray(proposal_labels, dtype=np.int64), np.asarray(role_labels, dtype=np.int64)


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, object]:
    proposal_count = positive_count = negative_count = 0
    role_counts = {role: 0 for role in ROLE_ORDER}
    stream = sha256()
    for scene in scenes:
        candidates = proposals(scene.raster)
        labels, roles = proposal_targets(scene, candidates)
        for truth_index, truth in enumerate(scene.truths):
            matches = [
                candidate_index
                for candidate_index, candidate in enumerate(candidates)
                if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"{scene.scene_id} truth {truth_index} ({truth.role}) has {len(matches)} production proposals"
                )
            role_counts[truth.role] += 1
        proposal_count += len(candidates)
        positive_count += int((labels == 1).sum())
        negative_count += int((labels == 0).sum())
        stream.update(scene.scene_id.encode())
        stream.update(scene.raster.tobytes(order="C"))
        stream.update(labels.tobytes(order="C"))
        stream.update(roles.tobytes(order="C"))
    if any(value == 0 for value in role_counts.values()) or negative_count == 0:
        raise RuntimeError("OCR V21 split lacks required roles or structure negatives")
    return {
        "scene_count": len(scenes),
        "proposal_count": proposal_count,
        "positive_proposal_count": positive_count,
        "negative_proposal_count": negative_count,
        "role_truth_counts": role_counts,
        "proposal_label_stream_sha256": stream.hexdigest(),
        "exactly_one_production_proposal_per_truth": True,
    }


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(
                f"{truth.box.left},{truth.box.top},{truth.box.right},{truth.box.bottom}|{truth.role}|{truth.text}\n".encode()
            )
    return digest.hexdigest()


def save_archive(scenes: tuple[SceneSample, ...], split: Split, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"OCR V21 identity already exists and cannot be regenerated: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    image_bytes: list[tuple[str, bytes]] = []
    for scene in scenes:
        stream = BytesIO()
        Image.fromarray(scene.raster, mode="L").save(stream, format="PNG", optimize=False, compress_level=9)
        relative_path = f"images/{scene.scene_id}.png"
        payload = stream.getvalue()
        image_bytes.append((relative_path, payload))
        cases.append({
            "scene_id": scene.scene_id,
            "renderer_family": scene.renderer_family,
            "degradation_family": scene.degradation_family,
            "source_path": relative_path,
            "source_sha256": sha256(payload).hexdigest(),
            "plot": [scene.plot.left, scene.plot.top, scene.plot.right, scene.plot.bottom],
            "truths": [
                {
                    "box": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom],
                    "role": truth.role,
                    "text": truth.text,
                }
                for truth in scene.truths
            ],
        })
    manifest: dict[str, object] = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-fixtures.v1",
        "revision": "graph-text-relational-scene-proposal-role-v21",
        "split": split,
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "required_roles": list(ROLE_ORDER),
        "cases": cases,
        "synthetic_only": True,
        "chandler_included": False,
        "generalization_label_included": False,
        "private_or_article_images": False,
        "production_approval": False,
        "release_eligible": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    timestamp = (2026, 8, 15, 0, 0, 0)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in [("manifest.json", manifest_bytes), *image_bytes]:
            info = ZipInfo(name, timestamp)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return {
        "archive_path": output_path.relative_to(REPO_ROOT).as_posix(),
        "archive_sha256": sha256_file(output_path),
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "split_fingerprint": split_fingerprint(scenes),
        "proposal_summary": proposal_summary(scenes),
    }


def load_archive(path: Path) -> tuple[SceneSample, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        split = manifest["split"]
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["source_path"])
            if sha256(payload).hexdigest() != case["source_sha256"]:
                raise RuntimeError(f"OCR V21 fixture hash mismatch: {case['scene_id']}")
            raster = np.asarray(Image.open(BytesIO(payload)).convert("L"), dtype=np.uint8).copy()
            scenes.append(SceneSample(
                case["scene_id"],
                split,
                case["renderer_family"],
                case["degradation_family"],
                raster,
                Box(*case["plot"]),
                tuple(RoleTruth(Box(*truth["box"]), truth["role"], truth["text"]) for truth in case["truths"]),
            ))
    return tuple(scenes)


__all__ = [
    "RoleTruth",
    "SceneSample",
    "build_split",
    "encode_scene",
    "label_vocabulary",
    "load_archive",
    "proposal_summary",
    "proposal_targets",
    "render_scene",
    "save_archive",
    "split_fingerprint",
]
