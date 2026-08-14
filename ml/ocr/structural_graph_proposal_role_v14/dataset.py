# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh multi-scale structural graph scenes for OCR V14."""

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
from ml.ocr.component_context_detector_v7.dataset import (
    box_iou,
    encode_proposal as encode_v7_proposal,
    proposals,
)
from ml.ocr.component_region_detector_v6.dataset import Box, Component
from .protocol import (
    ENCODED_WIDTH, GEOMETRY_FEATURE_COUNT, INPUT_CHANNELS, ROLE_ORDER,
    SCENE_HEIGHT, SCENE_WIDTH, TRUTH_MATCH_IOU_MINIMUM, split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]
_Y_TICKS = ("0", "6", "15", "32", "58", "90", "2.5", "-4", "70%")
_X_TICKS = ("2", "5", "8", "12", "17", "21", "30", "42")
_AXIS_TITLES = ("Visits", "Trials", "Days", "Blocks", "Checks")
_PHASES = ("Initial", "Coaching", "Practice", "Review", "Transfer")
_LEGENDS = ("Observed", "Planned", "Sample", "Median", "Range")
_PARTICIPANTS = ("Case B", "Pupil", "Reader", "Member", "Reporter")
_ANNOTATIONS = ("Schedule shift", "Review note", "Rule update", "Check event", "Level note")
_OTHERS = ("Morning", "Monthly", "Result", "Overview", "Frequency")


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
    truths: tuple[RoleTruth, ...]


def _rng(split: Split, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"structural-graph-proposal-role-v14:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: Split, font_index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    path = REPO_ROOT / registration.font_paths[font_index % len(registration.font_paths)]
    return ImageFont.truetype(str(path), size=size)


def _text_mask(
    size: tuple[int, int], text: str, position: tuple[int, int], font: ImageFont.FreeTypeFont,
) -> tuple[Image.Image, Box]:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.text(position, text, font=font, fill=255, anchor="lt")
    pixel_bounds = mask.getbbox()
    if pixel_bounds is None:
        raise RuntimeError("OCR V14 renderer produced an empty text mask")
    bounds = Box(*pixel_bounds)
    return mask, bounds


def _marker(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, ink: int, style: int) -> None:
    box = (x - radius, y - radius, x + radius, y + radius)
    if style % 3 == 0:
        draw.ellipse(box, fill=ink)
    elif style % 3 == 1:
        draw.ellipse(box, outline=ink, width=1 + style % 2)
    else:
        draw.rectangle(box, outline=ink, width=1 + style % 2)


def _structures(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    plot: tuple[int, int, int, int],
    ink: int,
    index: int,
) -> None:
    left, top, right, bottom = plot
    axis_width = 1 + (index // 3) % 3
    draw.line((left, top - 4, left, bottom), fill=ink, width=axis_width)
    draw.line((left, bottom, right + 3, bottom), fill=ink, width=axis_width)

    x_ticks = 9 + index % 7
    y_ticks = 5 + (index // 2) % 5
    for tick in range(x_ticks + 1):
        x = round(left + tick * (right - left) / x_ticks)
        length = 3 + (tick * 5 + index) % 10
        draw.line((x, bottom - 1, x, bottom + length), fill=ink, width=1 + (tick + index) % 2)
        if tick % 3 == index % 3:
            draw.line((x - 2, bottom + length, x + 2, bottom + length), fill=ink, width=1)
    for tick in range(y_ticks + 1):
        y = round(bottom - tick * (bottom - top) / y_ticks)
        length = 4 + (tick * 7 + index) % 9
        draw.line((left - length, y, left + 2, y), fill=ink, width=1 + tick % 2)
        if 0 < tick < y_ticks:
            grid_ink = min(232, max(172, ink + 164 + (index + tick) % 23))
            draw.line((left + 2, y, right, y), fill=grid_ink, width=1)
            if (tick + index) % 2 == 0:
                for x in range(left + 7, right, 13 + (index % 5)):
                    draw.point((x, y + 2), fill=grid_ink)

    divider_count = 1 + index % 3
    for divider in range(divider_count):
        x = round(left + (divider + 1) * (right - left) / (divider_count + 1))
        dash = 3 + (index + 2 * divider) % 7
        for y in range(top - 11, bottom, dash * 2 + 1):
            draw.line((x, y, x, min(bottom, y + dash)), fill=ink, width=1 + (index + divider) % 2)

    for series in range(2 + (index // 5) % 3):
        count = 8 + (index + 3 * series) % 9
        points: list[tuple[int, int]] = []
        for point in range(count):
            x = round(left + 13 + point * (right - left - 26) / max(1, count - 1))
            wave = np.cos(point * (0.43 + 0.07 * series) + index * 0.23)
            y = round(top + 21 + series * 29 + (wave + 1.0) * (22 + series * 4) + rng.integers(-8, 9))
            points.append((x, min(bottom - 9, max(top + 9, y))))
        for p0, p1 in zip(points, points[1:]):
            draw.line((*p0, *p1), fill=ink, width=1 + (series + index // 2) % 2)
        for point, (x, y) in enumerate(points):
            _marker(draw, x, y, 2 + (point + 2 * series + index) % 5, ink, point + series + index)

    legend_left = right + 11
    legend_top = top + 25 + index % 17
    legend_right = min(SCENE_WIDTH - 7, legend_left + 142)
    legend_bottom = min(bottom - 2, legend_top + 92)
    draw.rounded_rectangle((legend_left, legend_top, legend_right, legend_bottom), radius=3 + index % 5, outline=ink, width=1 + index % 2)
    _marker(draw, legend_left + 15, legend_top + 18, 2 + index % 4, ink, index)
    draw.line((legend_left + 6, legend_top + 49, legend_left + 31, legend_top + 49), fill=ink, width=2)
    for offset in range(0, max(0, legend_right - legend_left - 12), 11 + index % 4):
        y = legend_bottom - 8 - (offset % 17)
        draw.line((legend_left + 6 + offset, legend_bottom - 4, min(legend_right - 4, legend_left + 14 + offset), y), fill=ink, width=1)

    bracket_y = top + 8 + index % 22
    bracket_left = left + 19 + (index * 5) % 47
    bracket_right = min(right - 15, bracket_left + 42 + (index * 11) % 91)
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=1 + index % 2)
    hook = 7 + index % 13
    draw.arc((bracket_left - 3, bracket_y, bracket_left + 5, bracket_y + hook), 90, 270, fill=ink, width=2)
    draw.arc((bracket_right - 5, bracket_y, bracket_right + 3, bracket_y + hook), 270, 90, fill=ink, width=2)

    arrow_y = top + 51 + (index * 13) % max(25, bottom - top - 76)
    arrow_right = right - 12
    arrow_left = max(left + 91, arrow_right - 41 - index % 71)
    rise = -8 + index % 17
    draw.line((arrow_left, arrow_y - rise, arrow_right, arrow_y), fill=ink, width=1 + index % 2)
    draw.polygon(((arrow_right, arrow_y), (arrow_right - 11, arrow_y - 7), (arrow_right - 7, arrow_y), (arrow_right - 11, arrow_y + 7)), fill=ink)

    cross_x = left + 21 + (index * 37) % max(31, right - left - 47)
    cross_y = top + 27 + (index * 41) % max(27, bottom - top - 53)
    span = 6 + index % 13
    draw.line((cross_x - span, cross_y, cross_x + span, cross_y), fill=ink, width=1 + index % 3)
    draw.line((cross_x, cross_y - span, cross_x, cross_y + span), fill=ink, width=1 + (index // 2) % 3)
    draw.rectangle((cross_x - 4, cross_y - 4, cross_x + 4, cross_y + 4), outline=ink, width=1)

    for decoy in range(2 + index % 5):
        x = left + 17 + (index * 29 + decoy * 73) % max(25, right - left - 34)
        y = top + 19 + (index * 31 + decoy * 47) % max(23, bottom - top - 38)
        radius = 3 + (index + decoy) % 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=1 + decoy % 2)
        draw.line((x - radius - 3, y + radius + 2, x + radius + 4, y - radius - 2), fill=ink, width=1)


def _clear_and_draw_text(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    split: Split,
    index: int,
    background: int,
    ink: int,
    labels: tuple[tuple[str, str, tuple[int, int], int], ...],
) -> tuple[RoleTruth, ...]:
    truths: list[RoleTruth] = []
    for label_index, (text, role, position, font_size) in enumerate(labels):
        font = _font(split, index * 3 + label_index, font_size)
        mask, bounds = _text_mask(image.size, text, position, font)
        if role == "Annotation":
            horizontal = 120
        elif role == "LegendText":
            horizontal = 130
        elif role == "YTick":
            horizontal = 120
        elif role == "XTick":
            horizontal = 18
        else:
            horizontal = 7
        if role in {"YTick", "XTick"}:
            vertical = 20
        else:
            vertical = 5 + (3 if font_size <= 13 else 0)
        clear = (
            max(0, bounds.left - horizontal), max(0, bounds.top - vertical),
            min(SCENE_WIDTH - 1, bounds.right + horizontal), min(SCENE_HEIGHT - 1, bounds.bottom + vertical),
        )
        draw.rectangle(clear, fill=background)
        image.paste(ink, mask=mask)
        truths.append(RoleTruth(bounds, role, text))
    return tuple(truths)


def _degrade(image: Image.Image, split: Split, index: int, rng: np.random.Generator) -> Image.Image:
    if split == "train":
        if index % 3 != 1:
            image = image.filter(ImageFilter.BoxBlur(radius=float(rng.uniform(0.05, 0.29))))
        if index % 4 in {1, 2}:
            width = 653 + index % 31
            image = image.resize((width, SCENE_HEIGHT), resample=Image.Resampling.BICUBIC).resize(
                (SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR,
            )
        pixels = np.asarray(image, dtype=np.float32).copy()
        pixels = np.minimum(pixels, np.roll(pixels, 1 + index % 2, axis=1) + float(rng.uniform(3.0, 13.0)))
        pixels = np.clip((pixels - 132.0) * float(rng.uniform(0.91, 1.13)) + 132.0, 0, 255)
    elif split == "validation":
        width = 671 + index % 24
        height = 329 + (index * 5) % 18
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS).resize(
            (SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
        shadow = np.roll(np.roll(pixels, 1 + index % 3, axis=0), 1 + (index // 3) % 2, axis=1)
        pixels = np.minimum(pixels, shadow + 18.0 + index % 9)
        band = 19 + (index * 47) % (SCENE_HEIGHT - 38)
        pixels[band:band + 2, :] = np.maximum(pixels[band:band + 2, :], 198)
    else:
        filtered = image.filter(ImageFilter.MaxFilter(3)) if index % 2 == 0 else image.filter(ImageFilter.MinFilter(3))
        pixels = np.asarray(filtered, dtype=np.float32).copy()
        pixels = np.where(pixels < 214 + index % 17, pixels, 255.0)
        column = 27 + (index * 67) % (SCENE_WIDTH - 54)
        pixels[:, column:column + 2] = np.maximum(pixels[:, column:column + 2], 202)
    for _ in range(int(rng.integers(2, 13))):
        y = int(rng.integers(0, SCENE_HEIGHT))
        x = int(rng.integers(0, SCENE_WIDTH))
        pixels[y, x] = float(rng.integers(165, 249))
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8), mode="L")


def render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    if not 0 <= index < registration.scene_count:
        raise IndexError(f"OCR V14 {split} scene index out of range: {index}")
    rng = _rng(split, index)
    background = int(rng.integers(248, 256))
    ink = int(rng.integers(6, 52))
    plane = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    plane -= np.linspace(0.0, float(rng.integers(0, 7)), SCENE_WIDTH, dtype=np.float32)[None, :]
    plane -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(plane).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)

    if split == "train":
        left, top = int(rng.integers(98, 137)), int(rng.integers(48, 78))
        right, bottom = int(rng.integers(501, 541)), int(rng.integers(255, 289))
    elif split == "validation":
        left, top = 101 + (index * 3) % 33, 51 + (index * 7) % 23
        right, bottom = 507 + (index * 11) % 31, 259 + (index * 13) % 27
    else:
        left, top = 96 + (index * 11) % 39, 47 + (index * 13) % 28
        right, bottom = 503 + (index * 17) % 35, 253 + (index * 19) % 33
    plot = (left, top, right, bottom)
    _structures(draw, rng, plot, ink, index)
    image = _degrade(image, split, index, rng)
    draw = ImageDraw.Draw(image)

    small = 12 + (index * 3) % 4
    medium = 15 + (index * 5) % 5
    large = 18 + (index * 7) % 5
    labels = (
        (_Y_TICKS[(index * 7 + 2) % len(_Y_TICKS)], "YTick", (max(6, left - 64), top + 43 + index % max(25, bottom - top - 79)), small),
        (_X_TICKS[(index * 5 + 3) % len(_X_TICKS)], "XTick", (left + 23 + (index * 7) % max(29, (right - left) // 3), bottom + 16), small),
        (_AXIS_TITLES[(index * 4 + 2) % len(_AXIS_TITLES)], "AxisTitle", ((left + right) // 2 - 37, min(328, bottom + 31)), medium),
        (_PHASES[(index * 3 + 1) % len(_PHASES)], "PhaseHeading", (left + 43 + (index * 5) % max(31, right - left - 179), max(3, top - 37)), large),
        (_LEGENDS[(index * 6 + 2) % len(_LEGENDS)], "LegendText", (right + 70, top + 40 + index % 13), small),
        (_PARTICIPANTS[(index * 4 + 1) % len(_PARTICIPANTS)], "Participant", (right + 10, min(327, bottom + 18)), medium),
        (_ANNOTATIONS[(index * 5 + 2) % len(_ANNOTATIONS)], "Annotation", (right - 175 - index % 31, top + 89 + index % max(23, bottom - top - 128)), small),
        (_OTHERS[(index * 3 + 2) % len(_OTHERS)], "Other", (8 + index % 31, 4 + (index * 5) % 18), medium),
    )
    truths = _clear_and_draw_text(image, draw, split, index, background, ink, labels)
    return SceneSample(
        f"structural-graph-proposal-role-v14-{split}-{index:05d}", split,
        registration.renderer_family, registration.degradation_family,
        np.asarray(image, dtype=np.uint8).copy(), truths,
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    base = encode_v7_proposal(gray, proposal)
    if base.shape != (INPUT_CHANNELS, 32, ENCODED_WIDTH - 4):
        raise RuntimeError("OCR V14 predecessor encoding changed")
    result = np.zeros((INPUT_CHANNELS, 32, ENCODED_WIDTH), dtype=np.float32)
    result[:, :, :base.shape[2]] = base
    result[:, :, -4:] = np.asarray((
        (proposal.left + proposal.right + 1.0) / (2.0 * gray.shape[1]),
        (proposal.top + proposal.bottom + 1.0) / (2.0 * gray.shape[0]),
        proposal.left / gray.shape[1], proposal.top / gray.shape[0],
    ), dtype=np.float32)[None, None, :]
    if result.shape[2] != 128 + GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V14 geometry contract changed")
    return result


def proposal_targets(
    scene: SceneSample, items: tuple[Component, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = proposals(scene.raster) if items is None else items
    accepted: list[int] = []
    roles: list[int] = []
    for candidate in candidates:
        matches = [
            (index, box_iou(candidate.box, truth.box)) for index, truth in enumerate(scene.truths)
            if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
        ]
        if not matches:
            accepted.append(0)
            roles.append(-1)
        else:
            best = max(matches, key=lambda item: item[1])[0]
            accepted.append(1)
            roles.append(ROLE_ORDER.index(scene.truths[best].role))
    return np.asarray(accepted, dtype=np.int64), np.asarray(roles, dtype=np.int64)


def _negative_family(candidate: Component) -> str:
    density = candidate.area / max(1, candidate.width * candidate.height)
    aspect = candidate.width / max(1, candidate.height)
    if candidate.height >= 2.8 * max(1, candidate.width):
        return "vertical-ruler-divider"
    if aspect >= 4.4:
        return "horizontal-bracket-connector"
    if candidate.width <= 17 and candidate.height <= 17 and density >= 0.34:
        return "compact-marker-intersection"
    if candidate.count >= 6:
        return "multi-component-crosshatch"
    if candidate.count >= 3:
        return "clustered-legend-arrow"
    if density >= 0.52:
        return "dense-arrowhead-fill"
    if candidate.count == 1 and density <= 0.28:
        return "sparse-single-stroke"
    return "mixed-topology-structure"


def _select_negative_indices(candidates: tuple[Component, ...], labels: np.ndarray, cap: int) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index in np.flatnonzero(labels == 0).tolist():
        groups.setdefault(_negative_family(candidates[index]), []).append(index)
    selected: list[int] = []
    ordered = sorted(groups)
    offset = 0
    while len(selected) < cap:
        added = False
        for family in ordered:
            if offset < len(groups[family]):
                selected.append(groups[family][offset])
                added = True
                if len(selected) == cap:
                    break
        if not added:
            break
        offset += 1
    return selected


def training_examples(
    scenes: tuple[SceneSample, ...], *, negative_cap_per_scene: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    values: list[np.ndarray] = []
    proposal_labels: list[int] = []
    role_labels: list[int] = []
    stream = sha256()
    family_counts: dict[str, int] = {}
    positives = negatives = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        accept, roles = proposal_targets(scene, candidates)
        positive_indices = np.flatnonzero(accept == 1).tolist()
        negative_indices = _select_negative_indices(candidates, accept, negative_cap_per_scene)
        for item_index in positive_indices + negative_indices:
            value = encode_proposal(scene.raster, candidates[item_index])
            values.append(value)
            proposal_labels.append(int(accept[item_index]))
            role_labels.append(int(roles[item_index]))
            stream.update(scene.scene_id.encode())
            stream.update(value.tobytes(order="C"))
            stream.update(bytes((int(accept[item_index]), int(roles[item_index]) + 1)))
            if accept[item_index] == 0:
                family = _negative_family(candidates[item_index])
                family_counts[family] = family_counts.get(family, 0) + 1
        positives += len(positive_indices)
        negatives += len(negative_indices)
    if not values or positives == 0 or negatives == 0 or len(family_counts) < 5:
        raise RuntimeError("OCR V14 training requires positive text and at least five structural-negative families")
    evidence: dict[str, object] = {
        "scene_count": len(scenes),
        "negative_cap_per_scene": negative_cap_per_scene,
        "negative_sampling": "deterministic-round-robin-by-structural-family-v2",
        "negative_family_counts": dict(sorted(family_counts.items())),
        "proposal_count": len(values),
        "positive_proposal_count": positives,
        "negative_proposal_count": negatives,
        "tensor_label_stream_sha256": stream.hexdigest(),
        "validation_or_public_pixels_used": False,
        "predecessor_fixture_bytes_used": False,
        "v13_public_fixture_bytes_scene_truth_or_case_identity_used": False,
    }
    return (
        np.stack(values).astype(np.float32), np.asarray(proposal_labels, dtype=np.int64),
        np.asarray(role_labels, dtype=np.int64), evidence,
    )


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(f"{truth.box.left},{truth.box.top},{truth.box.right},{truth.box.bottom}|{truth.role}|{truth.text}\n".encode())
    return digest.hexdigest()


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, object]:
    proposal_count = positive_count = 0
    role_counts = {role: 0 for role in ROLE_ORDER}
    negative_families: dict[str, int] = {}
    for scene in scenes:
        candidates = proposals(scene.raster)
        accept, _ = proposal_targets(scene, candidates)
        proposal_count += len(candidates)
        positive_count += int(accept.sum())
        for candidate, label in zip(candidates, accept, strict=True):
            if label == 0:
                family = _negative_family(candidate)
                negative_families[family] = negative_families.get(family, 0) + 1
        for truth in scene.truths:
            matches = sum(box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM for candidate in candidates)
            if matches != 1:
                raise RuntimeError(f"{scene.scene_id} {truth.role} truth has {matches} proposals")
            role_counts[truth.role] += 1
    return {
        "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "proposal_count": proposal_count, "positive_proposal_count": positive_count,
        "negative_proposal_count": proposal_count - positive_count, "role_truth_counts": role_counts,
        "negative_family_counts": dict(sorted(negative_families.items())),
    }


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def save_sealed_public_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    manifest = {
        "schema": "graphreader.ocr-structural-graph-proposal-role-fixtures.v1",
        "split": "sealed_public", **proposal_summary(scenes), "cases": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        for scene in scenes:
            name = f"images/{scene.scene_id}.png"
            payload = _png_bytes(scene.raster)
            _zip_write(archive, name, payload)
            manifest["cases"].append({
                "scene_id": scene.scene_id, "image_path": name, "source_sha256": sha256(payload).hexdigest(),
                "renderer_family": scene.renderer_family, "degradation_family": scene.degradation_family,
            })
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
    private = {
        "schema": "graphreader.ocr-structural-graph-proposal-role-private-manifest.v1",
        **proposal_summary(scenes), "fixture_archive_sha256": sha256_file(path),
        "truths": [
            {"scene_id": scene.scene_id, "items": [
                {"box": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom], "role": truth.role, "text": truth.text}
                for truth in scene.truths
            ]} for scene in scenes
        ],
    }
    return private


def load_sealed_public_archive(path: Path, private_manifest_path: Path) -> tuple[SceneSample, ...]:
    private = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    truth_by_id = {case["scene_id"]: case["items"] for case in private["truths"]}
    scenes: list[SceneSample] = []
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256(payload).hexdigest() != case["source_sha256"]:
                raise RuntimeError("OCR V14 sealed fixture checksum mismatch")
            raster = np.asarray(Image.open(BytesIO(payload)).convert("L"), dtype=np.uint8).copy()
            truths = tuple(RoleTruth(Box(*item["box"]), item["role"], item["text"]) for item in truth_by_id[case["scene_id"]])
            scenes.append(SceneSample(
                case["scene_id"], "sealed_public", case["renderer_family"],
                case["degradation_family"], raster, truths,
            ))
    return tuple(scenes)


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_proposal", "load_sealed_public_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene", "save_sealed_public_archive",
    "split_fingerprint", "training_examples",
]
