# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh plot-layout-conditioned scientific graph scenes for OCR V15."""

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
    BASE_GEOMETRY_FEATURE_COUNT,
    CROP_WIDTH,
    ENCODED_WIDTH,
    GEOMETRY_FEATURE_COUNT,
    INPUT_CHANNELS,
    PLOT_GEOMETRY_FEATURE_COUNT,
    ROLE_ORDER,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    TRUTH_MATCH_IOU_MINIMUM,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]
_Y_TICKS = ("0", "9", "18", "27", "45", "72", "3.5", "-6", "85%")
_X_TICKS = ("1", "4", "7", "11", "16", "22", "31", "48")
_AXIS_TITLES = ("Sessions", "Probes", "Intervals", "Samples", "Attempts")
_PHASES = ("Orientation", "Acquisition", "Fluency", "Retention", "Followup")
_LEGENDS = ("Measured", "Target", "Probe", "Average", "Band")
_PARTICIPANTS = ("Learner A", "Student C", "Observer", "Client D", "Group B")
_ANNOTATIONS = ("Prompt changed", "Schedule revised", "Rule applied", "Probe moved", "Level checked")
_OTHERS = ("Summary", "Weekly", "Outcome", "Record", "Figure")


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
    material = f"layout-conditioned-proposal-role-v15:{registration.seed_offset}:{split}:{index}".encode()
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
    bounds = mask.getbbox()
    if bounds is None:
        raise RuntimeError("OCR V15 renderer produced an empty text mask")
    return mask, Box(*bounds)


def _marker(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, ink: int, style: int) -> None:
    bounds = (x - radius, y - radius, x + radius, y + radius)
    if style % 4 == 0:
        draw.ellipse(bounds, fill=ink)
    elif style % 4 == 1:
        draw.ellipse(bounds, outline=ink, width=1 + style % 2)
    elif style % 4 == 2:
        draw.rectangle(bounds, outline=ink, width=1 + style % 2)
    else:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), outline=ink)


def _structures(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    plot: Box,
    ink: int,
    index: int,
) -> None:
    left, top, right, bottom = plot.left, plot.top, plot.right, plot.bottom
    axis_width = 1 + (index % 3)
    draw.line((left, top - 3, left, bottom), fill=ink, width=axis_width)
    draw.line((left, bottom, right + 4, bottom), fill=ink, width=axis_width)

    x_ticks = 8 + (index * 3) % 9
    y_ticks = 5 + (index * 5) % 6
    for tick in range(x_ticks + 1):
        x = round(left + tick * (right - left) / x_ticks)
        length = 4 + (index + tick * 7) % 9
        draw.line((x, bottom - 1, x, bottom + length), fill=ink, width=1 + (index + tick) % 2)
    for tick in range(y_ticks + 1):
        y = round(bottom - tick * (bottom - top) / y_ticks)
        length = 4 + (index * 2 + tick * 5) % 10
        draw.line((left - length, y, left + 2, y), fill=ink, width=1 + tick % 2)
        if 0 < tick < y_ticks:
            gray = min(238, ink + 174 + (index + tick) % 19)
            draw.line((left + 2, y, right, y), fill=gray, width=1)

    divider_count = 1 + (index // 3) % 3
    for divider in range(divider_count):
        x = round(left + (divider + 1) * (right - left) / (divider_count + 1))
        dash = 3 + (index + divider) % 6
        for y in range(top - 13, bottom, dash * 2 + 2):
            draw.line((x, y, x, min(bottom, y + dash)), fill=ink, width=1 + divider % 2)

    for series in range(2 + index % 3):
        count = 7 + (index * 2 + series * 3) % 10
        points: list[tuple[int, int]] = []
        for point in range(count):
            x = round(left + 12 + point * (right - left - 24) / max(1, count - 1))
            phase = point * (0.51 + series * 0.08) + index * 0.17
            y = round(top + 24 + series * 31 + (np.sin(phase) + 1.0) * (18 + 3 * series))
            y += int(rng.integers(-7, 8))
            points.append((x, min(bottom - 8, max(top + 8, y))))
        for first, second in zip(points, points[1:]):
            draw.line((*first, *second), fill=ink, width=1 + (index + series) % 2)
        for point, (x, y) in enumerate(points):
            _marker(draw, x, y, 2 + (index + point + series) % 5, ink, index + point + series)

    legend_left = right + 12
    legend_top = top + 22 + index % 19
    legend_right = min(SCENE_WIDTH - 8, legend_left + 143)
    legend_bottom = min(bottom - 3, legend_top + 86)
    draw.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=ink, width=1 + index % 2)
    _marker(draw, legend_left + 16, legend_top + 18, 3 + index % 3, ink, index)
    draw.line((legend_left + 8, legend_top + 46, legend_left + 34, legend_top + 46), fill=ink, width=2)
    draw.arc((legend_left + 9, legend_top + 61, legend_left + 33, legend_top + 78), 190, 350, fill=ink, width=1)

    bracket_y = top + 11 + index % 24
    bracket_left = left + 24 + (index * 11) % 53
    bracket_right = min(right - 18, bracket_left + 37 + (index * 17) % 97)
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=1 + index % 2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 10 + index % 7), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 10 + index % 7), fill=ink, width=2)

    arrow_y = top + 63 + (index * 11) % max(23, bottom - top - 86)
    arrow_left = left + 67 + (index * 13) % max(41, right - left - 171)
    arrow_right = min(right - 9, arrow_left + 58 + index % 41)
    draw.line((arrow_left, arrow_y + 7, arrow_right, arrow_y), fill=ink, width=1 + index % 2)
    draw.polygon(((arrow_right, arrow_y), (arrow_right - 10, arrow_y - 6), (arrow_right - 7, arrow_y), (arrow_right - 10, arrow_y + 7)), fill=ink)

    for decoy in range(3 + index % 5):
        x = left + 18 + (index * 43 + decoy * 61) % max(27, right - left - 36)
        y = top + 21 + (index * 37 + decoy * 53) % max(25, bottom - top - 42)
        radius = 3 + (index + decoy * 2) % 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=1 + decoy % 2)
        draw.line((x - radius - 4, y, x + radius + 4, y), fill=ink, width=1)
        draw.line((x, y - radius - 4, x, y + radius + 4), fill=ink, width=1)


def _clear_and_draw_text(
    image: Image.Image,
    split: Split,
    index: int,
    background: int,
    ink: int,
    labels: tuple[tuple[str, str, tuple[int, int], int], ...],
) -> tuple[RoleTruth, ...]:
    draw = ImageDraw.Draw(image)
    truths: list[RoleTruth] = []
    for label_index, (text, role, position, font_size) in enumerate(labels):
        font = _font(split, index * 5 + label_index, font_size)
        mask, bounds = _text_mask(image.size, text, position, font)
        horizontal = 132 if role in {"Annotation", "LegendText"} else 116 if role == "YTick" else 18 if role == "XTick" else 8
        vertical = 21 if role in {"YTick", "XTick"} else 8
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
        if index % 4 != 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.04, 0.24))))
        width = 645 + (index * 7) % 47
        image = image.resize((width, SCENE_HEIGHT), Image.Resampling.BICUBIC).resize(
            (SCENE_WIDTH, SCENE_HEIGHT), Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
        shifted = np.roll(pixels, 1 + index % 2, axis=1)
        pixels = np.minimum(pixels, shifted + 8.0 + index % 11)
    elif split == "validation":
        width = 662 + (index * 11) % 37
        height = 331 + (index * 3) % 15
        image = image.resize((width, height), Image.Resampling.LANCZOS).resize(
            (SCENE_WIDTH, SCENE_HEIGHT), Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
        shadow = np.roll(np.roll(pixels, 1 + index % 2, axis=0), 2, axis=1)
        pixels = np.minimum(pixels, shadow + 17.0 + index % 7)
        for step in range(3):
            row = 23 + (index * 31 + step * 79) % (SCENE_HEIGHT - 46)
            pixels[row:row + 1, :] = np.maximum(pixels[row:row + 1, :], 204)
    else:
        filtered = image.filter(ImageFilter.MinFilter(3)) if index % 3 == 0 else image.filter(ImageFilter.MaxFilter(3))
        pixels = np.asarray(filtered, dtype=np.float32).copy()
        row = 31 + (index * 59) % (SCENE_HEIGHT - 62)
        column = 29 + (index * 71) % (SCENE_WIDTH - 58)
        pixels[row:row + 2, :] = np.maximum(pixels[row:row + 2, :], 201)
        pixels[:, column:column + 1] = np.maximum(pixels[:, column:column + 1], 209)
    pixels = np.clip((pixels - 130.0) * float(rng.uniform(0.92, 1.12)) + 130.0, 0, 255)
    for _ in range(int(rng.integers(3, 15))):
        y = int(rng.integers(0, SCENE_HEIGHT))
        x = int(rng.integers(0, SCENE_WIDTH))
        pixels[y, x] = float(rng.integers(168, 247))
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8), mode="L")


def render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    if not 0 <= index < registration.scene_count:
        raise IndexError(f"OCR V15 {split} scene index out of range: {index}")
    rng = _rng(split, index)
    background = int(rng.integers(248, 256))
    ink = int(rng.integers(5, 49))
    plane = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    plane -= np.linspace(0.0, float(rng.integers(0, 8)), SCENE_WIDTH, dtype=np.float32)[None, :]
    plane -= np.linspace(0.0, float(rng.integers(0, 6)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(plane).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)

    if split == "train":
        left, top = int(rng.integers(104, 146)), int(rng.integers(53, 83))
        right, bottom = int(rng.integers(493, 532)), int(rng.integers(251, 282))
    elif split == "validation":
        left, top = 106 + (index * 7) % 37, 54 + (index * 11) % 27
        right, bottom = 496 + (index * 13) % 34, 253 + (index * 17) % 29
    else:
        left, top = 101 + (index * 13) % 43, 50 + (index * 17) % 31
        right, bottom = 491 + (index * 19) % 39, 249 + (index * 23) % 33
    plot = Box(left, top, right, bottom)
    _structures(draw, rng, plot, ink, index)
    image = _degrade(image, split, index, rng)

    small = 12 + (index * 5) % 4
    medium = 15 + (index * 7) % 5
    large = 18 + (index * 11) % 5
    labels = (
        (_Y_TICKS[(index * 5 + 1) % len(_Y_TICKS)], "YTick", (max(6, left - 63), top + 39 + index % max(29, bottom - top - 73)), small),
        (_X_TICKS[(index * 7 + 2) % len(_X_TICKS)], "XTick", (left + 31 + (index * 11) % max(31, (right - left) // 3), bottom + 15), small),
        (_AXIS_TITLES[(index * 3 + 1) % len(_AXIS_TITLES)], "AxisTitle", ((left + right) // 2 - 43, min(326, bottom + 30)), medium),
        (_PHASES[(index * 2 + 3) % len(_PHASES)], "PhaseHeading", (left + 92 + (index * 13) % max(29, right - left - 251), max(3, top - 38)), large),
        (_LEGENDS[(index * 4 + 1) % len(_LEGENDS)], "LegendText", (right + 66, top + 37 + index % 17), small),
        (_PARTICIPANTS[(index * 3 + 2) % len(_PARTICIPANTS)], "Participant", (right + 8, min(326, bottom + 17)), medium),
        (_ANNOTATIONS[(index * 2 + 4) % len(_ANNOTATIONS)], "Annotation", (right - 181 - index % 29, top + 83 + index % max(29, bottom - top - 127)), small),
        (_OTHERS[(index * 4 + 2) % len(_OTHERS)], "Other", (7 + index % 29, 4 + (index * 7) % 19), medium),
    )
    truths = _clear_and_draw_text(image, split, index, background, ink, labels)
    return SceneSample(
        f"layout-conditioned-proposal-role-v15-{split}-{index:05d}", split,
        registration.renderer_family, registration.degradation_family,
        np.asarray(image, dtype=np.uint8).copy(), plot, truths,
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: Component, plot: Box) -> np.ndarray:
    base = encode_v7_proposal(gray, proposal)
    if base.shape != (INPUT_CHANNELS, 32, CROP_WIDTH + 12):
        raise RuntimeError("OCR V15 predecessor encoding changed")
    plot_width = max(1.0, float(plot.right - plot.left + 1))
    plot_height = max(1.0, float(plot.bottom - plot.top + 1))
    center_x = (proposal.left + proposal.right + 1.0) / 2.0
    center_y = (proposal.top + proposal.bottom + 1.0) / 2.0
    absolute = np.asarray((
        center_x / gray.shape[1], center_y / gray.shape[0],
        proposal.left / gray.shape[1], proposal.top / gray.shape[0],
    ), dtype=np.float32)
    relative = np.asarray((
        (center_x - plot.left) / plot_width,
        (center_y - plot.top) / plot_height,
        (proposal.left - plot.left) / plot_width,
        (proposal.right - plot.left + 1.0) / plot_width,
        (proposal.top - plot.top) / plot_height,
        (proposal.bottom - plot.top + 1.0) / plot_height,
        proposal.width / plot_width,
        proposal.height / plot_height,
    ), dtype=np.float32)
    result = np.zeros((INPUT_CHANNELS, 32, ENCODED_WIDTH), dtype=np.float32)
    result[:, :, :base.shape[2]] = base
    result[:, :, base.shape[2]:base.shape[2] + 4] = absolute[None, None, :]
    result[:, :, base.shape[2] + 4:] = relative[None, None, :]
    if result.shape[2] != CROP_WIDTH + GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V15 geometry contract changed")
    if BASE_GEOMETRY_FEATURE_COUNT != 16 or len(relative) != PLOT_GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V15 plot geometry width changed")
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
    if candidate.height >= 2.7 * max(1, candidate.width):
        return "vertical-divider-ruler"
    if aspect >= 4.2:
        return "horizontal-bracket-connector"
    if candidate.width <= 18 and candidate.height <= 18 and density >= 0.32:
        return "compact-marker-cross"
    if candidate.count >= 6:
        return "multi-component-frame"
    if candidate.count >= 3:
        return "clustered-arrow-legend"
    if density >= 0.50:
        return "dense-fill-arrowhead"
    if candidate.count == 1 and density <= 0.29:
        return "sparse-stroke"
    return "mixed-layout-structure"


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
            value = encode_proposal(scene.raster, candidates[item_index], scene.plot)
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
        raise RuntimeError("OCR V15 training requires text and at least five structural-negative families")
    evidence: dict[str, object] = {
        "scene_count": len(scenes),
        "negative_cap_per_scene": negative_cap_per_scene,
        "negative_sampling": "deterministic-round-robin-by-structural-family-v3",
        "negative_family_counts": dict(sorted(family_counts.items())),
        "proposal_count": len(values),
        "positive_proposal_count": positives,
        "negative_proposal_count": negatives,
        "tensor_label_stream_sha256": stream.hexdigest(),
        "validation_or_public_pixels_used": False,
        "predecessor_fixture_bytes_used": False,
        "v14_validation_fixture_bytes_scene_truth_or_case_identity_used": False,
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
        digest.update(f"{scene.plot.left},{scene.plot.top},{scene.plot.right},{scene.plot.bottom}\n".encode())
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
    manifest: dict[str, object] = {
        "schema": "graphreader.ocr-layout-conditioned-proposal-role-fixtures.v1",
        "split": "sealed_public", **proposal_summary(scenes), "cases": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        cases: list[dict[str, object]] = []
        for scene in scenes:
            name = f"images/{scene.scene_id}.png"
            payload = _png_bytes(scene.raster)
            _zip_write(archive, name, payload)
            cases.append({
                "scene_id": scene.scene_id, "image_path": name,
                "source_sha256": sha256(payload).hexdigest(),
                "renderer_family": scene.renderer_family,
                "degradation_family": scene.degradation_family,
                "plot_box": [scene.plot.left, scene.plot.top, scene.plot.right, scene.plot.bottom],
            })
        manifest["cases"] = cases
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
    private = {
        "schema": "graphreader.ocr-layout-conditioned-proposal-role-private-manifest.v1",
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
                raise RuntimeError("OCR V15 sealed fixture checksum mismatch")
            raster = np.asarray(Image.open(BytesIO(payload)).convert("L"), dtype=np.uint8).copy()
            truths = tuple(RoleTruth(Box(*item["box"]), item["role"], item["text"]) for item in truth_by_id[case["scene_id"]])
            scenes.append(SceneSample(
                case["scene_id"], "sealed_public", case["renderer_family"],
                case["degradation_family"], raster, Box(*case["plot_box"]), truths,
            ))
    return tuple(scenes)


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_proposal", "load_sealed_public_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene", "save_sealed_public_archive",
    "split_fingerprint", "training_examples",
]
