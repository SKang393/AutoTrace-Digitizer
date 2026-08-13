# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh multi-scale morphology graph scenes for OCR V13."""

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
_Y_TICKS = ("0", "8", "20", "45", "72", "95", "5", "1.5", "85%")
_X_TICKS = ("1", "4", "9", "14", "18", "23", "28", "36")
_AXIS_TITLES = ("Session", "Measure", "Observation", "Trial", "Week")
_PHASES = ("Baseline", "Support", "Treatment", "Follow up", "Maintenance")
_LEGENDS = ("Target", "Control", "Probe", "Rate", "Level")
_PARTICIPANTS = ("Case A", "Learner", "Student", "Client", "Observer")
_ANNOTATIONS = ("Level shift", "Probe note", "Rule change", "Check point", "Data note")
_OTHERS = ("Daily", "Weekly", "Outcome", "Summary", "Response")


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
    material = f"morphology-proposal-role-v13:{registration.seed_offset}:{split}:{index}".encode()
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
        raise RuntimeError("OCR V13 renderer produced an empty text mask")
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
    axis_width = 1 + index % 3
    draw.line((left, top, left, bottom), fill=ink, width=axis_width)
    draw.line((left, bottom, right, bottom), fill=ink, width=axis_width)

    x_ticks = 7 + index % 8
    y_ticks = 4 + index % 5
    for tick in range(x_ticks + 1):
        x = round(left + tick * (right - left) / x_ticks)
        tick_length = 4 + (tick + index) % 7
        draw.line((x, bottom - 2, x, bottom + tick_length), fill=ink, width=1 + tick % 2)
    for tick in range(y_ticks + 1):
        y = round(bottom - tick * (bottom - top) / y_ticks)
        tick_length = 4 + (tick * 3 + index) % 7
        draw.line((left - tick_length, y, left + 3, y), fill=ink, width=1 + tick % 2)
        if tick not in (0, y_ticks) and (tick + index) % 3 != 0:
            grid_ink = min(229, max(156, ink + 148 + index % 19))
            draw.line((left + 2, y, right, y), fill=grid_ink, width=1)

    for divider in range(index % 4):
        x = round(left + (divider + 1) * (right - left) / (index % 4 + 1))
        dash = 5 + (index + divider) % 5
        for y in range(top - 7, bottom, dash * 2):
            draw.line((x, y, x, min(bottom, y + dash)), fill=ink, width=1 + divider % 2)

    for series in range(2 + index % 3):
        count = 7 + (index + 2 * series) % 10
        points: list[tuple[int, int]] = []
        for point in range(count):
            x = round(left + 15 + point * (right - left - 30) / max(1, count - 1))
            wave = np.sin(point * (0.52 + 0.08 * series) + index * 0.17)
            y = round(top + 30 + series * 27 + (wave + 1.0) * (25 + series * 3) + rng.integers(-7, 8))
            points.append((x, min(bottom - 10, max(top + 10, y))))
        for p0, p1 in zip(points, points[1:]):
            draw.line((*p0, *p1), fill=ink, width=1 + (series + index) % 2)
        for point, (x, y) in enumerate(points):
            _marker(draw, x, y, 3 + (point + series + index) % 4, ink, point + series + index)

    legend_left = right + 10
    legend_top = top + 34 + index % 14
    legend_right = min(SCENE_WIDTH - 8, legend_left + 126)
    legend_bottom = min(bottom - 3, legend_top + 84)
    draw.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=ink, width=1 + index % 2)
    _marker(draw, legend_left + 14, legend_top + 17, 3 + index % 3, ink, index)
    draw.line((legend_left + 7, legend_top + 48, legend_left + 28, legend_top + 48), fill=ink, width=2)

    bracket_y = top + 12 + index % 18
    bracket_left = left + 28 + index % 39
    bracket_right = min(right - 18, bracket_left + 38 + (index * 7) % 81)
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=1 + index % 2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 8 + index % 10), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 8 + index % 10), fill=ink, width=2)

    arrow_y = top + 58 + (index * 11) % max(24, bottom - top - 82)
    arrow_right = right - 18
    arrow_left = max(left + 85, arrow_right - 32 - index % 63)
    draw.line((arrow_left, arrow_y, arrow_right, arrow_y), fill=ink, width=1 + index % 2)
    draw.polygon(((arrow_right, arrow_y), (arrow_right - 10, arrow_y - 6), (arrow_right - 7, arrow_y), (arrow_right - 10, arrow_y + 6)), fill=ink)

    cross_x = left + 25 + (index * 31) % max(30, right - left - 50)
    cross_y = top + 30 + (index * 37) % max(25, bottom - top - 60)
    span = 7 + index % 11
    draw.line((cross_x - span, cross_y, cross_x + span, cross_y), fill=ink, width=1 + index % 3)
    draw.line((cross_x, cross_y - span, cross_x, cross_y + span), fill=ink, width=1 + index % 3)


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
        if index % 3 == 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.08, 0.36))))
        if index % 4 == 1:
            width = 640 + index % 19
            image = image.resize((width, 320), resample=Image.Resampling.BICUBIC).resize(
                (SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR,
            )
        pixels = np.asarray(image, dtype=np.float32).copy()
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(0.94, 1.10)) + 128.0, 0, 255)
    elif split == "validation":
        width = 625 + index % 28
        height = 309 + (index * 3) % 19
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS).resize(
            (SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
        band = 25 + (index * 43) % 281
        pixels[max(0, band - 1):min(SCENE_HEIGHT, band + 2), :] = np.maximum(
            pixels[max(0, band - 1):min(SCENE_HEIGHT, band + 2), :], 186,
        )
    else:
        pixels = np.asarray(image, dtype=np.float32).copy()
        pixels = 255.0 * np.power(pixels / 255.0, float(rng.uniform(0.91, 1.09)))
        column = 31 + (index * 61) % 607
        pixels[:, max(0, column - 1):min(SCENE_WIDTH, column + 2)] = np.maximum(
            pixels[:, max(0, column - 1):min(SCENE_WIDTH, column + 2)], 188,
        )
    for _ in range(int(rng.integers(0, 9))):
        y = int(rng.integers(0, SCENE_HEIGHT))
        x = int(rng.integers(0, SCENE_WIDTH))
        pixels[y, x] = float(rng.integers(178, 245))
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8), mode="L")


def render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    if not 0 <= index < registration.scene_count:
        raise IndexError(f"OCR V13 {split} scene index out of range: {index}")
    rng = _rng(split, index)
    background = int(rng.integers(248, 256))
    ink = int(rng.integers(6, 52))
    plane = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    plane -= np.linspace(0.0, float(rng.integers(0, 7)), SCENE_WIDTH, dtype=np.float32)[None, :]
    plane -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(plane).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)

    if split == "train":
        left, top = int(rng.integers(91, 128)), int(rng.integers(51, 81))
        right, bottom = int(rng.integers(487, 530)), int(rng.integers(244, 276))
    elif split == "validation":
        left, top = 97 + index % 29, 56 + (index * 3) % 21
        right, bottom = 491 + (index * 5) % 33, 247 + (index * 7) % 25
    else:
        left, top = 93 + (index * 7) % 34, 53 + (index * 11) % 24
        right, bottom = 485 + (index * 13) % 39, 243 + (index * 17) % 30
    plot = (left, top, right, bottom)
    _structures(draw, rng, plot, ink, index)
    image = _degrade(image, split, index, rng)
    draw = ImageDraw.Draw(image)

    small = 12 + index % 4
    medium = 15 + (index * 3) % 5
    large = 17 + (index * 5) % 6
    labels = (
        (_Y_TICKS[(index * 5 + 1) % len(_Y_TICKS)], "YTick", (max(5, left - 60), top + 39 + index % max(24, bottom - top - 73)), small),
        (_X_TICKS[(index * 7 + 2) % len(_X_TICKS)], "XTick", (left + 18 + index % max(28, (right - left) // 3), bottom + 17), small),
        (_AXIS_TITLES[(index * 3 + 1) % len(_AXIS_TITLES)], "AxisTitle", ((left + right) // 2 - 34, min(312, bottom + 30)), medium),
        (_PHASES[(index * 2 + 3) % len(_PHASES)], "PhaseHeading", (left + 49 + index % max(30, right - left - 165), max(3, top - 34)), large),
        (_LEGENDS[(index * 4 + 1) % len(_LEGENDS)], "LegendText", (right + 72, top + 42 + index % 10), small),
        (_PARTICIPANTS[(index * 3 + 2) % len(_PARTICIPANTS)], "Participant", (right + 9, min(313, bottom + 19)), medium),
        (_ANNOTATIONS[(index * 4 + 3) % len(_ANNOTATIONS)], "Annotation", (right - 160 - index % 29, top + 83 + index % max(22, bottom - top - 119)), small),
        (_OTHERS[(index * 2 + 1) % len(_OTHERS)], "Other", (7 + index % 27, 5 + (index * 3) % 15), medium),
    )
    truths = _clear_and_draw_text(image, draw, split, index, background, ink, labels)
    return SceneSample(
        f"morphology-proposal-role-v13-{split}-{index:05d}", split,
        registration.renderer_family, registration.degradation_family,
        np.asarray(image, dtype=np.uint8).copy(), truths,
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    base = encode_v7_proposal(gray, proposal)
    if base.shape != (INPUT_CHANNELS, 32, ENCODED_WIDTH - 4):
        raise RuntimeError("OCR V13 predecessor encoding changed")
    result = np.zeros((INPUT_CHANNELS, 32, ENCODED_WIDTH), dtype=np.float32)
    result[:, :, :base.shape[2]] = base
    result[:, :, -4:] = np.asarray((
        (proposal.left + proposal.right + 1.0) / (2.0 * gray.shape[1]),
        (proposal.top + proposal.bottom + 1.0) / (2.0 * gray.shape[0]),
        proposal.left / gray.shape[1], proposal.top / gray.shape[0],
    ), dtype=np.float32)[None, None, :]
    if result.shape[2] != 128 + GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V13 geometry contract changed")
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
    if candidate.height >= 3 * max(1, candidate.width):
        return "vertical-stroke-divider"
    if aspect >= 5.0:
        return "horizontal-stroke-bracket"
    if candidate.width <= 16 and candidate.height <= 16:
        return "marker-tick-intersection"
    if density >= 0.48:
        return "dense-marker-arrow"
    if candidate.count == 1 and density <= 0.30:
        return "sparse-single-component"
    return "mixed-connected-structure"


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
        raise RuntimeError("OCR V13 training requires positive text and at least five morphology-negative families")
    evidence: dict[str, object] = {
        "scene_count": len(scenes),
        "negative_cap_per_scene": negative_cap_per_scene,
        "negative_sampling": "deterministic-round-robin-by-morphology-family-v1",
        "negative_family_counts": dict(sorted(family_counts.items())),
        "proposal_count": len(values),
        "positive_proposal_count": positives,
        "negative_proposal_count": negatives,
        "tensor_label_stream_sha256": stream.hexdigest(),
        "validation_or_public_pixels_used": False,
        "predecessor_fixture_bytes_used": False,
        "v12_public_fixture_bytes_scene_truth_or_case_identity_used": False,
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
        "schema": "graphreader.ocr-morphology-proposal-role-fixtures.v1",
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
        "schema": "graphreader.ocr-morphology-proposal-role-private-manifest.v1",
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
                raise RuntimeError("OCR V13 sealed fixture checksum mismatch")
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
