# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Independent variable-layout scientific graph scenes for OCR V12."""

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
_Y_TICKS = ("0", "15", "25", "50", "75", "100", "-10", "2.5", "60%")
_X_TICKS = ("10", "13", "16", "19", "12", "18", "24", "30")
_AXIS_TITLES = ("Session", "Observation", "Trial", "Visit", "Day")
_PHASES = ("Baseline", "Intervention", "Treatment", "Followup", "Maintenance")
_LEGENDS = ("Target", "Probe", "Rate", "Level", "Data")
_PARTICIPANTS = ("Learner", "Student", "Client", "Observer", "Case")
_ANNOTATIONS = ("Check point", "Level shift", "Probe note", "Rule change", "Data note")
_OTHERS = ("Weekly", "Daily", "Outcome", "Measure", "Summary")


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
    material = f"full-scene-proposal-role-v12:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: Split, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    return ImageFont.truetype(str(REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]), size=size)


def _text_mask(
    size: tuple[int, int], text: str, position: tuple[int, int], font: ImageFont.FreeTypeFont,
) -> tuple[Image.Image, Box]:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    bounds = draw.textbbox(position, text, font=font, anchor="lt")
    draw.text(position, text, font=font, fill=255, anchor="lt", stroke_width=0)
    return mask, Box(*bounds)


def _draw_marker(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, ink: int, filled: bool) -> None:
    bounds = (x - radius, y - radius, x + radius, y + radius)
    if filled:
        draw.ellipse(bounds, fill=ink)
    else:
        draw.ellipse(bounds, outline=ink, width=2)


def _draw_graph_structures(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    plot: tuple[int, int, int, int],
    ink: int,
    index: int,
) -> None:
    left, top, right, bottom = plot
    draw.line((left, top, left, bottom), fill=ink, width=2)
    draw.line((left, bottom, right, bottom), fill=ink, width=2)
    x_ticks = 8 + index % 5
    y_ticks = 5 + index % 3
    for tick in range(x_ticks + 1):
        x = round(left + tick * (right - left) / x_ticks)
        draw.line((x, bottom - 4, x, bottom + 7), fill=ink, width=2)
    for tick in range(y_ticks + 1):
        y = round(bottom - tick * (bottom - top) / y_ticks)
        draw.line((left - 7, y, left + 4, y), fill=ink, width=2)
        if tick not in {0, y_ticks} and (tick + index) % 2 == 0:
            draw.line((left + 1, y, right, y), fill=max(ink, 178), width=1)

    divider_count = 1 + index % 3
    for divider in range(divider_count):
        x = round(left + (divider + 1) * (right - left) / (divider_count + 1))
        draw.line((x, top - 5, x, bottom), fill=ink, width=2)

    for series in range(2 + index % 2):
        point_count = 9 + (index + series) % 6
        points: list[tuple[int, int]] = []
        for point in range(point_count):
            x = round(left + 18 + point * (right - left - 36) / max(1, point_count - 1))
            wave = np.sin((point + series * 1.7 + index * 0.11) * 0.83)
            y = round(top + 42 + series * 38 + (wave + 1.0) * 30 + rng.integers(-5, 6))
            y = min(bottom - 13, max(top + 13, y))
            points.append((x, y))
        draw.line(points, fill=ink, width=2)
        for point_index, (x, y) in enumerate(points):
            _draw_marker(draw, x, y, 4 + (series + point_index) % 2, ink, (series + point_index + index) % 3 != 0)

    legend_left = right + 12
    legend_top = top + 42
    legend_right = min(SCENE_WIDTH - 8, legend_left + 112)
    legend_bottom = min(bottom - 8, legend_top + 74)
    draw.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=ink, width=1)
    _draw_marker(draw, legend_left + 13, legend_top + 17, 4, ink, index % 2 == 0)
    draw.line((legend_left + 7, legend_top + 48, legend_left + 24, legend_top + 48), fill=ink, width=2)

    bracket_y = top + 14
    bracket_left = left + 35
    bracket_right = min(right - 20, bracket_left + 62 + index % 37)
    draw.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink, width=2)
    draw.line((bracket_left, bracket_y, bracket_left, bracket_y + 13), fill=ink, width=2)
    draw.line((bracket_right, bracket_y, bracket_right, bracket_y + 13), fill=ink, width=2)

    arrow_y = top + 68 + index % 43
    arrow_right = right - 22
    arrow_left = max(left + 130, arrow_right - 68)
    draw.line((arrow_left, arrow_y, arrow_right, arrow_y), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 11, arrow_y - 7), fill=ink, width=2)
    draw.line((arrow_right, arrow_y, arrow_right - 11, arrow_y + 7), fill=ink, width=2)

    cross_x = left + 34 + index % 97
    cross_y = bottom - 27 - index % 41
    draw.line((cross_x - 12, cross_y, cross_x + 12, cross_y), fill=ink, width=2)
    draw.line((cross_x, cross_y - 12, cross_x, cross_y + 12), fill=ink, width=2)


def render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(247, 256))
    ink = int(rng.integers(8, 48))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(0.0, float(rng.integers(0, 6)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(0.0, float(rng.integers(0, 4)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)

    left = int(rng.integers(86, 117))
    top = int(rng.integers(47, 72))
    right = int(rng.integers(466, 506))
    bottom = int(rng.integers(232, 260))
    if split == "validation":
        left, top = 91 + index % 22, 52 + index % 17
        right, bottom = 474 + index % 25, 237 + index % 19
    elif split == "sealed_public":
        left, top = 88 + index % 27, 49 + index % 19
        right, bottom = 469 + index % 31, 234 + index % 22
    plot = (left, top, right, bottom)
    _draw_graph_structures(draw, rng, plot, ink, index)

    labels = (
        (_Y_TICKS[(index * 5 + 2) % len(_Y_TICKS)], "YTick", (max(7, left - 58), top + 54 + index % max(25, bottom - top - 88))),
        (_X_TICKS[(index * 7 + 1) % len(_X_TICKS)], "XTick", (left + 58 + index % max(40, right - left - 118), bottom + 6)),
        (_AXIS_TITLES[(index * 3 + 2) % len(_AXIS_TITLES)], "AxisTitle", ((left + right) // 2 - 30, min(298, bottom + 30))),
        (_PHASES[(index * 2 + 1) % len(_PHASES)], "PhaseHeading", (left + 66 + index % max(35, right - left - 180), max(3, top - 31))),
        (_LEGENDS[(index * 4 + 2) % len(_LEGENDS)], "LegendText", (right + 31, top + 47)),
        (_PARTICIPANTS[(index * 3 + 1) % len(_PARTICIPANTS)], "Participant", (right + 10, min(292, bottom + 20))),
        (_ANNOTATIONS[(index * 4 + 1) % len(_ANNOTATIONS)], "Annotation", (right - 151 - index % 31, top + 91 + index % max(20, bottom - top - 128))),
        (_OTHERS[(index * 2 + 2) % len(_OTHERS)], "Other", (8 + index % 21, 5 + index % 12)),
    )
    truths: list[RoleTruth] = []
    for label_index, (text, role, position) in enumerate(labels):
        size = 15 + int(rng.integers(0, 4))
        font = _font(split, index + label_index, size)
        mask, bounds = _text_mask(image.size, text, position, font)
        if role == "YTick":
            clear = (
                0, max(0, bounds.top - 9),
                min(SCENE_WIDTH - 1, left + 48), min(SCENE_HEIGHT - 1, bounds.bottom + 12),
            )
        elif role == "Annotation":
            clear = (
                max(0, bounds.left - 110), max(0, bounds.top - 14),
                min(SCENE_WIDTH - 1, bounds.right + 80), min(SCENE_HEIGHT - 1, bounds.bottom + 15),
            )
        elif role == "LegendText":
            clear = (
                max(0, right - 45), max(0, bounds.top - 10),
                min(SCENE_WIDTH - 1, bounds.right + 12), min(SCENE_HEIGHT - 1, bounds.bottom + 8),
            )
        else:
            clear = (
                max(0, bounds.left - 3), max(0, bounds.top - 3),
                min(SCENE_WIDTH - 1, bounds.right + 3), min(SCENE_HEIGHT - 1, bounds.bottom + 3),
            )
        draw.rectangle(clear, fill=background)
        image.paste(ink, mask=mask)
        truths.append(RoleTruth(bounds, role, text))

    if split == "train":
        pixels = np.asarray(image, dtype=np.float32)
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(0.98, 1.08)) + 128.0, 0, 255)
        image = Image.fromarray(np.rint(pixels).astype(np.uint8), mode="L")
        if index % 4 == 0:
            image = image.filter(ImageFilter.BoxBlur(radius=0.18))
    elif split == "validation":
        reduced = image.resize((608, 304), resample=Image.Resampling.BICUBIC)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.uint8).copy()
        row = 39 + (index * 29) % 244
        pixels[max(0, row - 1):min(SCENE_HEIGHT, row + 1), :] = np.maximum(
            pixels[max(0, row - 1):min(SCENE_HEIGHT, row + 1), :], 183,
        )
        image = Image.fromarray(pixels, mode="L")
    else:
        pixels = np.asarray(image, dtype=np.float32)
        pixels = 255.0 * np.power(pixels / 255.0, float(rng.uniform(0.94, 1.06)))
        pixels = np.rint(pixels).astype(np.uint16) // 3 * 3
        column = 43 + (index * 47) % 552
        pixels[:, max(0, column - 1):min(SCENE_WIDTH, column + 1)] = np.maximum(
            pixels[:, max(0, column - 1):min(SCENE_WIDTH, column + 1)], 184,
        )
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")

    raster = np.asarray(image, dtype=np.uint8).copy()
    for _ in range(int(rng.integers(0, 7))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(190, 242))
    return SceneSample(
        f"full-scene-proposal-role-v12-{split}-{index:05d}", split,
        registration.renderer_family, registration.degradation_family, raster, tuple(truths),
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    base = encode_v7_proposal(gray, proposal)
    if base.shape != (INPUT_CHANNELS, 32, ENCODED_WIDTH - 4):
        raise RuntimeError("OCR V12 predecessor encoding changed")
    result = np.zeros((INPUT_CHANNELS, 32, ENCODED_WIDTH), dtype=np.float32)
    result[:, :, :base.shape[2]] = base
    position = np.asarray((
        (proposal.left + proposal.right + 1.0) / (2.0 * gray.shape[1]),
        (proposal.top + proposal.bottom + 1.0) / (2.0 * gray.shape[0]),
        proposal.left / gray.shape[1],
        proposal.top / gray.shape[0],
    ), dtype=np.float32)
    result[:, :, -4:] = position[None, None, :]
    if result.shape[2] != 128 + GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V12 geometry contract changed")
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
            continue
        best = max(matches, key=lambda item: item[1])[0]
        accepted.append(1)
        roles.append(ROLE_ORDER.index(scene.truths[best].role))
    return np.asarray(accepted, dtype=np.int64), np.asarray(roles, dtype=np.int64)


def _negative_family(candidate: Component) -> str:
    density = candidate.area / max(1, candidate.width * candidate.height)
    if candidate.height > max(22, candidate.width * 2.2):
        return "vertical-line-or-divider"
    if candidate.width > max(36, candidate.height * 4.0):
        return "horizontal-line-or-bracket"
    if candidate.width <= 20 and candidate.height <= 20:
        return "marker-or-intersection"
    if density >= 0.42:
        return "dense-symbol-or-arrow"
    return "sparse-connected-structure"


def _select_negative_indices(
    candidates: tuple[Component, ...], labels: np.ndarray, cap: int,
) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index in np.flatnonzero(labels == 0).tolist():
        groups.setdefault(_negative_family(candidates[index]), []).append(index)
    selected: list[int] = []
    ordered_families = sorted(groups)
    offset = 0
    while len(selected) < cap:
        added = False
        for family in ordered_families:
            values = groups[family]
            if offset < len(values):
                selected.append(values[offset])
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
    positives = negatives = 0
    family_counts: dict[str, int] = {}
    for scene in scenes:
        candidates = proposals(scene.raster)
        accept, roles = proposal_targets(scene, candidates)
        positive_indices = np.flatnonzero(accept == 1).tolist()
        negative_indices = _select_negative_indices(candidates, accept, negative_cap_per_scene)
        for index in positive_indices + negative_indices:
            encoded = encode_proposal(scene.raster, candidates[index])
            values.append(encoded)
            proposal_labels.append(int(accept[index]))
            role_labels.append(int(roles[index]))
            stream.update(scene.scene_id.encode())
            stream.update(encoded.tobytes(order="C"))
            stream.update(bytes((int(accept[index]), int(roles[index]) + 1)))
            if accept[index] == 0:
                family = _negative_family(candidates[index])
                family_counts[family] = family_counts.get(family, 0) + 1
        positives += len(positive_indices)
        negatives += len(negative_indices)
    if not values or positives == 0 or negatives == 0 or len(family_counts) < 4:
        raise RuntimeError("OCR V12 training examples require positives and diverse structure negatives")
    evidence: dict[str, object] = {
        "scene_count": len(scenes),
        "negative_cap_per_scene": negative_cap_per_scene,
        "negative_sampling": "deterministic-round-robin-by-structure-family-v1",
        "negative_family_counts": dict(sorted(family_counts.items())),
        "proposal_count": len(values),
        "positive_proposal_count": positives,
        "negative_proposal_count": negatives,
        "tensor_label_stream_sha256": stream.hexdigest(),
        "validation_or_public_pixels_used": False,
        "predecessor_fixture_bytes_used": False,
        "v3_fixture_bytes_or_scene_truth_used": False,
    }
    return (
        np.stack(values).astype(np.float32),
        np.asarray(proposal_labels, dtype=np.int64),
        np.asarray(role_labels, dtype=np.int64),
        evidence,
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
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "proposal_count": proposal_count,
        "positive_proposal_count": positive_count,
        "negative_proposal_count": proposal_count - positive_count,
        "role_truth_counts": role_counts,
        "negative_family_counts": dict(sorted(negative_families.items())),
    }


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def save_sealed_public_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    manifest = {
        "schema": "graphreader.ocr-full-scene-proposal-role-fixtures.v1",
        "split": "sealed_public",
        **proposal_summary(scenes),
        "split_fingerprint": split_fingerprint(scenes),
        "synthetic_only": True,
        "cases": [],
    }
    with ZipFile(path, "x") as archive:
        for scene in scenes:
            name = f"fixtures/{scene.scene_id}.png"
            payload = _png_bytes(scene.raster)
            manifest["cases"].append({
                "scene_id": scene.scene_id,
                "renderer_family": scene.renderer_family,
                "degradation_family": scene.degradation_family,
                "source_path": name,
                "source_sha256": sha256(payload).hexdigest(),
                "truths": [
                    {"box": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom], "role": truth.role, "text": truth.text}
                    for truth in scene.truths
                ],
            })
            _zip_write(archive, name, payload)
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
    return {
        "schema": "graphreader.ocr-full-scene-proposal-role-private-manifest.v1",
        **proposal_summary(scenes),
        "split_fingerprint": split_fingerprint(scenes),
        "fixture_archive_sha256": sha256_file(path),
        "synthetic_only": True,
        "production_approval": False,
        "release_eligible": False,
    }


def load_sealed_public_archive(path: Path) -> tuple[SceneSample, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["source_path"])
            if sha256(payload).hexdigest() != case["source_sha256"]:
                raise RuntimeError("OCR V12 sealed fixture hash mismatch")
            raster = np.asarray(Image.open(BytesIO(payload)).convert("L"), dtype=np.uint8).copy()
            truths = tuple(RoleTruth(Box(*truth["box"]), truth["role"], truth["text"]) for truth in case["truths"])
            scenes.append(SceneSample(
                case["scene_id"], "sealed_public", case["renderer_family"],
                case["degradation_family"], raster, truths,
            ))
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR V12 sealed split fingerprint mismatch")
    return result


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_proposal", "load_sealed_public_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene", "save_sealed_public_archive",
    "split_fingerprint", "training_examples",
]
