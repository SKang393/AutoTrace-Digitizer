# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh feasible scenes for the mask-consensus marker-center V8 defect class."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ml.markers.center.feasible_dense_v6.dataset import (
    DenseScene,
    HARD_NEGATIVE_TOLERANCE,
    KIND_TO_INDEX,
    MATCH_TOLERANCE,
    MINIMUM_CENTER_SEPARATION,
    PROHIBITED_KINDS,
    REQUIRED_DISJOINT_CLEARANCE,
    feasibility_summary,
    read_archive,
    validate_scene_feasibility,
    write_archive,
)


WIDTH = 192
HEIGHT = 144
TRAIN_SCENE_COUNT = 512
VALIDATION_SCENE_COUNT = 128
PUBLIC_SCENE_COUNT = 160
PUBLIC_DATASET_SEED = 8117
SPLITS = {
    "train": {
        "count": TRAIN_SCENE_COUNT,
        "seed_offset": 8_117_031,
        "renderer_family": "mask-consensus-layout-bank-v8-train",
        "degradation_family": "mask-consensus-capture-bank-v8-train",
    },
    "validation": {
        "count": VALIDATION_SCENE_COUNT,
        "seed_offset": 8_917_031,
        "renderer_family": "mask-consensus-heldout-layout-v8-validation",
        "degradation_family": "mask-consensus-heldout-capture-v8-validation",
    },
    "sealed_public": {
        "count": PUBLIC_SCENE_COUNT,
        "seed_offset": 9_717_031,
        "renderer_family": "mask-consensus-hidden-layout-v8-public",
        "degradation_family": "mask-consensus-hidden-capture-v8-public",
    },
}
RENDERER_FAMILIES = {
    "train": (
        "split-rise-mixed-glyph-v8-train",
        "dual-plateau-compact-legend-v8-train",
        "reversal-bracket-diamond-v8-train",
        "staggered-phase-triangle-v8-train",
        "asymmetric-wave-square-v8-train",
    ),
    "validation": (
        "heldout-offset-cycle-v8-validation",
        "heldout-late-step-v8-validation",
        "heldout-wide-echo-v8-validation",
    ),
    "sealed_public": (
        "hidden-broken-slope-v8-public",
        "hidden-reverse-tail-v8-public",
        "hidden-multiphase-fan-v8-public",
    ),
}
DEGRADATION_FAMILIES = {
    "train": (
        "anisotropic-ink-v8-train",
        "soft-copy-v8-train",
        "quantized-scan-v8-train",
        "low-contrast-noise-v8-train",
        "mask-underfill-v8-train",
        "mask-overfill-v8-train",
    ),
    "validation": (
        "heldout-print-blur-v8-validation",
        "heldout-gray-cast-v8-validation",
        "heldout-compressed-scan-v8-validation",
    ),
    "sealed_public": (
        "hidden-photocopy-chain-v8-public",
        "hidden-display-capture-v8-public",
        "hidden-document-noise-v8-public",
    ),
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _marker(
    draw: ImageDraw.ImageDraw,
    *,
    shape: int,
    x: int,
    y: int,
    radius: int,
    ink: int,
    filled: bool,
) -> None:
    if shape == 0:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=ink if filled else None,
            outline=ink,
            width=2,
        )
    elif shape == 1:
        draw.rectangle(
            (x - radius, y - radius, x + radius, y + radius),
            fill=ink if filled else None,
            outline=ink,
            width=2,
        )
    elif shape == 2:
        outer = (
            (x, y - radius - 1),
            (x + radius + 1, y),
            (x, y + radius + 1),
            (x - radius - 1, y),
        )
        draw.polygon(outer, fill=ink if filled else None, outline=ink)
        if not filled:
            inner = max(1, radius - 2)
            draw.polygon(
                ((x, y - inner), (x + inner, y), (x, y + inner), (x - inner, y)),
                fill=255,
            )
    else:
        outer = (
            (x, y - radius - 1),
            (x + radius + 1, y + radius),
            (x - radius - 1, y + radius),
        )
        draw.polygon(outer, fill=ink if filled else None, outline=ink)
        if not filled:
            inner = max(1, radius - 2)
            draw.polygon(
                ((x, y - inner), (x + inner, y + inner), (x - inner, y + inner)),
                fill=255,
            )


def _trajectory(split: str, index: int, ordinal: int, count: int, top: int, bottom: int) -> int:
    t = ordinal / max(1, count - 1)
    low = top + 38
    high = bottom - 11
    span = high - low
    if split == "train":
        mode = index % 9
        values = (
            low + span * (0.5 + 0.34 * math.sin(t * math.tau + index * 0.11)),
            low + span * (0.16 + 0.68 * t),
            low + span * (0.82 - 0.63 * t),
            low + span * (0.22 if t < 0.31 else 0.69 if t < 0.66 else 0.41),
            low + span * (0.19 + 0.66 * abs(2 * t - 1)),
            low + span * ((ordinal * 7 + index * 5) % 19) / 19.0,
            low + span * (0.15 + 0.72 * t * t),
            low + span * (0.78 - 0.61 * math.sqrt(t)),
            low + span * (0.27 + 0.48 * math.sin(t * math.pi * 3 + 0.7)),
        )
    elif split == "validation":
        mode = index % 6
        values = (
            low + span * (0.2 + 0.59 * (3 * t * t - 2 * t * t * t)),
            low + span * (0.72 if ordinal % 3 == 0 else 0.2 + 0.38 * t),
            low + span * ((ordinal * 11 + index * 3 + ordinal % 2 * 5) % 29) / 29.0,
            low + span * (0.18 if t < 0.2 else 0.65 if t < 0.56 else 0.34 if t < 0.82 else 0.73),
            low + span * (0.3 + 0.43 * math.cos(t * math.tau * 1.25 + 0.4)),
            low + span * (0.18 + 0.68 / (1 + math.exp(-8 * (t - 0.48)))),
        )
    else:
        mode = index % 6
        values = (
            low + span * (0.23 + 0.5 * math.cos(t * math.tau * 1.5 + 0.2)),
            low + span * (0.18 + 0.62 * math.sqrt(t)),
            low + span * ((ordinal * 17 + index * 7 + ordinal % 3 * 3) % 31) / 31.0,
            low + span * (0.73 if t < 0.27 else 0.24 if t < 0.63 else 0.6),
            low + span * (0.24 + 0.53 * abs(math.sin(t * math.pi * 1.5))),
            low + span * (0.76 - 0.58 * (3 * t * t - 2 * t * t * t)),
        )
    return int(round(values[mode]))


def _degrade(image: Image.Image, split: str, index: int, rng: np.random.Generator) -> np.ndarray:
    family_count = len(DEGRADATION_FAMILIES[split])
    mode = index % family_count
    if (index + mode) % 13 == 0:
        image = image.filter(ImageFilter.MinFilter(3))
    elif (index + 2 * mode) % 17 == 0:
        image = image.filter(ImageFilter.MaxFilter(3))
    blur = 0.06 + 0.09 * ((index * 5 + mode) % 10)
    if split == "sealed_public":
        blur += 0.08
    image = image.filter(ImageFilter.GaussianBlur(blur))
    pixels = np.asarray(image, dtype=np.float32)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    illumination = (
        ((xx / (WIDTH - 1)) - 0.5) * (-8.0 + (index % 9) * 2.0)
        + ((yy / (HEIGHT - 1)) - 0.5) * (7.0 - (index % 7) * 2.1)
    )
    contrast = 0.56 + ((index * 7 + mode) % 14) * 0.033
    pixels = 255.0 - ((255.0 - pixels) * contrast) + illumination
    gamma = 0.76 + ((index * 3 + mode) % 12) * 0.048
    pixels = 255.0 * np.power(np.clip(pixels, 0, 255) / 255.0, gamma)
    pixels += rng.normal(0.0, 0.5 + ((index * 11 + mode) % 11) * 0.48, pixels.shape)
    if (index + mode) % 4 == 0:
        pixels[:: 3 + index % 5] += 1.0 + index % 5
    quantum = (2, 3, 4, 5, 7, 9, 12)[(index + mode) % 7]
    pixels = np.round(np.clip(pixels, 0, 255) / quantum) * quantum
    return np.clip(pixels, 0, 255).astype(np.uint8)


def _mask_plane(image: Image.Image, split: str, index: int, *, artifact: bool) -> np.ndarray:
    selector = (index * (7 if artifact else 5) + (1 if split == "validation" else 2 if split == "sealed_public" else 0)) % 7
    if selector == 1:
        image = image.filter(ImageFilter.MaxFilter(3))
    elif selector == 2:
        image = image.filter(ImageFilter.MinFilter(3))
    elif selector == 3:
        image = image.filter(ImageFilter.GaussianBlur(0.7))
    values = np.asarray(image, dtype=np.float32) / 255.0
    if selector == 4:
        values *= 0.72
    elif selector == 5:
        values = np.clip(values * 1.18, 0.0, 1.0)
    return values.astype(np.float32)


def _clear_at(mask: np.ndarray, x: int, y: int, radius: float) -> bool:
    extent = int(math.ceil(radius))
    left = max(0, x - extent)
    right = min(WIDTH - 1, x + extent)
    top = max(0, y - extent)
    bottom = min(HEIGHT - 1, y + extent)
    yy, xx = np.mgrid[top : bottom + 1, left : right + 1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
    return not bool((mask[top : bottom + 1, left : right + 1][disk] >= 0.35).any())


def _place_center(
    *,
    preferred_x: int,
    preferred_y: int,
    top: int,
    bottom: int,
    artifact_truth: np.ndarray,
    text_values: np.ndarray,
    seed_values: np.ndarray,
    hard: list[tuple[str, float, float]],
    existing: list[tuple[float, float, float]],
    identity: str,
) -> tuple[int, int]:
    y_candidates = sorted(
        range(top + 36, bottom - 8),
        key=lambda value: (abs(value - preferred_y), value),
    )
    for x_offset in (0, -3, 3, -6, 6, -9, 9, -12, 12):
        x = int(np.clip(preferred_x + x_offset, 8, WIDTH - 9))
        for y in y_candidates:
            if not _clear_at(artifact_truth, x, y, 7.0):
                continue
            if not _clear_at(text_values, x, y, 5.0) or not _clear_at(seed_values, x, y, 5.0):
                continue
            if any(math.hypot(x - hx, y - hy) <= REQUIRED_DISJOINT_CLEARANCE for _, hx, hy in hard):
                continue
            if any(math.hypot(x - cx, y - cy) <= MINIMUM_CENTER_SEPARATION for cx, cy, _ in existing):
                continue
            return x, y
    raise RuntimeError(f"Unable to place feasible V8 center: {identity}")


def _draw_scene(split: str, index: int) -> DenseScene:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    count = int(SPLITS[split]["count"])
    if index < 0 or index >= count:
        raise IndexError(index)
    rng = np.random.default_rng(int(SPLITS[split]["seed_offset"]) + index)
    raster_image = Image.new("L", (WIDTH, HEIGHT), 255)
    text_image = Image.new("L", (WIDTH, HEIGHT), 0)
    seed_image = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_image = Image.new("L", (WIDTH, HEIGHT), 0)
    raster = ImageDraw.Draw(raster_image)
    text = ImageDraw.Draw(text_image)
    seed = ImageDraw.Draw(seed_image)
    artifact = ImageDraw.Draw(artifact_image)
    left = 13 + (index * 5) % 11
    top = 6 + (index * 7) % 9
    right = 186 - (index * 3) % 11
    bottom = 136 - (index * 2) % 9
    ink = 3 + (index * 11) % 49
    width = 1 + int(index % 6 == 0)
    hard: list[tuple[str, float, float]] = []

    raster.line((left, top, left, bottom), fill=ink, width=width)
    raster.line((left, bottom, right, bottom), fill=ink, width=width)
    seed.line((left, top, left, bottom), fill=255, width=3)
    seed.line((left, bottom, right, bottom), fill=255, width=3)
    artifact.line((left, top, left, bottom), fill=255, width=3)
    artifact.line((left, bottom, right, bottom), fill=255, width=3)
    hard.append(("axis", float(left), float(top + 23)))
    tick_count = 5 + index % 5
    for ordinal in range(1, tick_count + 1):
        x = left + int((right - left) * ordinal / (tick_count + 1))
        y = bottom - int((bottom - top) * ordinal / (tick_count + 1))
        raster.line((x, bottom - 4, x, bottom + 2), fill=ink, width=1)
        raster.line((left - 2, y, left + 4, y), fill=ink, width=1)
        seed.rectangle((x - 1, bottom - 4, x + 1, bottom + 2), fill=255)
        seed.rectangle((left - 2, y - 1, left + 4, y + 1), fill=255)
        artifact.rectangle((x - 1, bottom - 4, x + 1, bottom + 2), fill=255)
        artifact.rectangle((left - 2, y - 1, left + 4, y + 1), fill=255)
    hard.append(("tick", float(left + (right - left) // 3), float(bottom - 1)))

    divider_x = left + int((right - left) * (0.29 + 0.09 * (index % 6)))
    raster.line((divider_x, top + 9, divider_x, bottom - 8), fill=min(98, ink + 13), width=1)
    seed.line((divider_x, top + 9, divider_x, bottom - 8), fill=255, width=3)
    artifact.line((divider_x, top + 9, divider_x, bottom - 8), fill=255, width=3)
    hard.append(("divider", float(divider_x), float(top + 24)))

    text_specs = (
        (1, 8 + index % 8, 14 + index % 9),
        (2, 34 + (index * 3) % 11, 12 + index % 8),
        (30 + index % 17, bottom + 5, 18 + index % 13),
        (right - 52, 2 + index % 4, 36 + index % 13),
    )
    for ordinal, (x, y, text_width) in enumerate(text_specs):
        text_height = 3 + (index + ordinal) % 5
        for offset in range(0, text_width, 3):
            raster.rectangle((x + offset, y, x + offset + 1, y + text_height), fill=min(105, ink + 19))
        text.rectangle((x - 2, y - 2, x + text_width + 2, y + text_height + 2), fill=255)
        artifact.rectangle((x - 2, y - 2, x + text_width + 2, y + text_height + 2), fill=255)
    hard.append(("text", float(right - 29), float(5 + index % 4)))

    legend_left = right - 49 - index % 8
    legend_top = top + 8
    legend_right = right - 3
    legend_bottom = top + 29 + index % 7
    raster.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=min(105, ink + 18), width=1)
    artifact.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=255, width=3)
    hard.append(("legend", float(legend_right), float((legend_top + legend_bottom) / 2)))

    bracket_y = top + 3 + index % 5
    bracket_left = left + 17 + index % 11
    bracket_right = min(right - 61, bracket_left + 24 + index % 21)
    raster.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=min(100, ink + 14), width=1)
    raster.line((bracket_left, bracket_y, bracket_left, bracket_y + 7), fill=min(100, ink + 14), width=1)
    raster.line((bracket_right, bracket_y, bracket_right, bracket_y + 7), fill=min(100, ink + 14), width=1)
    artifact.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=255, width=3)
    artifact.line((bracket_left, bracket_y, bracket_left, bracket_y + 7), fill=255, width=3)
    artifact.line((bracket_right, bracket_y, bracket_right, bracket_y + 7), fill=255, width=3)
    hard.append(("bracket", float((bracket_left + bracket_right) / 2), float(bracket_y)))

    arrow_start = (right - 67, top + 38 + index % 11)
    arrow_end = (right - 17, top + 51 + (index * 5) % 13)
    if index % 4 == 0:
        arrow_start, arrow_end = arrow_end, arrow_start
    raster.line((*arrow_start, *arrow_end), fill=min(105, ink + 17), width=1)
    artifact.line((*arrow_start, *arrow_end), fill=255, width=3)
    sign = -1 if arrow_end[0] > arrow_start[0] else 1
    head = (
        (arrow_end[0] + sign * 8, arrow_end[1] - 6),
        arrow_end,
        (arrow_end[0] + sign * 8, arrow_end[1] + 6),
    )
    raster.polygon(head, fill=min(105, ink + 17))
    artifact.polygon(head, fill=255)
    hard.append(("arrow_shaft", float((arrow_start[0] + arrow_end[0]) / 2), float((arrow_start[1] + arrow_end[1]) / 2)))
    hard.append(("arrowhead", float(arrow_end[0]), float(arrow_end[1])))

    cross_x = right - 73 - index % 12
    cross_y = bottom - 17 - (index * 3) % 15
    raster.line((cross_x - 14, cross_y - 3, cross_x + 14, cross_y + 3), fill=min(112, ink + 21), width=1)
    raster.line((cross_x - 3, cross_y - 12, cross_x + 3, cross_y + 12), fill=min(112, ink + 21), width=1)
    artifact.line((cross_x - 14, cross_y - 3, cross_x + 14, cross_y + 3), fill=255, width=3)
    artifact.line((cross_x - 3, cross_y - 12, cross_x + 3, cross_y + 12), fill=255, width=3)
    hard.append(("line_intersection", float(cross_x), float(cross_y)))

    text_values = _mask_plane(text_image, split, index, artifact=False)
    seed_values = _mask_plane(seed_image, split, index, artifact=True)
    pre_marker_artifact = np.asarray(artifact_image, dtype=np.float32) / 255.0
    centers: list[tuple[float, float, float]] = []
    point_count = 8 + index % 4
    inner_left = left + 14
    inner_right = right - 14
    for ordinal in range(point_count):
        preferred_x = int(round(inner_left + ordinal * (inner_right - inner_left) / max(1, point_count - 1)))
        preferred_y = _trajectory(split, index, ordinal, point_count, top, bottom)
        x, y = _place_center(
            preferred_x=preferred_x,
            preferred_y=preferred_y,
            top=top,
            bottom=bottom,
            artifact_truth=pre_marker_artifact,
            text_values=text_values,
            seed_values=seed_values,
            hard=hard,
            existing=centers,
            identity=f"{split}:{index}:{ordinal}",
        )
        centers.append((float(x), float(y), float(2 + (ordinal + index) % 4)))

    connector_points = [(int(x), int(y)) for x, y, _ in centers]
    raster.line(connector_points, fill=min(112, ink + 23), width=1 + int(index % 7 == 0))
    artifact.line(connector_points, fill=255, width=2 + int(index % 7 == 0))
    for ordinal, (x_value, y_value, radius_value) in enumerate(centers):
        x = int(x_value)
        y = int(y_value)
        radius = int(radius_value)
        _marker(
            raster,
            shape=(ordinal * 3 + index) % 4,
            x=x,
            y=y,
            radius=radius,
            ink=ink,
            filled=(ordinal + index) % 2 == 0,
        )
        artifact.ellipse(
            (x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4),
            fill=0,
        )

    pixels = _degrade(raster_image, split, index, rng)
    artifact_values = np.asarray(artifact_image, dtype=np.float32) / 255.0
    ink_values = 1.0 - pixels.astype(np.float32) / 255.0
    tensor = np.stack((ink_values, text_values, seed_values)).astype(np.float32)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    center_target = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    radius_target = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    for x, y, radius in centers:
        distance = (xx - x) ** 2 + (yy - y) ** 2
        center_target[0] = np.maximum(
            center_target[0],
            np.exp(-distance / (2.0 * 1.35**2)).astype(np.float32),
        )
        radius_target[0][distance <= 3.0**2] = radius
    artifact_target = artifact_values[np.newaxis, ...].astype(np.float32)
    renderer_family = RENDERER_FAMILIES[split][index % len(RENDERER_FAMILIES[split])]
    degradation_family = DEGRADATION_FAMILIES[split][index % len(DEGRADATION_FAMILIES[split])]
    scene_id = f"marker-mask-consensus-v8-{split}-{index:04d}"
    source_sha256 = _sha256(pixels.tobytes(order="C"))
    ground_truth_sha256 = _sha256(
        _canonical_json_bytes(
            {
                "artifact_target_sha256": _sha256(artifact_target.tobytes(order="C")),
                "centers": centers,
                "hard_negatives": hard,
                "scene_id": scene_id,
            }
        )
    )
    scene = DenseScene(
        scene_id,
        split,
        renderer_family,
        degradation_family,
        pixels,
        tensor,
        center_target,
        radius_target,
        artifact_target,
        tuple(centers),
        tuple(hard),
        source_sha256,
        ground_truth_sha256,
    )
    validate_scene_feasibility(scene)
    return scene


def render_split(split: str) -> tuple[DenseScene, ...]:
    return tuple(_draw_scene(split, index) for index in range(int(SPLITS[split]["count"])))


__all__ = [
    "DEGRADATION_FAMILIES",
    "HEIGHT",
    "HARD_NEGATIVE_TOLERANCE",
    "KIND_TO_INDEX",
    "MATCH_TOLERANCE",
    "MINIMUM_CENTER_SEPARATION",
    "PROHIBITED_KINDS",
    "PUBLIC_DATASET_SEED",
    "PUBLIC_SCENE_COUNT",
    "RENDERER_FAMILIES",
    "REQUIRED_DISJOINT_CLEARANCE",
    "SPLITS",
    "TRAIN_SCENE_COUNT",
    "VALIDATION_SCENE_COUNT",
    "WIDTH",
    "feasibility_summary",
    "read_archive",
    "render_split",
    "validate_scene_feasibility",
    "write_archive",
]
