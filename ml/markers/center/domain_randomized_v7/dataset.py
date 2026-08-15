# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh domain-randomized and mutually feasible marker scenes for V7."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

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
)


WIDTH = 192
HEIGHT = 144
TRAIN_SCENE_COUNT = 384
VALIDATION_SCENE_COUNT = 96
PUBLIC_SCENE_COUNT = 96
PUBLIC_DATASET_SEED = 7431
MAX_CENTERS = 11
MAX_HARD_NEGATIVES = 16
ARTIFACT_PIXEL_CLEARANCE = 7.0
SPLITS = {
    "train": {
        "count": TRAIN_SCENE_COUNT,
        "seed_offset": 3_110_431,
        "renderer_family": "domain-randomized-layout-mixture-v7-train",
        "degradation_family": "domain-randomized-scan-mixture-v7-train",
    },
    "validation": {
        "count": VALIDATION_SCENE_COUNT,
        "seed_offset": 3_730_431,
        "renderer_family": "heldout-geometry-mixture-v7-validation",
        "degradation_family": "heldout-document-mixture-v7-validation",
    },
    "sealed_public": {
        "count": PUBLIC_SCENE_COUNT,
        "seed_offset": 4_310_431,
        "renderer_family": "truth-hidden-layout-mixture-v7-public",
        "degradation_family": "truth-hidden-capture-mixture-v7-public",
    },
}
RENDERER_FAMILIES = {
    "train": (
        "offset-axis-multishape-v7-train",
        "compact-legend-step-v7-train",
        "wide-bracket-curve-v7-train",
        "dense-annotation-mixed-series-v7-train",
    ),
    "validation": (
        "heldout-staggered-series-v7-validation",
        "heldout-sparse-plateau-v7-validation",
        "heldout-asymmetric-layout-v7-validation",
    ),
    "sealed_public": (
        "truth-hidden-segmented-trajectory-v7-public",
        "truth-hidden-reversed-annotation-v7-public",
    ),
}
DEGRADATION_FAMILIES = {
    "train": (
        "illumination-noise-v7-train",
        "blur-compression-v7-train",
        "stroke-contrast-v7-train",
        "quantized-scanline-v7-train",
        "mixed-mask-imperfection-v7-train",
    ),
    "validation": (
        "heldout-soft-document-v7-validation",
        "heldout-hard-copy-v7-validation",
        "heldout-low-contrast-v7-validation",
    ),
    "sealed_public": (
        "truth-hidden-capture-chain-v7-public",
        "truth-hidden-print-chain-v7-public",
    ),
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _artifact_clear(mask: np.ndarray, x: int, y: int) -> bool:
    radius = int(math.ceil(ARTIFACT_PIXEL_CLEARANCE))
    left = max(0, x - radius)
    right = min(mask.shape[1] - 1, x + radius)
    top = max(0, y - radius)
    bottom = min(mask.shape[0] - 1, y + radius)
    yy, xx = np.mgrid[top : bottom + 1, left : right + 1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= ARTIFACT_PIXEL_CLEARANCE**2
    return not bool(mask[top : bottom + 1, left : right + 1][disk].any())


def _place_center(
    *,
    x: int,
    preferred_y: int,
    plot_top: int,
    plot_bottom: int,
    pre_marker_artifact: np.ndarray,
    hard: list[tuple[str, float, float]],
    existing: list[tuple[float, float, float]],
    scene_identity: str,
) -> tuple[int, int]:
    minimum_y = plot_top + 34
    maximum_y = plot_bottom - 9
    candidates = sorted(range(minimum_y, maximum_y + 1), key=lambda value: (abs(value - preferred_y), value))
    x_candidates = tuple(int(np.clip(x + offset, 8, WIDTH - 9)) for offset in (0, -3, 3, -6, 6, -9, 9))
    for candidate_x in x_candidates:
        for y in candidates:
            if not _artifact_clear(pre_marker_artifact, candidate_x, y):
                continue
            if any(math.hypot(candidate_x - hx, y - hy) <= REQUIRED_DISJOINT_CLEARANCE for _, hx, hy in hard):
                continue
            if any(math.hypot(candidate_x - cx, y - cy) <= MINIMUM_CENTER_SEPARATION for cx, cy, _ in existing):
                continue
            return candidate_x, y
    raise RuntimeError(f"Unable to place a mutually feasible V7 marker center in {scene_identity}")


def _trajectory(split: str, index: int, ordinal: int, count: int, top: int, bottom: int) -> int:
    t = ordinal / max(1, count - 1)
    low = top + 37
    high = bottom - 10
    span = high - low
    if split == "train":
        mode = index % 8
        values = (
            low + span * (0.5 + 0.32 * math.sin(t * math.tau + index * 0.17)),
            low + span * (0.18 if t < 0.32 else 0.72 if t < 0.7 else 0.43),
            low + span * (0.15 + 0.7 * t),
            low + span * (0.82 - 0.64 * t),
            low + span * (0.2 + 0.62 * abs(2 * t - 1)),
            low + span * ((ordinal * 5 + index * 3) % 17) / 17.0,
            low + span * (0.18 + 0.62 * t * t),
            low + span * (0.77 - 0.58 / (1 + math.exp(-8 * (t - 0.5)))),
        )
    elif split == "validation":
        mode = index % 5
        values = (
            low + span * (0.3 + 0.42 * math.sin(t * math.pi * 3 + 0.3)),
            low + span * (0.68 if ordinal % 3 == 0 else 0.24 + 0.32 * t),
            low + span * ((ordinal * 9 + index * 5 + ordinal % 2 * 4) % 23) / 23.0,
            low + span * (0.16 + 0.68 * math.sqrt(t)),
            low + span * (0.22 if t < 0.2 else 0.64 if t < 0.55 else 0.37 if t < 0.8 else 0.75),
        )
    else:
        mode = index % 4
        values = (
            low + span * (0.24 + 0.5 * math.cos(t * math.tau * 1.25 + 0.5)),
            low + span * (0.2 + 0.56 * (3 * t * t - 2 * t * t * t)),
            low + span * ((ordinal * 13 + index * 7 + (ordinal % 3) * 2) % 29) / 29.0,
            low + span * (0.7 if t < 0.28 else 0.26 if t < 0.62 else 0.58),
        )
    return int(round(values[mode]))


def _draw_marker(draw: ImageDraw.ImageDraw, shape: int, x: int, y: int, radius: int, ink: int, filled: bool) -> None:
    if shape == 0:
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, fill=ink if filled else None, outline=ink, width=2)
    elif shape == 1:
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.rectangle(box, fill=ink if filled else None, outline=ink, width=2)
    elif shape == 2:
        points = ((x, y - radius - 1), (x + radius + 1, y), (x, y + radius + 1), (x - radius - 1, y))
        draw.polygon(points, fill=ink if filled else None, outline=ink)
        if not filled:
            inset = max(1, radius - 2)
            draw.polygon(((x, y - inset), (x + inset, y), (x, y + inset), (x - inset, y)), fill=255)
    else:
        points = ((x, y - radius - 1), (x + radius + 1, y + radius), (x - radius - 1, y + radius))
        draw.polygon(points, fill=ink if filled else None, outline=ink)
        if not filled:
            draw.polygon(((x, y - max(1, radius - 2)), (x + max(1, radius - 2), y + radius - 1), (x - max(1, radius - 2), y + radius - 1)), fill=255)


def _degrade(image: Image.Image, split: str, index: int, rng: np.random.Generator) -> np.ndarray:
    mode_count = len(DEGRADATION_FAMILIES[split])
    mode = index % mode_count
    if (index + mode) % 7 == 0:
        image = image.filter(ImageFilter.MinFilter(3))
    elif (index + mode) % 11 == 0:
        image = image.filter(ImageFilter.MaxFilter(3))
    blur = 0.08 + 0.13 * ((index * 3 + mode) % 8)
    if split == "sealed_public":
        blur += 0.07
    image = image.filter(ImageFilter.GaussianBlur(blur))
    pixels = np.asarray(image, dtype=np.float32)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    illumination = (
        ((xx / max(1, WIDTH - 1)) - 0.5) * (4.0 + (index % 5) * 2.5)
        + ((yy / max(1, HEIGHT - 1)) - 0.5) * (-5.0 + (index % 4) * 3.0)
    )
    contrast = 0.58 + ((index * 7 + mode) % 12) * 0.038
    pixels = 255.0 - ((255.0 - pixels) * contrast) + illumination
    gamma = 0.78 + ((index * 5 + mode) % 10) * 0.055
    pixels = 255.0 * np.power(np.clip(pixels, 0, 255) / 255.0, gamma)
    noise_scale = 0.6 + ((index * 11 + mode) % 9) * 0.55
    pixels += rng.normal(0.0, noise_scale, size=pixels.shape).astype(np.float32)
    if (index + mode) % 4 == 0:
        scan_period = 3 + index % 5
        pixels[::scan_period] += 1.5 + index % 4
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if (index + 2 * mode) % 3 == 0:
        quality = 42 + (index * 13) % 48
        buffer = io.BytesIO()
        Image.fromarray(pixels, mode="L").save(buffer, format="JPEG", quality=quality, optimize=False)
        pixels = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("L"), dtype=np.uint8)
    quantization = (2, 3, 4, 6, 8, 12)[(index + mode) % 6]
    if (index + mode) % 2 == 0:
        pixels = np.clip(np.round(pixels.astype(np.float32) / quantization) * quantization, 0, 255).astype(np.uint8)
    return pixels


def _mask_values(image: Image.Image, split: str, index: int, *, artifact: bool) -> np.ndarray:
    selector = (index * (5 if artifact else 3) + (1 if split == "validation" else 2 if split == "sealed_public" else 0)) % 6
    if selector in (1, 4):
        image = image.filter(ImageFilter.MaxFilter(3 if selector == 1 else 5))
    elif selector == 2:
        image = image.filter(ImageFilter.MinFilter(3))
    return np.asarray(image, dtype=np.float32) / 255.0


def _draw_scene(split: str, index: int) -> DenseScene:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    split_config = SPLITS[split]
    if index < 0 or index >= int(split_config["count"]):
        raise IndexError(index)
    seed = int(split_config["seed_offset"]) + index
    rng = np.random.default_rng(seed)
    raster_image = Image.new("L", (WIDTH, HEIGHT), 255)
    text_mask_image = Image.new("L", (WIDTH, HEIGHT), 0)
    seed_artifact_image = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_truth_image = Image.new("L", (WIDTH, HEIGHT), 0)
    raster = ImageDraw.Draw(raster_image)
    text_mask = ImageDraw.Draw(text_mask_image)
    seed_artifact = ImageDraw.Draw(seed_artifact_image)
    artifact_truth = ImageDraw.Draw(artifact_truth_image)

    left = 15 + (index * 3) % 9
    top = 7 + (index * 5) % 8
    right = 184 - (index * 7) % 9
    bottom = 135 - (index * 2) % 8
    ink = int(4 + (index * 7) % 43)
    line_width = 1 + int(index % 5 == 0)
    hard: list[tuple[str, float, float]] = []

    raster.line((left, top, left, bottom), fill=ink, width=line_width)
    raster.line((left, bottom, right, bottom), fill=ink, width=line_width)
    seed_artifact.line((left, top, left, bottom), fill=255, width=3)
    seed_artifact.line((left, bottom, right, bottom), fill=255, width=3)
    artifact_truth.line((left, top, left, bottom), fill=255, width=3)
    artifact_truth.line((left, bottom, right, bottom), fill=255, width=3)
    hard.append(("axis", float(left), float(top + 24)))
    tick_count = 5 + index % 4
    for ordinal in range(1, tick_count + 1):
        x = left + int((right - left) * ordinal / (tick_count + 1))
        y = bottom - int((bottom - top) * ordinal / (tick_count + 1))
        raster.line((x, bottom - 3, x, bottom + 2), fill=ink, width=1)
        raster.line((left - 2, y, left + 3, y), fill=ink, width=1)
        seed_artifact.rectangle((x - 1, bottom - 4, x + 1, bottom + 2), fill=255)
        seed_artifact.rectangle((left - 2, y - 1, left + 3, y + 1), fill=255)
        artifact_truth.rectangle((x - 1, bottom - 4, x + 1, bottom + 2), fill=255)
        artifact_truth.rectangle((left - 2, y - 1, left + 3, y + 1), fill=255)
        if ordinal in (2, tick_count - 1):
            hard.append(("tick", float(x), float(bottom - 1)))

    divider_x = left + int((right - left) * (0.34 + 0.07 * (index % 5)))
    raster.line((divider_x, top + 9, divider_x, bottom - 7), fill=min(80, ink + 11), width=1)
    seed_artifact.line((divider_x, top + 9, divider_x, bottom - 7), fill=255, width=3)
    artifact_truth.line((divider_x, top + 9, divider_x, bottom - 7), fill=255, width=3)
    hard.append(("divider", float(divider_x), float(top + 20)))

    text_specs = (
        (2, 5 + index % 4, 13 + index % 8),
        (2, 31 + index % 9, 10 + (index * 3) % 10),
        (35 + index % 11, bottom + 5, 18 + index % 9),
        (right - 48, 2 + index % 3, 32 + index % 12),
    )
    for ordinal, (x, y, width) in enumerate(text_specs):
        height = 3 + (index + ordinal) % 4
        for offset in range(0, width, 3):
            raster.rectangle((x + offset, y, x + offset + 1, y + height), fill=min(90, ink + 13))
        text_mask.rectangle((x - 2, y - 2, x + width + 2, y + height + 2), fill=255)
        artifact_truth.rectangle((x - 2, y - 2, x + width + 2, y + height + 2), fill=255)
    hard.append(("text", float(right - 27), float(5 + index % 3)))

    legend_width = 31 + index % 13
    legend_left = right - legend_width - 4
    legend_top = top + 7
    legend_right = right - 3
    legend_bottom = top + 27 + index % 5
    raster.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=min(95, ink + 16), width=1)
    artifact_truth.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=255, width=3)
    hard.append(("legend", float(legend_right), float((legend_top + legend_bottom) / 2)))

    bracket_y = top + 3 + index % 4
    bracket_left = left + 19 + index % 9
    bracket_right = min(right - 57, bracket_left + 22 + index % 18)
    raster.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=min(90, ink + 9), width=1)
    raster.line((bracket_left, bracket_y, bracket_left, bracket_y + 6), fill=min(90, ink + 9), width=1)
    raster.line((bracket_right, bracket_y, bracket_right, bracket_y + 6), fill=min(90, ink + 9), width=1)
    artifact_truth.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=255, width=3)
    artifact_truth.line((bracket_left, bracket_y, bracket_left, bracket_y + 6), fill=255, width=3)
    artifact_truth.line((bracket_right, bracket_y, bracket_right, bracket_y + 6), fill=255, width=3)
    hard.append(("bracket", float((bracket_left + bracket_right) / 2), float(bracket_y)))

    reverse_arrow = index % 3 == 0
    arrow_start = (right - 61, top + 39 + index % 7)
    arrow_end = (right - 18, top + 51 + (index * 3) % 9)
    if reverse_arrow:
        arrow_start, arrow_end = arrow_end, arrow_start
    raster.line((*arrow_start, *arrow_end), fill=min(90, ink + 8), width=1)
    artifact_truth.line((*arrow_start, *arrow_end), fill=255, width=3)
    direction = -1 if arrow_end[0] > arrow_start[0] else 1
    head = ((arrow_end[0] + direction * 7, arrow_end[1] - 5), arrow_end, (arrow_end[0] + direction * 7, arrow_end[1] + 5))
    raster.polygon(head, fill=min(90, ink + 8))
    artifact_truth.polygon(head, fill=255)
    hard.append(("arrow_shaft", float((arrow_start[0] + arrow_end[0]) / 2), float((arrow_start[1] + arrow_end[1]) / 2)))
    hard.append(("arrowhead", float(arrow_end[0]), float(arrow_end[1])))

    cross_x = right - 69 - index % 9
    cross_y = bottom - 17 - (index * 3) % 12
    raster.line((cross_x - 13, cross_y - 2, cross_x + 13, cross_y + 2), fill=min(100, ink + 18), width=1)
    raster.line((cross_x - 2, cross_y - 11, cross_x + 2, cross_y + 11), fill=min(100, ink + 18), width=1)
    artifact_truth.line((cross_x - 13, cross_y - 2, cross_x + 13, cross_y + 2), fill=255, width=3)
    artifact_truth.line((cross_x - 2, cross_y - 11, cross_x + 2, cross_y + 11), fill=255, width=3)
    hard.append(("line_intersection", float(cross_x), float(cross_y)))

    pre_marker_artifact = np.asarray(artifact_truth_image, dtype=np.uint8) > 0
    centers: list[tuple[float, float, float]] = []
    point_count = 7 + index % 5
    inner_left = left + 13
    inner_right = right - 13
    for ordinal in range(point_count):
        x = int(round(inner_left + ordinal * (inner_right - inner_left) / max(1, point_count - 1)))
        preferred_y = _trajectory(split, index, ordinal, point_count, top, bottom)
        x, y = _place_center(
            x=x,
            preferred_y=preferred_y,
            plot_top=top,
            plot_bottom=bottom,
            pre_marker_artifact=pre_marker_artifact,
            hard=hard,
            existing=centers,
            scene_identity=f"{split}:{index}:{ordinal}",
        )
        centers.append((float(x), float(y), float(2 + (ordinal + index) % 4)))

    connector_points = [(int(x), int(y)) for x, y, _ in centers]
    raster.line(connector_points, fill=min(100, ink + 19), width=1 + int(index % 6 == 0))
    artifact_truth.line(connector_points, fill=255, width=2 + int(index % 6 == 0))
    for ordinal, (x_value, y_value, radius_value) in enumerate(centers):
        x = int(x_value)
        y = int(y_value)
        radius = int(radius_value)
        _draw_marker(raster, (ordinal + index) % 4, x, y, radius, ink, (ordinal + index) % 2 == 0)
        artifact_truth.ellipse((x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4), fill=0)

    pixels = _degrade(raster_image, split, index, rng)
    text_values = _mask_values(text_mask_image, split, index, artifact=False)
    seed_artifact_values = _mask_values(seed_artifact_image, split, index, artifact=True)
    artifact_values = np.asarray(artifact_truth_image, dtype=np.float32) / 255.0
    ink_values = 1.0 - pixels.astype(np.float32) / 255.0
    tensor = np.stack((ink_values, text_values, seed_artifact_values)).astype(np.float32)

    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    center_target = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    radius_target = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    for x, y, radius in centers:
        distance = (xx - x) ** 2 + (yy - y) ** 2
        gaussian = np.exp(-distance / (2.0 * 1.35**2)).astype(np.float32)
        center_target[0] = np.maximum(center_target[0], gaussian)
        radius_target[0][distance <= 3.0**2] = radius
    artifact_target = artifact_values[np.newaxis, ...].astype(np.float32)
    renderer_family = RENDERER_FAMILIES[split][index % len(RENDERER_FAMILIES[split])]
    degradation_family = DEGRADATION_FAMILIES[split][index % len(DEGRADATION_FAMILIES[split])]
    scene_id = f"marker-domain-randomized-v7-{split}-{index:04d}"
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


def _array_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, value, allow_pickle=False)
    return output.getvalue()


def archive_bytes(scenes: tuple[DenseScene, ...]) -> bytes:
    inputs = np.stack([scene.tensor for scene in scenes]).astype(np.float32)
    centers = np.full((len(scenes), MAX_CENTERS, 3), -1, dtype=np.float32)
    center_counts = np.zeros(len(scenes), dtype=np.int32)
    hard_points = np.full((len(scenes), MAX_HARD_NEGATIVES, 2), -1, dtype=np.float32)
    hard_kinds = np.full((len(scenes), MAX_HARD_NEGATIVES), -1, dtype=np.int16)
    hard_counts = np.zeros(len(scenes), dtype=np.int32)
    for scene_index, scene in enumerate(scenes):
        center_counts[scene_index] = len(scene.centers)
        centers[scene_index, : len(scene.centers)] = np.asarray(scene.centers, dtype=np.float32)
        hard_counts[scene_index] = len(scene.hard_negatives)
        for ordinal, (kind, x, y) in enumerate(scene.hard_negatives):
            hard_points[scene_index, ordinal] = (x, y)
            hard_kinds[scene_index, ordinal] = KIND_TO_INDEX[kind]
    arrays = {
        "artifact_targets": np.stack([scene.artifact_target for scene in scenes]).astype(np.float32),
        "center_counts": center_counts,
        "center_targets": np.stack([scene.center_target for scene in scenes]).astype(np.float32),
        "centers": centers,
        "degradation_families": np.asarray([scene.degradation_family for scene in scenes], dtype="<U80"),
        "ground_truth_sha256": np.asarray([scene.ground_truth_sha256 for scene in scenes], dtype="<U64"),
        "hard_counts": hard_counts,
        "hard_kinds": hard_kinds,
        "hard_points": hard_points,
        "inputs": inputs,
        "radius_targets": np.stack([scene.radius_target for scene in scenes]).astype(np.float32),
        "renderer_families": np.asarray([scene.renderer_family for scene in scenes], dtype="<U80"),
        "scene_ids": np.asarray([scene.scene_id for scene in scenes], dtype="<U80"),
        "source_sha256": np.asarray([scene.source_sha256 for scene in scenes], dtype="<U64"),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _array_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def write_archive(path: Path, scenes: tuple[DenseScene, ...]) -> str:
    value = archive_bytes(scenes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _sha256(value)


__all__ = [
    "ARTIFACT_PIXEL_CLEARANCE",
    "DEGRADATION_FAMILIES",
    "DenseScene",
    "HARD_NEGATIVE_TOLERANCE",
    "HEIGHT",
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
    "archive_bytes",
    "feasibility_summary",
    "read_archive",
    "render_split",
    "validate_scene_feasibility",
    "write_archive",
]
