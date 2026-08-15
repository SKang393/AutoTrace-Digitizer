# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh mutually feasible multi-family dense marker scenes for V6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


WIDTH = 192
HEIGHT = 144
TRAIN_SCENE_COUNT = 192
VALIDATION_SCENE_COUNT = 48
PUBLIC_SCENE_COUNT = 64
PUBLIC_DATASET_SEED = 6393
MAX_CENTERS = 11
MAX_HARD_NEGATIVES = 16
MATCH_TOLERANCE = 5.0
HARD_NEGATIVE_TOLERANCE = 6.0
REQUIRED_DISJOINT_CLEARANCE = MATCH_TOLERANCE + HARD_NEGATIVE_TOLERANCE
MINIMUM_CENTER_SEPARATION = 12.0
ARTIFACT_PIXEL_CLEARANCE = 7.0
PROHIBITED_KINDS = (
    "text",
    "axis",
    "tick",
    "divider",
    "bracket",
    "arrow_shaft",
    "arrowhead",
    "legend",
    "line_intersection",
)
KIND_TO_INDEX = {kind: index for index, kind in enumerate(PROHIBITED_KINDS)}
SPLITS = {
    "train": {
        "count": TRAIN_SCENE_COUNT,
        "seed_offset": 1_310_393,
        "renderer_family": "feasible-multicurve-structure-v6-train",
        "degradation_family": "mixed-blur-noise-contrast-v6-train",
    },
    "validation": {
        "count": VALIDATION_SCENE_COUNT,
        "seed_offset": 1_730_393,
        "renderer_family": "feasible-heldout-piecewise-v6-validation",
        "degradation_family": "heldout-stroke-drop-lowcontrast-v6-validation",
    },
    "sealed_public": {
        "count": PUBLIC_SCENE_COUNT,
        "seed_offset": 2_110_393,
        "renderer_family": "feasible-hidden-composite-v6-public",
        "degradation_family": "hidden-quantized-anisotropic-v6-public",
    },
}


@dataclass(frozen=True)
class DenseScene:
    scene_id: str
    split: str
    renderer_family: str
    degradation_family: str
    raster: np.ndarray
    tensor: np.ndarray
    center_target: np.ndarray
    radius_target: np.ndarray
    artifact_target: np.ndarray
    centers: tuple[tuple[float, float, float], ...]
    hard_negatives: tuple[tuple[str, float, float], ...]
    source_sha256: str
    ground_truth_sha256: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _path_y(split: str, index: int, ordinal: int, count: int, top: int, bottom: int) -> int:
    t = ordinal / max(1, count - 1)
    usable = bottom - top - 52
    middle = top + 38 + usable / 2
    if split == "train":
        mode = index % 6
        values = (
            middle + math.sin(t * math.tau + (index % 7) * 0.31) * usable * 0.34,
            top + 40 + usable * (0.25 if t < 0.36 else 0.72 if t < 0.68 else 0.43),
            middle + ((ordinal + index) % 2 * 2 - 1) * usable * (0.18 + 0.05 * (ordinal % 3)),
            top + 42 + usable * (0.18 + 0.62 * t),
            top + 40 + usable * (0.15 + 0.72 * (2 * abs(t - 0.5))),
            top + 42 + usable * ((ordinal * 7 + index * 3) % 13) / 13.0,
        )
    elif split == "validation":
        mode = index % 4
        values = (
            top + 42 + usable * (0.22 + 0.55 * (t * t)),
            top + 42 + usable * (0.68 if ordinal % 3 == 0 else 0.28 + 0.22 * t),
            middle + math.sin(t * math.pi * 3 + 0.4) * usable * 0.28 + (t - 0.5) * usable * 0.22,
            top + 41 + usable * ((ordinal * 11 + index * 5 + ordinal % 2 * 3) % 17) / 17.0,
        )
    else:
        mode = index % 4
        values = (
            top + 41 + usable * (0.2 + 0.6 / (1 + math.exp(-8 * (t - 0.5)))),
            middle + math.cos(t * math.tau * 1.5 + 0.7) * usable * 0.3,
            top + 42 + usable * (0.2 if t < 0.25 else 0.75 if t < 0.5 else 0.38 if t < 0.75 else 0.62),
            top + 41 + usable * ((ordinal * 17 + index * 7 + (ordinal % 3) * 5) % 23) / 23.0,
        )
    return int(round(values[mode]))


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
    base_x: int,
    base_y: int,
    plot_left: int,
    plot_top: int,
    plot_right: int,
    plot_bottom: int,
    pre_marker_artifact: np.ndarray,
    hard: list[tuple[str, float, float]],
    existing: list[tuple[float, float, float]],
) -> tuple[int, int]:
    x_offsets = (0, -3, 3, -6, 6, -9, 9)
    minimum_y = plot_top + 35
    maximum_y = plot_bottom - 9
    y_candidates = sorted(range(minimum_y, maximum_y + 1), key=lambda value: (abs(value - base_y), value))
    for x_offset in x_offsets:
        x = int(np.clip(base_x + x_offset, plot_left + 9, plot_right - 9))
        for y in y_candidates:
            if not _artifact_clear(pre_marker_artifact, x, y):
                continue
            if any(math.hypot(x - hx, y - hy) <= REQUIRED_DISJOINT_CLEARANCE for _, hx, hy in hard):
                continue
            if any(math.hypot(x - cx, y - cy) <= MINIMUM_CENTER_SEPARATION for cx, cy, _ in existing):
                continue
            return x, y
    raise RuntimeError("Unable to place a mutually feasible marker center")


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

    left = 18 + index % 5
    top = 9 + (index // 5) % 4
    right = 181 - (index // 7) % 4
    bottom = 131 - index % 3
    ink = int(6 + (index % 6) * 5)
    line_width = 1 + int(index % 4 == 0)
    hard: list[tuple[str, float, float]] = []

    raster.line((left, top, left, bottom), fill=ink, width=line_width)
    raster.line((left, bottom, right, bottom), fill=ink, width=line_width)
    seed_artifact.line((left, top, left, bottom), fill=255, width=3)
    seed_artifact.line((left, bottom, right, bottom), fill=255, width=3)
    artifact_truth.line((left, top, left, bottom), fill=255, width=3)
    artifact_truth.line((left, bottom, right, bottom), fill=255, width=3)
    hard.append(("axis", float(left), float((top + bottom) // 2)))
    for ordinal in range(1, 7):
        x = left + int((right - left) * ordinal / 7)
        y = bottom - int((bottom - top) * ordinal / 7)
        raster.line((x, bottom - 3, x, bottom + 2), fill=ink, width=1)
        raster.line((left - 2, y, left + 3, y), fill=ink, width=1)
        seed_artifact.rectangle((x - 1, bottom - 4, x + 1, bottom + 2), fill=255)
        seed_artifact.rectangle((left - 2, y - 1, left + 3, y + 1), fill=255)
        artifact_truth.rectangle((x - 1, bottom - 4, x + 1, bottom + 2), fill=255)
        artifact_truth.rectangle((left - 2, y - 1, left + 3, y + 1), fill=255)
        if ordinal in (2, 5):
            hard.append(("tick", float(x), float(bottom - 2)))

    divider_x = left + int((right - left) * (0.43 + 0.04 * (index % 4)))
    raster.line((divider_x, top + 8, divider_x, bottom - 8), fill=ink + 8, width=1)
    seed_artifact.line((divider_x, top + 8, divider_x, bottom - 8), fill=255, width=3)
    artifact_truth.line((divider_x, top + 8, divider_x, bottom - 8), fill=255, width=3)
    hard.append(("divider", float(divider_x), float(top + 20)))

    text_specs = (
        (3, 7, 11),
        (3, 37 + index % 5, 9),
        (39 + index % 7, bottom + 6, 15),
        (right - 37, 3, 30),
    )
    for ordinal, (x, y, width) in enumerate(text_specs):
        height = 3 + (index + ordinal) % 3
        for offset in range(0, width, 3):
            raster.rectangle((x + offset, y, x + offset + 1, y + height), fill=ink + 9)
        text_mask.rectangle((x - 1, y - 1, x + width + 1, y + height + 1), fill=255)
        artifact_truth.rectangle((x - 1, y - 1, x + width + 1, y + height + 1), fill=255)
    hard.append(("text", float(right - 24), 6.0))

    legend_left = right - 43
    legend_top = top + 7
    legend_right = right - 5
    legend_bottom = top + 29
    raster.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=ink + 10, width=1)
    artifact_truth.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=255, width=3)
    hard.append(("legend", float(legend_right), float((legend_top + legend_bottom) / 2)))

    bracket_y = top + 4 + index % 3
    bracket_left = left + 23
    bracket_right = max(bracket_left + 18, divider_x - 9)
    raster.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink + 7, width=1)
    raster.line((bracket_left, bracket_y, bracket_left, bracket_y + 6), fill=ink + 7, width=1)
    raster.line((bracket_right, bracket_y, bracket_right, bracket_y + 6), fill=ink + 7, width=1)
    artifact_truth.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=255, width=3)
    artifact_truth.line((bracket_left, bracket_y, bracket_left, bracket_y + 6), fill=255, width=3)
    artifact_truth.line((bracket_right, bracket_y, bracket_right, bracket_y + 6), fill=255, width=3)
    hard.append(("bracket", float((bracket_left + bracket_right) / 2), float(bracket_y)))

    arrow_start = (right - 57, top + 40 + index % 3)
    arrow_end = (right - 20, top + 52 + (index // 3) % 4)
    raster.line((*arrow_start, *arrow_end), fill=ink + 6, width=1)
    artifact_truth.line((*arrow_start, *arrow_end), fill=255, width=3)
    head = ((arrow_end[0] - 7, arrow_end[1] - 6), arrow_end, (arrow_end[0] - 8, arrow_end[1] + 3))
    raster.polygon(head, fill=ink + 6)
    artifact_truth.polygon(head, fill=255)
    hard.append(("arrow_shaft", float((arrow_start[0] + arrow_end[0]) / 2), float((arrow_start[1] + arrow_end[1]) / 2)))
    hard.append(("arrowhead", float(arrow_end[0] - 3), float(arrow_end[1])))

    cross_x = right - 63 - index % 5
    cross_y = bottom - 19 - (index // 4) % 6
    raster.line((cross_x - 12, cross_y, cross_x + 12, cross_y), fill=ink + 11, width=1)
    raster.line((cross_x, cross_y - 10, cross_x, cross_y + 10), fill=ink + 11, width=1)
    artifact_truth.line((cross_x - 12, cross_y, cross_x + 12, cross_y), fill=255, width=3)
    artifact_truth.line((cross_x, cross_y - 10, cross_x, cross_y + 10), fill=255, width=3)
    hard.append(("line_intersection", float(cross_x), float(cross_y)))

    pre_marker_artifact = np.asarray(artifact_truth_image, dtype=np.uint8) > 0
    centers: list[tuple[float, float, float]] = []
    point_count = 8 + index % 3
    for ordinal in range(point_count):
        base_x = left + 13 + int(ordinal * (right - left - 30) / max(1, point_count - 1))
        base_y = _path_y(split, index, ordinal, point_count, top, bottom)
        x, y = _place_center(
            base_x=base_x,
            base_y=base_y,
            plot_left=left,
            plot_top=top,
            plot_right=right,
            plot_bottom=bottom,
            pre_marker_artifact=pre_marker_artifact,
            hard=hard,
            existing=centers,
        )
        centers.append((float(x), float(y), float(3 + (ordinal + index) % 2)))

    connector_points = [(int(x), int(y)) for x, y, _ in centers]
    raster.line(connector_points, fill=ink + 17, width=1)
    artifact_truth.line(connector_points, fill=255, width=2)
    for ordinal, (x_value, y_value, radius) in enumerate(centers):
        x = int(x_value)
        y = int(y_value)
        r = int(radius)
        if (ordinal + index) % 2:
            raster.ellipse((x - r, y - r, x + r, y + r), fill=ink)
        else:
            raster.ellipse((x - r, y - r, x + r, y + r), outline=ink, width=2)
            raster.ellipse((x - max(1, r - 2), y - max(1, r - 2), x + max(1, r - 2), y + max(1, r - 2)), fill=255)
        artifact_truth.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill=0)

    if split == "train":
        blur_radius = (index % 5) * 0.12
        noise_scale = 0.7 + (index % 4) * 0.55
        contrast = 0.82 + (index % 5) * 0.035
    elif split == "validation":
        blur_radius = 0.14 + (index % 4) * 0.13
        noise_scale = 1.25 + (index % 4) * 0.6
        contrast = 0.78 + (index % 4) * 0.04
    else:
        blur_radius = 0.08 + (index % 6) * 0.1
        noise_scale = 1.0 + (index % 5) * 0.58
        contrast = 0.76 + (index % 6) * 0.038
    if blur_radius:
        raster_image = raster_image.filter(ImageFilter.GaussianBlur(blur_radius))
    pixels = np.asarray(raster_image, dtype=np.float32)
    pixels = 255.0 - ((255.0 - pixels) * contrast)
    noise = rng.normal(0.0, noise_scale, size=pixels.shape).astype(np.float32)
    pixels = np.clip(pixels + noise, 0, 255)
    if split == "sealed_public":
        pixels = np.round(pixels / 2.0) * 2.0
    pixels = pixels.astype(np.uint8)
    text_values = np.asarray(text_mask_image, dtype=np.float32) / 255.0
    seed_artifact_values = np.asarray(seed_artifact_image, dtype=np.float32) / 255.0
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
    scene_id = f"marker-feasible-dense-v6-{split}-{index:04d}"
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
        str(split_config["renderer_family"]),
        str(split_config["degradation_family"]),
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


def validate_scene_feasibility(scene: DenseScene) -> None:
    centers = np.asarray([(x, y) for x, y, _ in scene.centers], dtype=np.float32)
    hard_points = np.asarray([(x, y) for _, x, y in scene.hard_negatives], dtype=np.float32)
    distances = np.sqrt(((centers[:, None, :] - hard_points[None, :, :]) ** 2).sum(axis=2))
    if float(distances.min()) <= REQUIRED_DISJOINT_CLEARANCE:
        raise RuntimeError("Marker truth and prohibited acceptance regions overlap")
    if len(centers) > 1:
        center_distances = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        center_distances += np.eye(len(centers), dtype=np.float32) * 1_000_000.0
        if float(center_distances.min()) <= MINIMUM_CENTER_SEPARATION:
            raise RuntimeError("Marker truths violate the fixed center separation")
    for x_value, y_value, _ in scene.centers:
        x = int(round(x_value))
        y = int(round(y_value))
        if scene.artifact_target[0, y, x] >= 0.5 or scene.tensor[1, y, x] >= 0.5 or scene.tensor[2, y, x] >= 0.5:
            raise RuntimeError("Marker truth intersects an artifact, text, or seed mask")
    for _, x_value, y_value in scene.hard_negatives:
        x = int(round(x_value))
        y = int(round(y_value))
        if scene.artifact_target[0, y, x] < 0.5:
            raise RuntimeError("Prohibited hard point is absent from artifact truth")


def render_split(split: str) -> tuple[DenseScene, ...]:
    return tuple(_draw_scene(split, index) for index in range(int(SPLITS[split]["count"])))


def feasibility_summary(scenes: tuple[DenseScene, ...]) -> dict[str, object]:
    minimum_truth_hard = math.inf
    minimum_center_distance = math.inf
    for scene in scenes:
        validate_scene_feasibility(scene)
        centers = np.asarray([(x, y) for x, y, _ in scene.centers], dtype=np.float32)
        hard_points = np.asarray([(x, y) for _, x, y in scene.hard_negatives], dtype=np.float32)
        minimum_truth_hard = min(
            minimum_truth_hard,
            float(np.sqrt(((centers[:, None, :] - hard_points[None, :, :]) ** 2).sum(axis=2)).min()),
        )
        if len(centers) > 1:
            distances = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
            distances += np.eye(len(centers), dtype=np.float32) * 1_000_000.0
            minimum_center_distance = min(minimum_center_distance, float(distances.min()))
    return {
        "scene_count": len(scenes),
        "truth_center_count": sum(len(scene.centers) for scene in scenes),
        "hard_negative_count": sum(len(scene.hard_negatives) for scene in scenes),
        "required_disjoint_clearance_px": REQUIRED_DISJOINT_CLEARANCE,
        "minimum_truth_to_hard_negative_distance_px": minimum_truth_hard,
        "minimum_center_separation_px": minimum_center_distance,
        "truth_hard_acceptance_overlap_count": 0,
        "truth_mask_conflict_count": 0,
        "hard_point_missing_from_artifact_truth_count": 0,
    }


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
    for index, scene in enumerate(scenes):
        center_counts[index] = len(scene.centers)
        centers[index, : len(scene.centers)] = np.asarray(scene.centers, dtype=np.float32)
        hard_counts[index] = len(scene.hard_negatives)
        for ordinal, (kind, x, y) in enumerate(scene.hard_negatives):
            hard_points[index, ordinal] = (x, y)
            hard_kinds[index, ordinal] = KIND_TO_INDEX[kind]
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


def read_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


__all__ = [
    "ARTIFACT_PIXEL_CLEARANCE",
    "DenseScene",
    "HARD_NEGATIVE_TOLERANCE",
    "HEIGHT",
    "KIND_TO_INDEX",
    "MATCH_TOLERANCE",
    "MINIMUM_CENTER_SEPARATION",
    "PROHIBITED_KINDS",
    "PUBLIC_DATASET_SEED",
    "PUBLIC_SCENE_COUNT",
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
