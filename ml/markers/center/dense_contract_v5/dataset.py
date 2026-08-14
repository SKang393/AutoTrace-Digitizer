# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural dense-contract scenes and deterministic sealed archives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


WIDTH = 192
HEIGHT = 144
TRAIN_SCENE_COUNT = 96
VALIDATION_SCENE_COUNT = 24
PUBLIC_SCENE_COUNT = 32
PUBLIC_DATASET_SEED = 393
MAX_CENTERS = 10
MAX_HARD_NEGATIVES = 16
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
        "seed_offset": 510_393,
        "renderer_family": "dense-offset-series-and-hollow-legend-v5-train",
        "degradation_family": "soft-ink-anisotropic-v5-train",
    },
    "validation": {
        "count": VALIDATION_SCENE_COUNT,
        "seed_offset": 730_393,
        "renderer_family": "dense-step-series-and-twin-bracket-v5-validation",
        "degradation_family": "broken-stroke-low-contrast-v5-validation",
    },
    "sealed_public": {
        "count": PUBLIC_SCENE_COUNT,
        "seed_offset": 910_393,
        "renderer_family": "dense-cross-series-and-nested-legend-v5-public",
        "degradation_family": "mixed-width-quantized-v5-public",
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

    left = 20 + (index % 3)
    top = 12 + ((index // 3) % 2)
    right = 178 - ((index // 5) % 3)
    bottom = 128 - (index % 2)
    ink = int(8 + (index % 5) * 6)
    line_width = 1 + (index % 3 == 0)

    # Axis, tick, and divider masks are deterministic seed evidence.
    raster.line((left, top, left, bottom), fill=ink, width=line_width)
    raster.line((left, bottom, right, bottom), fill=ink, width=line_width)
    seed_artifact.line((left, top, left, bottom), fill=255, width=3)
    seed_artifact.line((left, bottom, right, bottom), fill=255, width=3)
    artifact_truth.line((left, top, left, bottom), fill=255, width=3)
    artifact_truth.line((left, bottom, right, bottom), fill=255, width=3)
    hard: list[tuple[str, float, float]] = [
        ("axis", float(left), float((top + bottom) // 2)),
    ]
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

    divider_x = left + int((right - left) * (0.48 + 0.04 * ((index % 3) - 1)))
    raster.line((divider_x, top + 7, divider_x, bottom - 7), fill=ink + 10, width=1)
    seed_artifact.line((divider_x, top + 7, divider_x, bottom - 7), fill=255, width=3)
    artifact_truth.line((divider_x, top + 7, divider_x, bottom - 7), fill=255, width=3)
    hard.append(("divider", float(divider_x), float(top + 18)))

    # Text-like glyph blocks are always represented by the OCR seed mask.
    for ordinal, (x, y, width) in enumerate(
        (
            (3, 8, 10),
            (4, 40, 8),
            (38, bottom + 6, 14),
            (right - 35, 4, 28),
        )
    ):
        height = 3 + ((index + ordinal) % 3)
        for offset in range(0, width, 3):
            raster.rectangle((x + offset, y, x + offset + 1, y + height), fill=ink + 8)
        text_mask.rectangle((x - 1, y - 1, x + width + 1, y + height + 1), fill=255)
        artifact_truth.rectangle((x - 1, y - 1, x + width + 1, y + height + 1), fill=255)
    hard.append(("text", float(right - 24), 7.0))

    # Legend frame, bracket, arrow, and crossing lines are unseeded artifacts.
    legend_left = right - 42
    legend_top = top + 6
    legend_right = right - 5
    legend_bottom = top + 28
    raster.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=ink + 12, width=1)
    artifact_truth.rectangle((legend_left, legend_top, legend_right, legend_bottom), outline=255, width=3)
    hard.append(("legend", float(legend_right), float((legend_top + legend_bottom) / 2)))

    bracket_y = top + 4 + (index % 3)
    bracket_left = left + 25
    bracket_right = divider_x - 8
    raster.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=ink + 6, width=1)
    raster.line((bracket_left, bracket_y, bracket_left, bracket_y + 6), fill=ink + 6, width=1)
    raster.line((bracket_right, bracket_y, bracket_right, bracket_y + 6), fill=ink + 6, width=1)
    artifact_truth.line((bracket_left, bracket_y, bracket_right, bracket_y), fill=255, width=3)
    artifact_truth.line((bracket_left, bracket_y, bracket_left, bracket_y + 6), fill=255, width=3)
    artifact_truth.line((bracket_right, bracket_y, bracket_right, bracket_y + 6), fill=255, width=3)
    hard.append(("bracket", float((bracket_left + bracket_right) / 2), float(bracket_y)))

    arrow_start = (right - 58, top + 38 + (index % 4))
    arrow_end = (right - 22, top + 48 + ((index // 2) % 4))
    raster.line((*arrow_start, *arrow_end), fill=ink + 5, width=1)
    artifact_truth.line((*arrow_start, *arrow_end), fill=255, width=3)
    head = ((arrow_end[0] - 7, arrow_end[1] - 6), arrow_end, (arrow_end[0] - 8, arrow_end[1] + 2))
    raster.polygon(head, fill=ink + 5)
    artifact_truth.polygon(head, fill=255)
    hard.extend(
        (
            ("arrow_shaft", float((arrow_start[0] + arrow_end[0]) / 2), float((arrow_start[1] + arrow_end[1]) / 2)),
            ("arrowhead", float(arrow_end[0] - 3), float(arrow_end[1])),
        )
    )

    cross_x = right - 62 - (index % 4)
    cross_y = bottom - 20 - ((index // 3) % 5)
    raster.line((cross_x - 12, cross_y, cross_x + 12, cross_y), fill=ink + 12, width=1)
    raster.line((cross_x, cross_y - 10, cross_x, cross_y + 10), fill=ink + 12, width=1)
    artifact_truth.line((cross_x - 12, cross_y, cross_x + 12, cross_y), fill=255, width=3)
    artifact_truth.line((cross_x, cross_y - 10, cross_x, cross_y + 10), fill=255, width=3)
    hard.append(("line_intersection", float(cross_x), float(cross_y)))

    centers: list[tuple[float, float, float]] = []
    point_count = 8 + (index % 3)
    for ordinal in range(point_count):
        x = left + 12 + int(ordinal * (right - left - 32) / max(1, point_count - 1))
        if split == "train":
            y = bottom - 28 - int(15 * np.sin((ordinal + index % 5) * 0.72)) - (ordinal % 2) * 4
        elif split == "validation":
            y = bottom - 24 - ((ordinal * 11 + index * 3) % 46)
        else:
            y = bottom - 26 - ((ordinal * 17 + index * 5 + (ordinal % 3) * 7) % 50)
        y = int(np.clip(y, top + 38, bottom - 10))
        radius = float(3 + ((ordinal + index) % 2))
        centers.append((float(x), float(y), radius))

    connector_points = [(int(x), int(y)) for x, y, _ in centers]
    raster.line(connector_points, fill=ink + 18, width=1)
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

    blur_radius = (index % 4) * 0.18
    if blur_radius:
        raster_image = raster_image.filter(ImageFilter.GaussianBlur(blur_radius))
    pixels = np.asarray(raster_image, dtype=np.float32)
    noise_scale = (1.0, 2.0, 3.0)[index % 3]
    noise = rng.normal(0.0, noise_scale, size=pixels.shape).astype(np.float32)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    text_values = np.asarray(text_mask_image, dtype=np.float32) / 255.0
    seed_artifact_values = np.asarray(seed_artifact_image, dtype=np.float32) / 255.0
    artifact_values = np.asarray(artifact_truth_image, dtype=np.float32) / 255.0
    ink_values = 1.0 - (pixels.astype(np.float32) / 255.0)
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
    scene_id = f"marker-dense-contract-v5-{split}-{index:04d}"
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
    return DenseScene(
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


def tensor_stream_sha256(scenes: tuple[DenseScene, ...]) -> str:
    digest = hashlib.sha256()
    for scene in scenes:
        for value in (scene.tensor, scene.center_target, scene.radius_target, scene.artifact_target):
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "DenseScene",
    "HEIGHT",
    "KIND_TO_INDEX",
    "PROHIBITED_KINDS",
    "PUBLIC_DATASET_SEED",
    "PUBLIC_SCENE_COUNT",
    "SPLITS",
    "TRAIN_SCENE_COUNT",
    "VALIDATION_SCENE_COUNT",
    "WIDTH",
    "archive_bytes",
    "read_archive",
    "render_split",
    "tensor_stream_sha256",
    "write_archive",
]
