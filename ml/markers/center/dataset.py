# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic family-disjoint procedural marker-center dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch


ARTIFACT_KINDS = (
    "text",
    "tick",
    "arrowhead",
    "divider",
    "bracket",
    "line_intersection",
    "axis",
    "legend",
)
SPLIT_FAMILIES = {
    "train": ("vector_clean", "print_speckle"),
    "validation": ("scan_gaussian",),
    "test": ("golden_dense", "golden_open_touch", "golden_same_column"),
}
DEGRADATION_BY_FAMILY = {
    "vector_clean": "none",
    "print_speckle": "salt_pepper",
    "scan_gaussian": "gaussian_blur",
    "golden_dense": "contrast_compression",
    "golden_open_touch": "median_filter",
    "golden_same_column": "horizontal_banding",
}
DATASET_REVISION = "marker-center-procedural-v2"


@dataclass(frozen=True)
class Scene:
    scene_id: str
    split: str
    family: str
    degradation: str
    seed: int
    tensor: torch.Tensor
    center_target: torch.Tensor
    radius_target: torch.Tensor
    artifact_target: torch.Tensor
    centers: tuple[tuple[float, float], ...]
    radii: tuple[float, ...]
    hard_negatives: tuple[tuple[str, float, float], ...]


def _draw_marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, index: int) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    if index % 3 == 0:
        draw.ellipse(box, fill=0)
    elif index % 3 == 1:
        draw.ellipse(box, fill=255, outline=0, width=2)
    else:
        draw.rectangle(box, fill=0 if index % 2 else 255, outline=0, width=2)


def _artifact_geometry(
    draw: ImageDraw.ImageDraw,
    mask: ImageDraw.ImageDraw,
    kind: str,
    x: int,
    y: int,
) -> None:
    if kind == "text":
        draw.text((x - 5, y - 5), "Ab", fill=0)
        mask.rectangle((x - 7, y - 7, x + 13, y + 7), fill=255)
    elif kind == "tick":
        draw.line((x - 5, y, x + 5, y), fill=0, width=2)
        mask.rectangle((x - 7, y - 3, x + 7, y + 3), fill=255)
    elif kind == "arrowhead":
        draw.polygon(((x - 5, y - 4), (x + 5, y), (x - 5, y + 4)), fill=0)
        mask.ellipse((x - 7, y - 7, x + 7, y + 7), fill=255)
    elif kind == "divider":
        for yy in range(y - 9, y + 10, 4):
            draw.line((x, yy, x, yy + 1), fill=0, width=2)
        mask.rectangle((x - 3, y - 12, x + 3, y + 12), fill=255)
    elif kind == "bracket":
        draw.line((x - 5, y - 7, x - 5, y + 7, x + 4, y + 7), fill=0, width=2)
        mask.rectangle((x - 8, y - 10, x + 7, y + 10), fill=255)
    elif kind == "line_intersection":
        draw.line((x - 6, y - 6, x + 6, y + 6), fill=0, width=2)
        draw.line((x - 6, y + 6, x + 6, y - 6), fill=0, width=2)
        mask.ellipse((x - 8, y - 8, x + 8, y + 8), fill=255)
    elif kind == "axis":
        draw.line((x - 10, y + 7, x + 10, y + 7), fill=0, width=2)
        draw.line((x - 10, y - 8, x - 10, y + 8), fill=0, width=2)
        mask.rectangle((x - 13, y - 11, x + 13, y + 10), fill=255)
    elif kind == "legend":
        draw.rectangle((x - 9, y - 8, x + 9, y + 8), outline=0, width=1)
        draw.ellipse((x - 5, y - 3, x + 1, y + 3), fill=0)
        draw.line((x + 3, y - 2, x + 7, y - 2), fill=0, width=1)
        draw.line((x + 3, y + 2, x + 7, y + 2), fill=0, width=1)
        mask.rectangle((x - 11, y - 10, x + 11, y + 10), fill=255)
    else:
        raise ValueError(kind)


def _centers_for(family: str, variant: int) -> list[tuple[int, int]]:
    if family == "golden_dense":
        return [(18 + 14 * column, 27 + 17 * (column % 4)) for column in range(7)]
    if family == "golden_same_column":
        return [(34, 24), (34, 48), (34, 73), (66, 30), (84, 56), (104, 78)]
    return [(18, 27 + variant * 3), (37, 45), (56, 34 + variant * 3), (75, 65), (96, 39), (110, 76)]


def _apply_degradation(image: Image.Image, family: str, rng: np.random.Generator) -> np.ndarray:
    degradation = DEGRADATION_BY_FAMILY[family]
    if degradation == "gaussian_blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=0.55))
    elif degradation == "median_filter":
        image = image.filter(ImageFilter.MedianFilter(size=3))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if degradation == "salt_pepper":
        selector = rng.random(array.shape)
        array = np.where(selector < 0.002, 0.0, np.where(selector > 0.998, 1.0, array))
    elif degradation == "contrast_compression":
        array = 0.10 + 0.80 * array
    elif degradation == "horizontal_banding":
        bands = (np.arange(array.shape[0], dtype=np.float32)[:, None] % 7 == 0) * 0.035
        array = np.clip(array - bands, 0.0, 1.0)
    return array.astype(np.float32)


def _build_scene(split: str, family: str, variant: int, seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    size = 128
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    artifact_image = Image.new("L", (size, size), 0)
    artifact_draw = ImageDraw.Draw(artifact_image)
    text_image = Image.new("L", (size, size), 0)
    text_draw = ImageDraw.Draw(text_image)
    centers = _centers_for(family, variant)
    radii = [3 + ((index + variant) % 3) for index in range(len(centers))]
    for first, second in zip(centers, centers[1:]):
        draw.line((*first, *second), fill=35, width=1)
    for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        _draw_marker(draw, center, radius, index + variant)

    locations = (
        (15, 104), (31, 105), (47, 103), (62, 105),
        (77, 103), (92, 104), (108, 104), (112, 16),
    )
    hard_negatives: list[tuple[str, float, float]] = []
    for kind, (x, y) in zip(ARTIFACT_KINDS, locations, strict=True):
        target_mask = text_draw if kind == "text" else artifact_draw
        _artifact_geometry(draw, target_mask, kind, x, y)
        hard_negatives.append((kind, float(x), float(y)))

    array = _apply_degradation(image, family, rng)
    ink = 1.0 - array
    text_mask = np.asarray(text_image, dtype=np.float32) / 255.0
    artifact_mask = np.asarray(artifact_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.stack((ink, text_mask, artifact_mask), axis=0).copy())

    yy, xx = np.mgrid[:size, :size]
    center_target = np.zeros((size, size), dtype=np.float32)
    radius_target = np.zeros((size, size), dtype=np.float32)
    for (x, y), radius in zip(centers, radii, strict=True):
        distance2 = (xx - x) ** 2 + (yy - y) ** 2
        gaussian = np.exp(-distance2 / (2.0 * 1.35**2)).astype(np.float32)
        replace = gaussian > center_target
        center_target[replace] = gaussian[replace]
        radius_target[gaussian >= 0.20] = float(radius)
    artifact_target = np.maximum(text_mask, artifact_mask).astype(np.float32)
    return Scene(
        scene_id=f"{split}-{family}-{variant}",
        split=split,
        family=family,
        degradation=DEGRADATION_BY_FAMILY[family],
        seed=seed,
        tensor=tensor,
        center_target=torch.from_numpy(center_target[None]),
        radius_target=torch.from_numpy(radius_target[None]),
        artifact_target=torch.from_numpy(artifact_target[None]),
        centers=tuple((float(x), float(y)) for x, y in centers),
        radii=tuple(float(value) for value in radii),
        hard_negatives=tuple(hard_negatives),
    )


def build_fixed_dataset(split: str) -> tuple[Scene, ...]:
    if split not in SPLIT_FAMILIES:
        raise ValueError(f"Unknown split {split!r}")
    scenes: list[Scene] = []
    base = {"train": 1100, "validation": 2200, "test": 3300}[split]
    variants = {"train": 4, "validation": 3, "test": 2}[split]
    for family_index, family in enumerate(SPLIT_FAMILIES[split]):
        for variant in range(variants):
            scenes.append(_build_scene(split, family, variant, base + family_index * 100 + variant))
    return tuple(scenes)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def dataset_manifest() -> dict[str, object]:
    cases = []
    for split in ("train", "validation", "test"):
        for scene in build_fixed_dataset(split):
            cases.append(
                {
                    "scene_id": scene.scene_id,
                    "split": split,
                    "family": scene.family,
                    "degradation": scene.degradation,
                    "seed": scene.seed,
                    "tensor_sha256": _tensor_sha256(scene.tensor),
                    "center_target_sha256": _tensor_sha256(scene.center_target),
                    "radius_target_sha256": _tensor_sha256(scene.radius_target),
                    "artifact_target_sha256": _tensor_sha256(scene.artifact_target),
                    "center_count": len(scene.centers),
                    "hard_negative_kinds": [item[0] for item in scene.hard_negatives],
                }
            )
    return {
        "manifest_version": 1,
        "dataset_revision": DATASET_REVISION,
        "split_families": SPLIT_FAMILIES,
        "degradation_by_family": DEGRADATION_BY_FAMILY,
        "cases": cases,
    }


def seal_dataset_manifest(output_dir: Path) -> tuple[Path, str]:
    payload = json.dumps(dataset_manifest(), indent=2, sort_keys=True) + "\n"
    encoded = payload.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset-manifest.json"
    path.write_bytes(encoded)
    (output_dir / "dataset-manifest.sha256").write_text(f"{digest}  dataset-manifest.json\n", encoding="ascii")
    return path, digest


__all__ = [
    "ARTIFACT_KINDS",
    "DATASET_REVISION",
    "DEGRADATION_BY_FAMILY",
    "SPLIT_FAMILIES",
    "Scene",
    "build_fixed_dataset",
    "dataset_manifest",
    "seal_dataset_manifest",
]
