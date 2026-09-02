# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes covering a 6 to 25 pixel marker diameter envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch

from ml.markers.center.line_aware_v1.dataset import LineAwareScene, ProhibitedPoint

WIDTH, HEIGHT = 224, 168
DATASET_REVISION = "marker-center-proposal-geometry-procedural-v13"
FAMILIES = {
    "train": ("geometry_small_train", "geometry_medium_train", "geometry_joint_train"),
    "dev": ("geometry_wide_dev", "geometry_mixed_dev", "geometry_intersection_dev"),
}
VARIANTS = {"train": 5, "dev": 4}
SEEDS = {"train": 1_470_000, "dev": 1_570_000}


def _centers(variant: int) -> list[tuple[int, int]]:
    xs = np.linspace(54, 194, 8, dtype=np.int32)
    ys = np.asarray([104, 82, 110, 69, 96, 60, 91, 75], dtype=np.int32)
    return [(int(x + ((index + variant) % 3) - 1), int(y + ((variant * 2 + index) % 5) - 2)) for index, (x, y) in enumerate(zip(xs, ys, strict=True))]


def _radii(family: str, variant: int) -> list[int]:
    if "small" in family:
        values = (3, 4, 5, 6)
    elif "medium" in family:
        values = (5, 6, 8, 9)
    elif "wide" in family:
        values = (8, 9, 10, 12)
    elif "mixed" in family:
        values = (3, 6, 9, 12)
    elif "intersection" in family:
        values = (4, 7, 10, 12)
    else:
        values = (4, 6, 8, 10)
    return [values[(index + variant) % len(values)] for index in range(8)]


def _marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, index: int) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    width = max(1, radius // 4)
    if index % 3 == 0:
        draw.ellipse(box, fill=18, outline=8, width=width)
    elif index % 3 == 1:
        draw.rectangle(box, fill=246, outline=12, width=width)
    else:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), fill=18, outline=8)


def _text(draw: ImageDraw.ImageDraw, mask: ImageDraw.ImageDraw, x: int, y: int) -> list[ProhibitedPoint]:
    points: list[ProhibitedPoint] = []
    for index in range(7):
        left, top = x + index * 6, y + index % 2
        draw.line((left, top, left + 4, top + 8), fill=25, width=2)
        draw.line((left, top + 4, left + 5, top + 4), fill=25, width=1)
        mask.rectangle((left - 2, top - 2, left + 7, top + 10), fill=255)
        points.append(ProhibitedPoint("text", float(left + 2), float(top + 4)))
    return points


def build_scene(*, split: str, family: str, variant: int, seed: int) -> LineAwareScene:
    rng = np.random.default_rng(seed)
    image = Image.new("L", (WIDTH, HEIGHT), 250)
    text_image = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw, text, artifact = ImageDraw.Draw(image), ImageDraw.Draw(text_image), ImageDraw.Draw(artifact_image)
    prohibited: list[ProhibitedPoint] = []
    axis_x, axis_y = 30 + variant % 3, 145 - variant % 2
    draw.line((axis_x, 16, axis_x, axis_y), fill=20, width=2)
    draw.line((axis_x, axis_y, 213, axis_y), fill=20, width=2)
    artifact.line((axis_x, 13, axis_x, axis_y + 3), fill=255, width=7)
    artifact.line((axis_x - 3, axis_y, 215, axis_y), fill=255, width=7)
    prohibited.extend((ProhibitedPoint("axis", float(axis_x), 70.0), ProhibitedPoint("axis", 125.0, float(axis_y))))
    for x in range(54, 211, 26):
        draw.line((x, axis_y - 3, x, axis_y + 3), fill=20, width=2)
        artifact.rectangle((x - 4, axis_y - 5, x + 4, axis_y + 5), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(x), float(axis_y)))
    for y in range(34, 136, 22):
        draw.line((axis_x - 3, y, axis_x + 3, y), fill=20, width=2)
        artifact.rectangle((axis_x - 5, y - 4, axis_x + 5, y + 4), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(axis_x), float(y)))
    draw.rectangle((151, 21, 212, 50), outline=40, width=1)
    artifact.rectangle((147, 17, 216, 54), outline=255, width=7)
    prohibited.append(ProhibitedPoint("legend", 180.0, 36.0))
    prohibited.extend(_text(draw, text, 155, 28))
    draw.line((49, 28, 66, 17, 85, 28), fill=30, width=2)
    artifact.line((45, 31, 66, 14, 89, 31), fill=255, width=7)
    prohibited.append(ProhibitedPoint("bracket", 66.0, 20.0))
    centers, radii = _centers(variant), _radii(family, variant)
    draw.line(centers, fill=52, width=2, joint="curve")
    artifact.line(centers, fill=255, width=7, joint="curve")
    for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        _marker(draw, center, radius, index + variant)
        x, y = center
        clear = radius + 3
        artifact.ellipse((x - clear, y - clear, x + clear, y + clear), fill=0)
    if "joint" in family or "intersection" in family:
        for index, x in enumerate((72, 96, 120, 144, 168)):
            y = 56 + (index % 2) * 10
            draw.line((x - 6, y, x + 6, y), fill=28, width=2)
            draw.line((x, y - 6, x, y + 6), fill=28, width=2)
            artifact.ellipse((x - 9, y - 9, x + 9, y + 9), fill=255)
            prohibited.append(ProhibitedPoint("line_intersection", float(x), float(y)))
    if split == "dev":
        image = image.filter(ImageFilter.GaussianBlur(0.35 + 0.08 * (variant % 2)))
        if variant % 2:
            image = ImageEnhance.Contrast(image).enhance(0.8)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.clip(array + rng.normal(0.0, 0.003, array.shape), 0.0, 1.0)
    tensor = torch.from_numpy(np.stack((1.0 - array, np.asarray(text_image, dtype=np.float32) / 255.0, np.asarray(artifact_image, dtype=np.float32) / 255.0), axis=0).astype(np.float32, copy=False))
    return LineAwareScene(
        scene_id=f"{split}-{family}-{variant}", split=split, family=family,
        degradation="blur_contrast_dev" if split == "dev" else "light_sensor_noise_train", seed=seed,
        tensor=tensor, centers=tuple((float(x), float(y)) for x, y in centers), radii=tuple(float(value) for value in radii), prohibited=tuple(prohibited),
    )


def build_selection_scenes(split: str) -> tuple[LineAwareScene, ...]:
    return tuple(build_scene(split=split, family=family, variant=variant, seed=SEEDS[split] + family_index * 100 + variant) for family_index, family in enumerate(FAMILIES[split]) for variant in range(VARIANTS[split]))


def selection_manifest() -> dict[str, object]:
    return {
        "schema": "graphreader.marker-center-proposal-geometry-selection.v13", "dataset_revision": DATASET_REVISION,
        "synthetic_only": True, "private_or_article_images": False, "public_gate_archive_opened": False,
        "families": {key: list(value) for key, value in FAMILIES.items()}, "variants": VARIANTS, "seed_bases": SEEDS,
        "marker_diameter_envelope_pixels": [6, 25],
        "cases": [{"scene_id": scene.scene_id, "split": split, "family": scene.family, "degradation": scene.degradation, "seed": scene.seed, "tensor_sha256": hashlib.sha256(scene.tensor.numpy().tobytes()).hexdigest(), "center_count": len(scene.centers), "prohibited_kinds": sorted({item.kind for item in scene.prohibited})} for split in ("train", "dev") for scene in build_selection_scenes(split)],
    }


def seal_manifest(output_dir: Path) -> tuple[Path, str]:
    encoded = (json.dumps(selection_manifest(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset-manifest.json"
    path.write_bytes(encoded)
    (output_dir / "dataset-manifest.sha256").write_text(f"{digest}  dataset-manifest.json\n", encoding="ascii")
    return path, digest
