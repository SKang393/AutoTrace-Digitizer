# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes for the radial-feature marker-center defect class."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch

from ml.markers.center.line_aware_v1.dataset import LineAwareScene, ProhibitedPoint


WIDTH = 224
HEIGHT = 168
DATASET_REVISION = "marker-center-radial-feature-procedural-v1"
SELECTION_FAMILIES = {
    "train": (
        "burst_decay_train", "split_ramp_train", "nested_plateau_train",
        "delayed_switch_train", "microcycle_train",
    ),
    "validation": (
        "sawtooth_hold_validation", "paired_arc_validation", "rebound_gap_validation",
    ),
}
SEALED_PUBLIC_FAMILIES = (
    "staggered_echo_public", "reverse_fan_public",
    "broken_ladder_public", "late_probe_public",
)
DEGRADATIONS = {
    "train": ("ink_bleed_train", "vertical_shade_train", "salt_scan_train"),
    "validation": ("fax_resample_validation", "low_contrast_validation"),
    "sealed_public": ("horizontal_stretch_public", "gray_cast_public"),
}
SELECTION_SEED_BASE = {"train": 963_000, "validation": 1_074_000}
SELECTION_VARIANTS = {"train": 4, "validation": 3}
SEALED_PUBLIC_VARIANTS = 4


def _centers(family: str, variant: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    count = 8 if any(token in family for token in ("nested", "microcycle", "echo")) else 7
    xs = np.linspace(44, 198, count, dtype=np.float32)
    t = np.linspace(0.0, 1.0, count)
    if "burst_decay" in family:
        ys = 110 - (62 * np.exp(-3.0 * t)) + (12 * t)
    elif "split_ramp" in family:
        ys = np.where(t < 0.5, 112 - (70 * t), 91 - (50 * (t - 0.5)))
    elif "nested_plateau" in family:
        ys = np.asarray([108, 108, 82, 82, 56, 56, 78, 78], dtype=np.float32)
    elif "delayed_switch" in family:
        ys = np.asarray([105, 104, 103, 102, 66, 61, 57], dtype=np.float32)
    elif "microcycle" in family:
        ys = np.asarray([108, 74, 101, 67, 95, 60, 88, 54], dtype=np.float32)
    elif "sawtooth" in family:
        ys = np.asarray([108, 70, 101, 64, 92, 58, 58], dtype=np.float32)
    elif "paired_arc" in family:
        ys = 91 - (29 * np.sin(t * np.pi * 2.0))
    elif "rebound_gap" in family:
        ys = np.asarray([61, 75, 94, 110, 91, 72, 59], dtype=np.float32)
    elif "staggered_echo" in family:
        ys = np.asarray([109, 108, 83, 82, 58, 57, 81, 80], dtype=np.float32)
    elif "reverse_fan" in family:
        ys = 55 + (54 * np.power(t, 1.7))
    elif "broken_ladder" in family:
        ys = np.asarray([111, 96, 96, 78, 78, 59, 84], dtype=np.float32)
    else:
        ys = np.asarray([104, 101, 98, 95, 65, 61, 57], dtype=np.float32)
    jitter = rng.integers(-2, 3, size=(count, 2))
    return [
        (int(round(x + jitter[index, 0] + (variant % 2))), int(round(y + jitter[index, 1])))
        for index, (x, y) in enumerate(zip(xs, ys, strict=True))
    ]


def _marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, index: int) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    fill = 20 if index % 3 else 246
    if index % 4 == 0:
        draw.ellipse(box, fill=fill, outline=14, width=2)
    elif index % 4 == 1:
        draw.rectangle(box, fill=fill, outline=14, width=2)
    elif index % 4 == 2:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), fill=fill, outline=14)
    else:
        draw.polygon(((x, y - radius), (x + radius, y + radius), (x - radius, y + radius)), fill=fill, outline=14)


def _glyphs(draw: ImageDraw.ImageDraw, mask: ImageDraw.ImageDraw, x: int, y: int, count: int) -> list[ProhibitedPoint]:
    points: list[ProhibitedPoint] = []
    for index in range(count):
        left = x + (index * 5)
        top = y + (index % 3)
        draw.line((left, top, left + 4, top + 7), fill=28, width=1 + (index % 2))
        draw.line((left, top + 5, left + 4, top + 3), fill=28, width=1)
        mask.rectangle((left - 1, top - 1, left + 5, top + 9), fill=255)
        points.append(ProhibitedPoint("text", float(left + 2), float(top + 4)))
    return points


def _degrade(image: Image.Image, degradation: str, rng: np.random.Generator) -> np.ndarray:
    if degradation == "ink_bleed_train":
        return np.asarray(image.filter(ImageFilter.GaussianBlur(0.7)), dtype=np.uint8)
    array = np.asarray(image, dtype=np.float32)
    if degradation == "vertical_shade_train":
        return np.clip(array + np.linspace(-7, 11, array.shape[0])[:, None], 0, 255).astype(np.uint8)
    if degradation == "salt_scan_train":
        noise = rng.normal(0, 2.1, array.shape)
        return np.clip(array + noise, 0, 255).astype(np.uint8)
    if degradation == "fax_resample_validation":
        value = image.resize((160, 120), Image.Resampling.BILINEAR).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)
        return np.asarray(value, dtype=np.uint8)
    if degradation == "low_contrast_validation":
        return np.asarray(ImageEnhance.Contrast(image).enhance(0.64), dtype=np.uint8)
    if degradation == "horizontal_stretch_public":
        value = image.resize((205, HEIGHT), Image.Resampling.BILINEAR).resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
        return np.asarray(value, dtype=np.uint8)
    if degradation == "gray_cast_public":
        value = ImageEnhance.Brightness(image).enhance(0.93)
        return np.asarray(ImageEnhance.Contrast(value).enhance(0.82), dtype=np.uint8)
    return array.astype(np.uint8)


def build_scene(*, split: str, family: str, degradation: str, variant: int, seed: int) -> LineAwareScene:
    rng = np.random.default_rng(seed)
    image = Image.new("L", (WIDTH, HEIGHT), 250)
    text_image = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    text = ImageDraw.Draw(text_image)
    artifact = ImageDraw.Draw(artifact_image)
    prohibited: list[ProhibitedPoint] = []
    axis_x, axis_y = 29 + (variant % 4), 141 - (variant % 3)
    draw.line((axis_x, 17, axis_x, axis_y), fill=18, width=2)
    draw.line((axis_x, axis_y, 211, axis_y), fill=18, width=2)
    artifact.line((axis_x, 14, axis_x, axis_y + 4), fill=255, width=7)
    artifact.line((axis_x - 3, axis_y, 214, axis_y), fill=255, width=7)
    prohibited.extend((ProhibitedPoint("axis", float(axis_x), 75.0), ProhibitedPoint("axis", 121.0, float(axis_y))))
    for tick_x in range(47, 207, 23):
        draw.line((tick_x, axis_y - 3, tick_x, axis_y + 3), fill=22)
        artifact.rectangle((tick_x - 3, axis_y - 5, tick_x + 3, axis_y + 5), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(tick_x), float(axis_y)))
    for tick_y in range(30, 133, 19):
        draw.line((axis_x - 3, tick_y, axis_x + 3, tick_y), fill=22)
        artifact.rectangle((axis_x - 5, tick_y - 3, axis_x + 5, tick_y + 3), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(axis_x), float(tick_y)))
    dividers = (92 + (variant * 2), 151 + (variant % 3))
    for divider_x in dividers:
        draw.line((divider_x, 18, divider_x, axis_y), fill=48, width=1)
        artifact.rectangle((divider_x - 3, 15, divider_x + 3, axis_y), fill=255)
        prohibited.append(ProhibitedPoint("divider", float(divider_x), 52.0))
    draw.rectangle((150, 21, 210, 47), outline=44)
    artifact.rectangle((147, 18, 213, 50), outline=255, width=7)
    prohibited.append(ProhibitedPoint("legend", 180.0, 34.0))
    prohibited.extend(_glyphs(draw, text, 154, 27, 8))
    draw.line((50, 31, 67, 16, 84, 31), fill=34, width=2)
    artifact.line((47, 34, 67, 13, 87, 34), fill=255, width=7)
    prohibited.append(ProhibitedPoint("bracket", 67.0, 20.0))
    draw.line((94, 37, 109, 23), fill=34, width=2)
    draw.polygon(((109, 23), (101, 24), (107, 31)), fill=34)
    artifact.line((91, 40, 112, 20), fill=255, width=7)
    artifact.polygon(((112, 20), (99, 21), (107, 34)), fill=255)
    prohibited.extend((ProhibitedPoint("arrow_shaft", 100.0, 31.0), ProhibitedPoint("arrowhead", 107.0, 27.0)))
    prohibited.extend(_glyphs(draw, text, 42, 149, 8))
    prohibited.extend(_glyphs(draw, text, 3, 58, 4))
    centers = _centers(family, variant, rng)
    radii = [3 + ((index + variant + 1) % 3) for index in range(len(centers))]
    draw.line(centers, fill=54, width=2, joint="curve")
    artifact.line(centers, fill=255, width=7, joint="curve")
    for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        _marker(draw, center, radius, index + variant)
        x, y = center
        clear = radius + 3
        artifact.ellipse((x - clear, y - clear, x + clear, y + clear), fill=0)
    for divider_x in dividers:
        iy = int(np.interp(divider_x, [point[0] for point in centers], [point[1] for point in centers]))
        if min((divider_x - x) ** 2 + (iy - y) ** 2 for x, y in centers) > 81:
            prohibited.append(ProhibitedPoint("line_intersection", float(divider_x), float(iy)))
    array = _degrade(image, degradation, rng)
    tensor = torch.from_numpy(np.stack((
        1.0 - (array.astype(np.float32) / 255.0),
        np.asarray(text_image, dtype=np.float32) / 255.0,
        np.asarray(artifact_image, dtype=np.float32) / 255.0,
    ), axis=0).copy())
    return LineAwareScene(
        scene_id=f"{split}-{family}-{variant}", split=split, family=family,
        degradation=degradation, seed=seed, tensor=tensor,
        centers=tuple((float(x), float(y)) for x, y in centers),
        radii=tuple(float(radius) for radius in radii), prohibited=tuple(prohibited),
    )


def build_selection_scenes(split: str) -> tuple[LineAwareScene, ...]:
    if split not in SELECTION_FAMILIES:
        raise ValueError(f"Unsupported selection split {split!r}")
    return tuple(
        build_scene(
            split=split, family=family,
            degradation=DEGRADATIONS[split][(family_index + variant) % len(DEGRADATIONS[split])],
            variant=variant, seed=SELECTION_SEED_BASE[split] + (family_index * 100) + variant,
        )
        for family_index, family in enumerate(SELECTION_FAMILIES[split])
        for variant in range(SELECTION_VARIANTS[split])
    )


def build_sealed_public_scenes(secret_seed: int) -> tuple[LineAwareScene, ...]:
    return tuple(
        build_scene(
            split="sealed_public", family=family,
            degradation=DEGRADATIONS["sealed_public"][(family_index + variant) % len(DEGRADATIONS["sealed_public"])],
            variant=variant, seed=secret_seed + (family_index * 100) + variant,
        )
        for family_index, family in enumerate(SEALED_PUBLIC_FAMILIES)
        for variant in range(SEALED_PUBLIC_VARIANTS)
    )


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def scene_manifest(scene: LineAwareScene, *, expose_truth: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "scene_id": scene.scene_id, "split": scene.split, "family": scene.family,
        "degradation": scene.degradation,
        "tensor_sha256": _array_hash(scene.tensor.numpy().astype("<f4", copy=False)),
        "center_count": len(scene.centers),
        "prohibited_kinds": sorted({item.kind for item in scene.prohibited}),
    }
    if expose_truth:
        result.update({
            "seed": scene.seed, "centers": [[x, y] for x, y in scene.centers],
            "radii": list(scene.radii),
            "prohibited": [{"kind": item.kind, "x": item.x, "y": item.y} for item in scene.prohibited],
        })
    return result


def selection_manifest() -> dict[str, object]:
    return {
        "schema": "graphreader.marker-center-radial-feature-selection.v1",
        "dataset_revision": DATASET_REVISION, "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False,
        "split_families": {key: list(value) for key, value in SELECTION_FAMILIES.items()},
        "degradations": {key: list(value) for key, value in DEGRADATIONS.items()},
        "cases": [scene_manifest(scene, expose_truth=True) for split in ("train", "validation") for scene in build_selection_scenes(split)],
    }


def save_sealed_public_archive(scenes: Iterable[LineAwareScene], path: Path) -> dict[str, object]:
    ordered = tuple(scenes)
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, object]] = []
    for index, scene in enumerate(ordered):
        prefix = f"scene_{index:03d}"
        arrays[f"{prefix}_tensor"] = scene.tensor.numpy().astype("<f4", copy=False)
        arrays[f"{prefix}_centers"] = np.asarray(scene.centers, dtype="<f4")
        arrays[f"{prefix}_radii"] = np.asarray(scene.radii, dtype="<f4")
        arrays[f"{prefix}_prohibited_xy"] = np.asarray([(item.x, item.y) for item in scene.prohibited], dtype="<f4")
        cases.append(scene_manifest(scene, expose_truth=True))
    metadata = {
        "schema": "graphreader.marker-center-radial-feature-sealed-fixtures.v1",
        "dataset_revision": DATASET_REVISION,
        "case_order": [scene.scene_id for scene in ordered], "cases": cases,
    }
    arrays["metadata_json_utf8"] = np.frombuffer((json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return metadata


def load_sealed_public_archive(path: Path) -> tuple[LineAwareScene, ...]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_json_utf8"].tolist()).decode("utf-8"))
        scenes: list[LineAwareScene] = []
        for index, case in enumerate(metadata["cases"]):
            prefix = f"scene_{index:03d}"
            points = archive[f"{prefix}_prohibited_xy"]
            scenes.append(LineAwareScene(
                scene_id=case["scene_id"], split=case["split"], family=case["family"],
                degradation=case["degradation"], seed=int(case["seed"]),
                tensor=torch.from_numpy(archive[f"{prefix}_tensor"].copy()),
                centers=tuple(tuple(map(float, row)) for row in archive[f"{prefix}_centers"]),
                radii=tuple(map(float, archive[f"{prefix}_radii"])),
                prohibited=tuple(ProhibitedPoint(item["kind"], float(x), float(y)) for item, (x, y) in zip(case["prohibited"], points, strict=True)),
            ))
    return tuple(scenes)


__all__ = [
    "DATASET_REVISION", "DEGRADATIONS", "HEIGHT", "SEALED_PUBLIC_FAMILIES",
    "SELECTION_FAMILIES", "WIDTH", "build_sealed_public_scenes",
    "build_selection_scenes", "load_sealed_public_archive",
    "save_sealed_public_archive", "selection_manifest",
]
