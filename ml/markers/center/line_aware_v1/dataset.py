# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes for the line-aware marker-center defect class."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch


WIDTH = 224
HEIGHT = 168
DATASET_REVISION = "marker-center-line-aware-procedural-v1"
SELECTION_FAMILIES = {
    "train": (
        "segmented_growth_train",
        "dual_plateau_train",
        "late_reversal_train",
        "offset_clusters_train",
        "short_run_train",
    ),
    "validation": (
        "cross_phase_wave_validation",
        "sparse_recovery_validation",
        "alternating_level_validation",
    ),
}
SEALED_PUBLIC_FAMILIES = (
    "compressed_withdrawal_public",
    "asymmetric_stair_public",
    "three_phase_shift_public",
    "isolated_tail_public",
)
DEGRADATIONS = {
    "train": ("soft_scan_train", "paper_gradient_train", "light_speckle_train"),
    "validation": ("downsample_validation", "washed_validation"),
    "sealed_public": ("anisotropic_public", "dark_scan_public"),
}
SELECTION_SEED_BASE = {"train": 731_000, "validation": 842_000}
SELECTION_VARIANTS = {"train": 4, "validation": 3}
SEALED_PUBLIC_VARIANTS = 4


@dataclass(frozen=True)
class ProhibitedPoint:
    kind: str
    x: float
    y: float


@dataclass(frozen=True)
class LineAwareScene:
    scene_id: str
    split: str
    family: str
    degradation: str
    seed: int
    tensor: torch.Tensor
    centers: tuple[tuple[float, float], ...]
    radii: tuple[float, ...]
    prohibited: tuple[ProhibitedPoint, ...]


def _centers(family: str, variant: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    count = 8 if "cluster" in family or "three_phase" in family else 7
    xs = np.linspace(46, 196, count, dtype=np.int32)
    phase = np.linspace(0, np.pi * 1.7, count)
    if "plateau" in family or "level" in family:
        ys = np.asarray([105 if index % 2 == 0 else 66 for index in range(count)])
    elif "reversal" in family or "withdrawal" in family:
        ys = 88 - (26 * np.sin(phase))
    elif "stair" in family:
        ys = np.asarray([106, 94, 94, 78, 78, 61, 61], dtype=np.float32)
    elif "recovery" in family or "tail" in family:
        ys = np.asarray([58, 72, 88, 101, 90, 75, 55], dtype=np.float32)
    elif "three_phase" in family:
        ys = np.asarray([106, 99, 91, 69, 62, 55, 80, 72], dtype=np.float32)
    else:
        ys = 92 - np.linspace(0, 40, count) + (8 * np.sin(phase + variant))
    jitter = rng.integers(-2, 3, size=(count, 2), endpoint=False)
    return [
        (int(x + jitter[index, 0] + (variant % 2)), int(y + jitter[index, 1]))
        for index, (x, y) in enumerate(zip(xs, ys, strict=True))
    ]


def _draw_marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, index: int) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    fill = 18 if index % 3 != 1 else 248
    shape = index % 4
    if shape == 0:
        draw.ellipse(box, fill=fill, outline=16, width=2)
    elif shape == 1:
        draw.rectangle(box, fill=fill, outline=16, width=2)
    elif shape == 2:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), fill=fill, outline=16)
    else:
        draw.polygon(((x, y - radius), (x + radius, y + radius), (x - radius, y + radius)), fill=fill, outline=16)


def _pseudo_text(
    draw: ImageDraw.ImageDraw,
    mask: ImageDraw.ImageDraw,
    x: int,
    y: int,
    count: int,
) -> list[ProhibitedPoint]:
    points: list[ProhibitedPoint] = []
    for index in range(count):
        left = x + (index * 5)
        top = y + ((index + 1) % 2)
        draw.line((left, top, left + 3, top + 7), fill=24, width=2)
        draw.line((left, top + 4, left + 4, top + 4), fill=24, width=1)
        mask.rectangle((left - 1, top - 1, left + 5, top + 9), fill=255)
        points.append(ProhibitedPoint("text", float(left + 2), float(top + 4)))
    return points


def _degrade(image: Image.Image, degradation: str, rng: np.random.Generator) -> np.ndarray:
    value = image
    if degradation == "soft_scan_train":
        value = value.filter(ImageFilter.GaussianBlur(0.45))
    elif degradation == "paper_gradient_train":
        array = np.asarray(value, dtype=np.float32)
        gradient = np.linspace(10, -8, array.shape[1], dtype=np.float32)[None, :]
        return np.clip(array + gradient, 0, 255).astype(np.uint8)
    elif degradation == "light_speckle_train":
        array = np.asarray(value, dtype=np.float32)
        return np.clip(array + rng.normal(0, 1.8, array.shape), 0, 255).astype(np.uint8)
    elif degradation == "downsample_validation":
        value = value.resize((168, 126), Image.Resampling.BILINEAR).resize(
            (WIDTH, HEIGHT), Image.Resampling.BILINEAR
        )
    elif degradation == "washed_validation":
        value = ImageEnhance.Contrast(value).enhance(0.72)
    elif degradation == "anisotropic_public":
        value = value.resize((196, 168), Image.Resampling.BILINEAR).resize(
            (WIDTH, HEIGHT), Image.Resampling.BICUBIC
        )
    elif degradation == "dark_scan_public":
        value = ImageEnhance.Brightness(value).enhance(0.9)
        value = ImageEnhance.Contrast(value).enhance(1.18)
    return np.asarray(value, dtype=np.uint8)


def build_scene(*, split: str, family: str, degradation: str, variant: int, seed: int) -> LineAwareScene:
    if split not in {"train", "validation", "sealed_public"}:
        raise ValueError(f"Unsupported split {split!r}")
    rng = np.random.default_rng(seed)
    image = Image.new("L", (WIDTH, HEIGHT), 250)
    text_image = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    text = ImageDraw.Draw(text_image)
    artifact = ImageDraw.Draw(artifact_image)
    prohibited: list[ProhibitedPoint] = []

    axis_x = 30 + (variant % 3)
    axis_y = 142 - (variant % 2)
    draw.line((axis_x, 18, axis_x, axis_y), fill=20, width=2)
    draw.line((axis_x, axis_y, 210, axis_y), fill=20, width=2)
    artifact.line((axis_x, 15, axis_x, axis_y + 3), fill=255, width=7)
    artifact.line((axis_x - 3, axis_y, 212, axis_y), fill=255, width=7)
    prohibited.extend((ProhibitedPoint("axis", float(axis_x), 70.0), ProhibitedPoint("axis", 120.0, float(axis_y))))
    for tick_x in range(48, 205, 25):
        draw.line((tick_x, axis_y - 3, tick_x, axis_y + 3), fill=20)
        artifact.rectangle((tick_x - 3, axis_y - 5, tick_x + 3, axis_y + 5), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(tick_x), float(axis_y)))
    for tick_y in range(32, 132, 22):
        draw.line((axis_x - 3, tick_y, axis_x + 3, tick_y), fill=20)
        artifact.rectangle((axis_x - 5, tick_y - 3, axis_x + 5, tick_y + 3), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(axis_x), float(tick_y)))

    dividers = (103 + (variant * 3), 158 - (variant * 2))
    for divider_x in dividers:
        draw.line((divider_x, 18, divider_x, axis_y), fill=46, width=1)
        artifact.rectangle((divider_x - 3, 15, divider_x + 3, axis_y), fill=255)
        prohibited.append(ProhibitedPoint("divider", float(divider_x), 50.0))

    draw.rectangle((147, 23, 207, 48), outline=42, width=1)
    artifact.rectangle((144, 20, 210, 51), outline=255, width=7)
    prohibited.append(ProhibitedPoint("legend", 177.0, 35.0))
    prohibited.extend(_pseudo_text(draw, text, 151, 29, 8))
    draw.line((49, 28, 66, 17, 82, 28), fill=35, width=2)
    artifact.line((46, 30, 66, 14, 85, 30), fill=255, width=7)
    prohibited.append(ProhibitedPoint("bracket", 66.0, 20.0))
    draw.line((91, 35, 105, 24), fill=34, width=2)
    draw.polygon(((105, 24), (97, 24), (103, 31)), fill=34)
    artifact.line((88, 38, 108, 21), fill=255, width=7)
    artifact.polygon(((108, 21), (95, 21), (103, 34)), fill=255)
    prohibited.extend((ProhibitedPoint("arrow_shaft", 96.0, 31.0), ProhibitedPoint("arrowhead", 103.0, 27.0)))
    prohibited.extend(_pseudo_text(draw, text, 40, 149, 9))
    prohibited.extend(_pseudo_text(draw, text, 3, 62, 4))

    centers = _centers(family, variant, rng)
    radii = [3 + ((index + variant) % 3) for index in range(len(centers))]
    draw.line(centers, fill=52, width=2, joint="curve")
    artifact.line(centers, fill=255, width=7, joint="curve")
    for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        _draw_marker(draw, center, radius, index + variant)
        x, y = center
        clear = radius + 3
        artifact.ellipse((x - clear, y - clear, x + clear, y + clear), fill=0)

    for divider_x in dividers:
        interpolated_y = int(np.interp(divider_x, [point[0] for point in centers], [point[1] for point in centers]))
        if min((divider_x - x) ** 2 + (interpolated_y - y) ** 2 for x, y in centers) > 81:
            prohibited.append(ProhibitedPoint("line_intersection", float(divider_x), float(interpolated_y)))

    array = _degrade(image, degradation, rng)
    tensor = torch.from_numpy(
        np.stack(
            (
                1.0 - (array.astype(np.float32) / 255.0),
                np.asarray(text_image, dtype=np.float32) / 255.0,
                np.asarray(artifact_image, dtype=np.float32) / 255.0,
            ),
            axis=0,
        ).copy()
    )
    return LineAwareScene(
        scene_id=f"{split}-{family}-{variant}",
        split=split,
        family=family,
        degradation=degradation,
        seed=seed,
        tensor=tensor,
        centers=tuple((float(x), float(y)) for x, y in centers),
        radii=tuple(float(radius) for radius in radii),
        prohibited=tuple(prohibited),
    )


def build_selection_scenes(split: str) -> tuple[LineAwareScene, ...]:
    if split not in SELECTION_FAMILIES:
        raise ValueError(f"Unsupported selection split {split!r}")
    scenes: list[LineAwareScene] = []
    families = SELECTION_FAMILIES[split]
    degradations = DEGRADATIONS[split]
    for family_index, family in enumerate(families):
        for variant in range(SELECTION_VARIANTS[split]):
            scenes.append(build_scene(
                split=split,
                family=family,
                degradation=degradations[(family_index + variant) % len(degradations)],
                variant=variant,
                seed=SELECTION_SEED_BASE[split] + (family_index * 100) + variant,
            ))
    return tuple(scenes)


def build_sealed_public_scenes(secret_seed: int) -> tuple[LineAwareScene, ...]:
    degradations = DEGRADATIONS["sealed_public"]
    return tuple(
        build_scene(
            split="sealed_public",
            family=family,
            degradation=degradations[(family_index + variant) % len(degradations)],
            variant=variant,
            seed=secret_seed + (family_index * 100) + variant,
        )
        for family_index, family in enumerate(SEALED_PUBLIC_FAMILIES)
        for variant in range(SEALED_PUBLIC_VARIANTS)
    )


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def scene_manifest(scene: LineAwareScene, *, expose_truth: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "scene_id": scene.scene_id,
        "split": scene.split,
        "family": scene.family,
        "degradation": scene.degradation,
        "tensor_sha256": _array_hash(scene.tensor.numpy().astype("<f4", copy=False)),
        "center_count": len(scene.centers),
        "prohibited_kinds": sorted({item.kind for item in scene.prohibited}),
    }
    if expose_truth:
        result.update({
            "seed": scene.seed,
            "centers": [[x, y] for x, y in scene.centers],
            "radii": list(scene.radii),
            "prohibited": [{"kind": item.kind, "x": item.x, "y": item.y} for item in scene.prohibited],
        })
    return result


def selection_manifest() -> dict[str, object]:
    return {
        "schema": "graphreader.marker-center-line-aware-selection.v1",
        "dataset_revision": DATASET_REVISION,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "split_families": {key: list(value) for key, value in SELECTION_FAMILIES.items()},
        "degradations": {key: list(value) for key, value in DEGRADATIONS.items()},
        "cases": [
            scene_manifest(scene, expose_truth=True)
            for split in ("train", "validation")
            for scene in build_selection_scenes(split)
        ],
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
        "schema": "graphreader.marker-center-line-aware-sealed-fixtures.v1",
        "dataset_revision": DATASET_REVISION,
        "case_order": [scene.scene_id for scene in ordered],
        "cases": cases,
    }
    arrays["metadata_json_utf8"] = np.frombuffer(
        (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        dtype=np.uint8,
    )
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
                scene_id=case["scene_id"],
                split=case["split"],
                family=case["family"],
                degradation=case["degradation"],
                seed=int(case["seed"]),
                tensor=torch.from_numpy(archive[f"{prefix}_tensor"].copy()),
                centers=tuple(tuple(map(float, row)) for row in archive[f"{prefix}_centers"]),
                radii=tuple(map(float, archive[f"{prefix}_radii"])),
                prohibited=tuple(
                    ProhibitedPoint(item["kind"], float(x), float(y))
                    for item, (x, y) in zip(case["prohibited"], points, strict=True)
                ),
            ))
    return tuple(scenes)


__all__ = [
    "DATASET_REVISION", "DEGRADATIONS", "HEIGHT", "LineAwareScene", "ProhibitedPoint",
    "SEALED_PUBLIC_FAMILIES", "SELECTION_FAMILIES", "WIDTH", "build_sealed_public_scenes",
    "build_selection_scenes", "load_sealed_public_archive", "save_sealed_public_archive",
    "selection_manifest",
]
