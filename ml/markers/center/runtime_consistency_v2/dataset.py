# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural scenes for runtime-consistent marker-center validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch

from ml.markers.center.line_aware_v1.dataset import LineAwareScene, ProhibitedPoint


WIDTH = 256
HEIGHT = 192
DATASET_REVISION = "marker-center-runtime-consistency-procedural-v2"
SELECTION_FAMILIES = {
    "train": (
        "alternating_rise_train",
        "stepped_fall_train",
        "paired_reversal_train",
        "sparse_probe_train",
        "dense_cycle_train",
        "offset_plateau_train",
    ),
    "validation": (
        "late_crossover_validation",
        "compress_expand_validation",
        "asymmetric_wave_validation",
        "session_gap_validation",
    ),
}
SEALED_PUBLIC_FAMILIES = (
    "double_rebound_public",
    "descending_echo_public",
    "interleaved_plateau_public",
    "late_surge_public",
    "wide_cycle_public",
)
DEGRADATIONS = {
    "train": (
        "soft_halo_train",
        "column_band_train",
        "speckle_train",
        "bicubic_scan_train",
    ),
    "validation": (
        "gamma_lift_validation",
        "row_streak_validation",
        "compact_resample_validation",
    ),
    "sealed_public": (
        "mixed_focus_public",
        "background_ramp_public",
        "anisotropic_scan_public",
    ),
}
SELECTION_SEED_BASE = {"train": 1_241_000, "validation": 1_357_000}
SELECTION_VARIANTS = {"train": 5, "validation": 3}
SEALED_PUBLIC_VARIANTS = 4


def _centers(family: str, variant: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    count = 9 if any(token in family for token in ("dense", "interleaved", "cycle")) else 8
    xs = np.linspace(48, 232, count, dtype=np.float32)
    t = np.linspace(0.0, 1.0, count)
    if "alternating_rise" in family:
        ys = 132 - (57 * t) + (12 * np.sin(t * np.pi * 5.0))
    elif "stepped_fall" in family:
        ys = np.asarray([66, 66, 82, 82, 101, 101, 122, 122], dtype=np.float32)
    elif "paired_reversal" in family:
        ys = 98 - (31 * np.sin(t * np.pi * 2.0)) - (9 * np.sin(t * np.pi * 4.0))
    elif "sparse_probe" in family:
        ys = np.asarray([126, 124, 121, 119, 84, 79, 75, 72], dtype=np.float32)
    elif "dense_cycle" in family:
        ys = np.asarray([128, 91, 119, 82, 111, 73, 103, 65, 94], dtype=np.float32)
    elif "offset_plateau" in family:
        ys = np.asarray([121, 120, 95, 94, 70, 69, 92, 91], dtype=np.float32)
    elif "late_crossover" in family:
        ys = np.asarray([71, 78, 87, 98, 112, 99, 81, 63], dtype=np.float32)
    elif "compress_expand" in family:
        ys = 97 + (34 * np.sin((t - 0.15) * np.pi * 1.6))
    elif "asymmetric_wave" in family:
        ys = np.asarray([125, 103, 79, 67, 75, 96, 113, 88], dtype=np.float32)
    elif "session_gap" in family:
        ys = np.asarray([124, 121, 117, 86, 82, 78, 65, 61], dtype=np.float32)
    elif "double_rebound" in family:
        ys = np.asarray([67, 91, 119, 93, 69, 88, 113, 76], dtype=np.float32)
    elif "descending_echo" in family:
        ys = 128 - (61 * np.power(t, 0.75)) + (8 * np.sin(t * np.pi * 4.0))
    elif "interleaved_plateau" in family:
        ys = np.asarray([124, 123, 98, 97, 74, 73, 94, 93, 68], dtype=np.float32)
    elif "late_surge" in family:
        ys = np.asarray([126, 125, 123, 119, 112, 84, 70, 58], dtype=np.float32)
    else:
        ys = 99 - (36 * np.sin(t * np.pi * 2.4)) + (13 * t)
    jitter = rng.integers(-2, 3, size=(count, 2))
    return [
        (
            int(round(x + jitter[index, 0] + ((variant + index) % 2))),
            int(round(y + jitter[index, 1])),
        )
        for index, (x, y) in enumerate(zip(xs, ys, strict=True))
    ]


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    ordinal: int,
) -> None:
    x, y = center
    bounds = (x - radius, y - radius, x + radius, y + radius)
    fill = 18 if ordinal % 3 else 248
    shape = ordinal % 5
    if shape == 0:
        draw.ellipse(bounds, fill=fill, outline=12, width=2)
    elif shape == 1:
        draw.rectangle(bounds, fill=fill, outline=12, width=2)
    elif shape == 2:
        draw.polygon(
            ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
            fill=fill,
            outline=12,
        )
    elif shape == 3:
        draw.polygon(
            ((x, y - radius), (x + radius, y + radius), (x - radius, y + radius)),
            fill=fill,
            outline=12,
        )
    else:
        draw.ellipse(bounds, fill=fill, outline=12, width=3)


def _draw_glyph_run(
    draw: ImageDraw.ImageDraw,
    text_mask: ImageDraw.ImageDraw,
    x: int,
    y: int,
    count: int,
) -> list[ProhibitedPoint]:
    points: list[ProhibitedPoint] = []
    for index in range(count):
        left = x + (index * 6)
        top = y + ((index + 1) % 3)
        draw.rectangle((left, top, left + 3, top + 7), outline=26, width=1)
        draw.line((left, top + 4, left + 4, top + 2), fill=26, width=1)
        text_mask.rectangle((left - 2, top - 2, left + 6, top + 10), fill=255)
        points.append(ProhibitedPoint("text", float(left + 2), float(top + 4)))
    return points


def _degrade(image: Image.Image, degradation: str, rng: np.random.Generator) -> np.ndarray:
    if degradation == "soft_halo_train":
        return np.asarray(image.filter(ImageFilter.GaussianBlur(0.55)), dtype=np.uint8)
    array = np.asarray(image, dtype=np.float32)
    if degradation == "column_band_train":
        band = 4.5 * np.sin(np.linspace(0.0, np.pi * 9.0, array.shape[1]))
        return np.clip(array + band[None, :], 0, 255).astype(np.uint8)
    if degradation == "speckle_train":
        noise = rng.normal(0.0, 2.4, array.shape)
        return np.clip(array + noise, 0, 255).astype(np.uint8)
    if degradation == "bicubic_scan_train":
        value = image.resize((211, 158), Image.Resampling.BICUBIC).resize(
            (WIDTH, HEIGHT), Image.Resampling.BILINEAR
        )
        return np.asarray(value, dtype=np.uint8)
    if degradation == "gamma_lift_validation":
        normalized = np.clip(array / 255.0, 0.0, 1.0)
        return np.clip(np.power(normalized, 0.82) * 255.0, 0, 255).astype(np.uint8)
    if degradation == "row_streak_validation":
        streak = np.zeros_like(array)
        streak[::13] = rng.uniform(-8.0, 5.0, size=(len(streak[::13]), 1))
        return np.clip(array + streak, 0, 255).astype(np.uint8)
    if degradation == "compact_resample_validation":
        value = image.resize((176, 132), Image.Resampling.BILINEAR).resize(
            (WIDTH, HEIGHT), Image.Resampling.NEAREST
        )
        return np.asarray(value, dtype=np.uint8)
    if degradation == "mixed_focus_public":
        sharp = ImageEnhance.Sharpness(image.filter(ImageFilter.GaussianBlur(0.35))).enhance(1.35)
        return np.asarray(sharp, dtype=np.uint8)
    if degradation == "background_ramp_public":
        ramp = np.linspace(-9.0, 12.0, array.shape[0])[:, None]
        return np.clip(array + ramp, 0, 255).astype(np.uint8)
    if degradation == "anisotropic_scan_public":
        value = image.resize((238, 169), Image.Resampling.BILINEAR).resize(
            (WIDTH, HEIGHT), Image.Resampling.BICUBIC
        )
        return np.asarray(ImageEnhance.Contrast(value).enhance(0.86), dtype=np.uint8)
    return array.astype(np.uint8)


def build_scene(
    *,
    split: str,
    family: str,
    degradation: str,
    variant: int,
    seed: int,
) -> LineAwareScene:
    rng = np.random.default_rng(seed)
    image = Image.new("L", (WIDTH, HEIGHT), 250)
    text_image = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    text = ImageDraw.Draw(text_image)
    artifact = ImageDraw.Draw(artifact_image)
    prohibited: list[ProhibitedPoint] = []

    axis_x = 31 + (variant % 3)
    axis_y = 164 - (variant % 4)
    draw.line((axis_x, 16, axis_x, axis_y), fill=16, width=2)
    draw.line((axis_x, axis_y, 244, axis_y), fill=16, width=2)
    artifact.line((axis_x, 13, axis_x, axis_y + 5), fill=255, width=7)
    artifact.line((axis_x - 4, axis_y, 247, axis_y), fill=255, width=7)
    prohibited.extend(
        (
            ProhibitedPoint("axis", float(axis_x), 88.0),
            ProhibitedPoint("axis", 134.0, float(axis_y)),
        )
    )
    for tick_x in range(52, 241, 24):
        draw.line((tick_x, axis_y - 3, tick_x, axis_y + 4), fill=22, width=1)
        artifact.rectangle((tick_x - 3, axis_y - 5, tick_x + 3, axis_y + 6), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(tick_x), float(axis_y)))
    for tick_y in range(31, 151, 20):
        draw.line((axis_x - 4, tick_y, axis_x + 4, tick_y), fill=22, width=1)
        artifact.rectangle((axis_x - 6, tick_y - 3, axis_x + 6, tick_y + 3), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(axis_x), float(tick_y)))

    dividers = (113 + (variant % 3), 183 - (variant % 2))
    for divider_x in dividers:
        draw.line((divider_x, 17, divider_x, axis_y), fill=45, width=1)
        artifact.rectangle((divider_x - 3, 14, divider_x + 3, axis_y), fill=255)
        prohibited.append(ProhibitedPoint("divider", float(divider_x), 62.0))

    draw.rectangle((184, 19, 245, 48), outline=40, width=1)
    artifact.rectangle((181, 16, 248, 51), outline=255, width=7)
    prohibited.append(ProhibitedPoint("legend", 214.0, 34.0))
    prohibited.extend(_draw_glyph_run(draw, text, 188, 27, 8))

    draw.line((49, 36, 69, 17, 89, 36), fill=30, width=2)
    artifact.line((46, 39, 69, 14, 92, 39), fill=255, width=7)
    prohibited.append(ProhibitedPoint("bracket", 69.0, 23.0))
    draw.line((101, 39, 120, 22), fill=30, width=2)
    draw.polygon(((120, 22), (111, 24), (118, 31)), fill=30)
    artifact.line((98, 42, 123, 19), fill=255, width=7)
    artifact.polygon(((123, 19), (108, 21), (118, 35)), fill=255)
    prohibited.extend(
        (
            ProhibitedPoint("arrow_shaft", 110.0, 31.0),
            ProhibitedPoint("arrowhead", 118.0, 27.0),
        )
    )
    prohibited.extend(_draw_glyph_run(draw, text, 43, 174, 10))
    prohibited.extend(_draw_glyph_run(draw, text, 4, 68, 4))

    centers = _centers(family, variant, rng)
    radii = [3 + ((index + variant) % 4) for index in range(len(centers))]
    draw.line(centers, fill=52, width=2, joint="curve")
    artifact.line(centers, fill=255, width=7, joint="curve")
    for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        _draw_marker(draw, center, radius, index + variant)
        x, y = center
        clear = radius + 3
        artifact.ellipse((x - clear, y - clear, x + clear, y + clear), fill=0)
    for divider_x in dividers:
        intersection_y = int(
            np.interp(
                divider_x,
                [point[0] for point in centers],
                [point[1] for point in centers],
            )
        )
        if min((divider_x - x) ** 2 + (intersection_y - y) ** 2 for x, y in centers) > 81:
            prohibited.append(
                ProhibitedPoint("line_intersection", float(divider_x), float(intersection_y))
            )

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
    return tuple(
        build_scene(
            split=split,
            family=family,
            degradation=DEGRADATIONS[split][
                (family_index + variant) % len(DEGRADATIONS[split])
            ],
            variant=variant,
            seed=SELECTION_SEED_BASE[split] + (family_index * 100) + variant,
        )
        for family_index, family in enumerate(SELECTION_FAMILIES[split])
        for variant in range(SELECTION_VARIANTS[split])
    )


def build_sealed_public_scenes(secret_seed: int) -> tuple[LineAwareScene, ...]:
    return tuple(
        build_scene(
            split="sealed_public",
            family=family,
            degradation=DEGRADATIONS["sealed_public"][
                (family_index + variant) % len(DEGRADATIONS["sealed_public"])
            ],
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
        result.update(
            {
                "seed": scene.seed,
                "centers": [[x, y] for x, y in scene.centers],
                "radii": list(scene.radii),
                "prohibited": [
                    {"kind": item.kind, "x": item.x, "y": item.y}
                    for item in scene.prohibited
                ],
            }
        )
    return result


def selection_manifest() -> dict[str, object]:
    return {
        "schema": "graphreader.marker-center-runtime-consistency-selection.v2",
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


def save_sealed_public_archive(
    scenes: Iterable[LineAwareScene],
    path: Path,
) -> dict[str, object]:
    ordered = tuple(scenes)
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, object]] = []
    for index, scene in enumerate(ordered):
        prefix = f"scene_{index:03d}"
        arrays[f"{prefix}_tensor"] = scene.tensor.numpy().astype("<f4", copy=False)
        arrays[f"{prefix}_centers"] = np.asarray(scene.centers, dtype="<f4")
        arrays[f"{prefix}_radii"] = np.asarray(scene.radii, dtype="<f4")
        arrays[f"{prefix}_prohibited_xy"] = np.asarray(
            [(item.x, item.y) for item in scene.prohibited], dtype="<f4"
        )
        cases.append(scene_manifest(scene, expose_truth=True))
    metadata = {
        "schema": "graphreader.marker-center-runtime-consistency-sealed-fixtures.v2",
        "dataset_revision": DATASET_REVISION,
        "case_order": [scene.scene_id for scene in ordered],
        "cases": cases,
    }
    arrays["metadata_json_utf8"] = np.frombuffer(
        (
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8"),
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
            scenes.append(
                LineAwareScene(
                    scene_id=case["scene_id"],
                    split=case["split"],
                    family=case["family"],
                    degradation=case["degradation"],
                    seed=int(case["seed"]),
                    tensor=torch.from_numpy(archive[f"{prefix}_tensor"].copy()),
                    centers=tuple(
                        tuple(map(float, row))
                        for row in archive[f"{prefix}_centers"]
                    ),
                    radii=tuple(map(float, archive[f"{prefix}_radii"])),
                    prohibited=tuple(
                        ProhibitedPoint(item["kind"], float(x), float(y))
                        for item, (x, y) in zip(case["prohibited"], points, strict=True)
                    ),
                )
            )
    return tuple(scenes)


__all__ = [
    "DATASET_REVISION",
    "DEGRADATIONS",
    "HEIGHT",
    "SEALED_PUBLIC_FAMILIES",
    "SELECTION_FAMILIES",
    "WIDTH",
    "build_sealed_public_scenes",
    "build_selection_scenes",
    "load_sealed_public_archive",
    "save_sealed_public_archive",
    "selection_manifest",
]
