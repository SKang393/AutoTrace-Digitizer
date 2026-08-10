# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Procedural, family-disjoint scenes for candidate-level marker detection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch


WIDTH = 192
HEIGHT = 144
DATASET_REVISION = "marker-center-candidate-level-procedural-v1"
SELECTION_FAMILIES = {
    "train": (
        "rising_clusters_train",
        "alternating_plateau_train",
        "descending_probe_train",
        "shared_baseline_train",
        "dense_turn_train",
        "sparse_open_train",
    ),
    "validation": (
        "reversal_arc_validation",
        "maintenance_steps_validation",
        "paired_phase_validation",
    ),
}
SEALED_PUBLIC_FAMILIES = (
    "mixed_open_filled_public",
    "stair_connector_public",
    "clustered_generalization_public",
    "small_marker_public",
)
DEGRADATIONS = {
    "train": ("clean_train", "low_contrast_train", "thin_scan_train"),
    "validation": ("box_blur_validation", "uneven_gray_validation"),
    "sealed_public": ("resample_noise_public", "high_contrast_public"),
}
SELECTION_SEED_BASE = {"train": 521_000, "validation": 622_000}
SELECTION_VARIANTS = {"train": 4, "validation": 3}
SEALED_PUBLIC_VARIANTS = 4


@dataclass(frozen=True)
class ProhibitedPoint:
    kind: str
    x: float
    y: float


@dataclass(frozen=True)
class CandidateScene:
    scene_id: str
    split: str
    family: str
    degradation: str
    seed: int
    tensor: torch.Tensor
    centers: tuple[tuple[float, float], ...]
    radii: tuple[float, ...]
    prohibited: tuple[ProhibitedPoint, ...]


def _draw_pseudo_text(
    image: ImageDraw.ImageDraw,
    mask: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    glyphs: int,
) -> list[ProhibitedPoint]:
    points: list[ProhibitedPoint] = []
    for index in range(glyphs):
        left = x + (index * 5)
        top = y + (index % 2)
        image.rectangle((left, top, left + 2, top + 7), fill=20)
        image.line((left, top, left + 4, top + 3), fill=20, width=1)
        mask.rectangle((left - 1, top - 1, left + 5, top + 8), fill=255)
        points.append(ProhibitedPoint("text", float(left + 2), float(top + 4)))
    return points


def _family_centers(family: str, variant: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    xs = np.linspace(42, 166, 7, dtype=np.int32)
    jitter = rng.integers(-2, 3, size=(7, 2), endpoint=False)
    if "rising" in family or "reversal" in family:
        ys = np.array((103, 92, 79, 67, 56, 69, 82), dtype=np.int32)
    elif "alternating" in family or "mixed" in family:
        ys = np.array((92, 62, 91, 61, 90, 60, 89), dtype=np.int32)
    elif "descending" in family or "stair" in family:
        ys = np.array((48, 57, 57, 70, 70, 84, 96), dtype=np.int32)
    elif "shared" in family or "paired" in family:
        ys = np.array((94, 84, 73, 73, 58, 58, 43), dtype=np.int32)
    elif "dense" in family or "clustered" in family:
        xs = np.array((43, 60, 77, 94, 111, 128, 145, 162), dtype=np.int32)
        ys = np.array((86, 77, 69, 76, 62, 55, 63, 47), dtype=np.int32)
        jitter = rng.integers(-2, 3, size=(8, 2), endpoint=False)
    elif "small" in family:
        ys = np.array((98, 81, 67, 52, 65, 79, 91), dtype=np.int32)
    else:
        ys = np.array((88, 73, 61, 76, 58, 71, 49), dtype=np.int32)
    centers = []
    for index, (x_value, y_value) in enumerate(zip(xs, ys, strict=True)):
        dx, dy = jitter[index]
        centers.append((int(x_value + dx + (variant % 2)), int(y_value + dy)))
    return centers


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    marker_index: int,
) -> None:
    x, y = center
    bounds = (x - radius, y - radius, x + radius, y + radius)
    shape = marker_index % 4
    filled = marker_index % 3 != 1
    fill = 12 if filled else 245
    if shape == 0:
        draw.ellipse(bounds, outline=12, fill=fill, width=2)
    elif shape == 1:
        draw.rectangle(bounds, outline=12, fill=fill, width=2)
    elif shape == 2:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), outline=12, fill=fill)
    else:
        draw.polygon(((x, y - radius), (x + radius, y + radius), (x - radius, y + radius)), outline=12, fill=fill)


def _apply_degradation(image: Image.Image, degradation: str, rng: np.random.Generator) -> np.ndarray:
    value = image
    if degradation == "low_contrast_train":
        value = ImageEnhance.Contrast(value).enhance(0.68)
    elif degradation == "thin_scan_train":
        value = value.filter(ImageFilter.GaussianBlur(radius=0.35))
        value = ImageEnhance.Contrast(value).enhance(0.82)
    elif degradation == "box_blur_validation":
        value = value.filter(ImageFilter.BoxBlur(radius=0.55))
    elif degradation == "uneven_gray_validation":
        array = np.asarray(value, dtype=np.float32)
        gradient = np.linspace(0, 18, array.shape[1], dtype=np.float32)[None, :]
        return np.clip(array - gradient, 0, 255).astype(np.uint8)
    elif degradation == "resample_noise_public":
        value = value.resize((144, 108), Image.Resampling.BILINEAR).resize(
            (WIDTH, HEIGHT), Image.Resampling.BILINEAR
        )
        array = np.asarray(value, dtype=np.float32)
        noise = rng.normal(0, 2.5, size=array.shape)
        return np.clip(array + noise, 0, 255).astype(np.uint8)
    elif degradation == "high_contrast_public":
        value = ImageEnhance.Contrast(value).enhance(1.28)
        value = value.filter(ImageFilter.GaussianBlur(radius=0.25))
    return np.asarray(value, dtype=np.uint8)


def build_scene(*, split: str, family: str, degradation: str, variant: int, seed: int) -> CandidateScene:
    if split not in {"train", "validation", "sealed_public"}:
        raise ValueError(f"Unsupported split {split!r}")
    rng = np.random.default_rng(seed)
    image = Image.new("L", (WIDTH, HEIGHT), 250)
    text_mask_image = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_mask_image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    text_mask = ImageDraw.Draw(text_mask_image)
    artifact_mask = ImageDraw.Draw(artifact_mask_image)
    prohibited: list[ProhibitedPoint] = []

    axis_x = 27 + (variant % 2)
    axis_y = 119 - (variant % 3)
    draw.line((axis_x, 18, axis_x, axis_y), fill=22, width=2)
    draw.line((axis_x, axis_y, 178, axis_y), fill=22, width=2)
    artifact_mask.line((axis_x - 2, 15, axis_x + 2, axis_y + 2), fill=255, width=5)
    artifact_mask.line((axis_x - 2, axis_y - 2, 180, axis_y + 2), fill=255, width=5)
    prohibited.extend(
        (
            ProhibitedPoint("axis", float(axis_x), 62.0),
            ProhibitedPoint("axis", 100.0, float(axis_y)),
        )
    )
    for tick_x in range(43, 175, 22):
        draw.line((tick_x, axis_y - 3, tick_x, axis_y + 3), fill=20, width=1)
        artifact_mask.rectangle((tick_x - 2, axis_y - 5, tick_x + 2, axis_y + 5), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(tick_x), float(axis_y)))
    for tick_y in range(31, 112, 20):
        draw.line((axis_x - 3, tick_y, axis_x + 3, tick_y), fill=20, width=1)
        artifact_mask.rectangle((axis_x - 5, tick_y - 2, axis_x + 5, tick_y + 2), fill=255)
        prohibited.append(ProhibitedPoint("tick", float(axis_x), float(tick_y)))

    divider_x = 104 + ((variant % 3) * 5)
    draw.line((divider_x, 20, divider_x, axis_y), fill=50, width=1)
    artifact_mask.rectangle((divider_x - 2, 17, divider_x + 2, axis_y), fill=255)
    prohibited.append(ProhibitedPoint("divider", float(divider_x), 47.0))

    draw.rectangle((132, 23, 177, 43), outline=45, width=1)
    artifact_mask.rectangle((130, 21, 179, 45), outline=255, width=5)
    prohibited.append(ProhibitedPoint("legend", 155.0, 33.0))
    prohibited.extend(_draw_pseudo_text(draw, text_mask, 137, 28, glyphs=6))

    draw.line((52, 26, 65, 17, 77, 26), fill=38, width=2)
    artifact_mask.line((50, 27, 65, 15, 79, 27), fill=255, width=5)
    prohibited.append(ProhibitedPoint("bracket", 65.0, 19.0))
    draw.line((81, 32, 91, 24), fill=35, width=2)
    draw.polygon(((91, 24), (84, 24), (89, 30)), fill=35)
    artifact_mask.line((79, 34, 93, 22), fill=255, width=6)
    artifact_mask.polygon(((91, 22), (82, 22), (89, 32)), fill=255)
    prohibited.append(ProhibitedPoint("arrowhead", 89.0, 26.0))
    prohibited.extend(_draw_pseudo_text(draw, text_mask, 31, 124, glyphs=7))
    prohibited.extend(_draw_pseudo_text(draw, text_mask, 5, 51, glyphs=3))

    centers = _family_centers(family, variant, rng)
    if split == "sealed_public" and family == "clustered_generalization_public":
        centers.extend(((149, 83), (164, 72)))
    radii = [3 + ((index + variant) % 3) for index in range(len(centers))]
    if family == "small_marker_public":
        radii = [3 for _ in centers]
    draw.line(centers, fill=48, width=2, joint="curve")
    artifact_mask.line(centers, fill=255, width=5, joint="curve")
    for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        _draw_marker(draw, center, radius, index + variant)
        x, y = center
        clear_radius = radius + 4
        artifact_mask.ellipse(
            (x - clear_radius, y - clear_radius, x + clear_radius, y + clear_radius),
            fill=0,
        )

    intersection = (divider_x, int(np.interp(divider_x, [point[0] for point in centers], [point[1] for point in centers])))
    if min((intersection[0] - x) ** 2 + (intersection[1] - y) ** 2 for x, y in centers) > 100:
        prohibited.append(ProhibitedPoint("line_intersection", float(intersection[0]), float(intersection[1])))

    array = _apply_degradation(image, degradation, rng)
    ink = 1.0 - (array.astype(np.float32) / 255.0)
    text_values = np.asarray(text_mask_image, dtype=np.float32) / 255.0
    artifact_values = np.asarray(artifact_mask_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.stack((ink, text_values, artifact_values), axis=0).copy())
    return CandidateScene(
        scene_id=f"{split}-{family}-{variant}",
        split=split,
        family=family,
        degradation=degradation,
        seed=seed,
        tensor=tensor,
        centers=tuple((float(x), float(y)) for x, y in centers),
        radii=tuple(float(value) for value in radii),
        prohibited=tuple(prohibited),
    )


def build_selection_scenes(split: str) -> tuple[CandidateScene, ...]:
    if split not in SELECTION_FAMILIES:
        raise ValueError(f"Unsupported selection split {split!r}")
    scenes: list[CandidateScene] = []
    families = SELECTION_FAMILIES[split]
    degradations = DEGRADATIONS[split]
    for family_index, family in enumerate(families):
        for variant in range(SELECTION_VARIANTS[split]):
            scenes.append(
                build_scene(
                    split=split,
                    family=family,
                    degradation=degradations[(family_index + variant) % len(degradations)],
                    variant=variant,
                    seed=SELECTION_SEED_BASE[split] + (family_index * 100) + variant,
                )
            )
    return tuple(scenes)


def build_sealed_public_scenes(secret_seed: int) -> tuple[CandidateScene, ...]:
    scenes: list[CandidateScene] = []
    degradations = DEGRADATIONS["sealed_public"]
    for family_index, family in enumerate(SEALED_PUBLIC_FAMILIES):
        for variant in range(SEALED_PUBLIC_VARIANTS):
            scenes.append(
                build_scene(
                    split="sealed_public",
                    family=family,
                    degradation=degradations[(family_index + variant) % len(degradations)],
                    variant=variant,
                    seed=secret_seed + (family_index * 100) + variant,
                )
            )
    return tuple(scenes)


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def scene_manifest(scene: CandidateScene, *, expose_truth: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "scene_id": scene.scene_id,
        "split": scene.split,
        "family": scene.family,
        "degradation": scene.degradation,
        "tensor_sha256": _sha256_array(scene.tensor.numpy().astype("<f4", copy=False)),
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
    cases = [
        scene_manifest(scene, expose_truth=True)
        for split in ("train", "validation")
        for scene in build_selection_scenes(split)
    ]
    return {
        "schema": "graphreader.marker-center-candidate-selection.v1",
        "dataset_revision": DATASET_REVISION,
        "public_or_private_images": False,
        "synthetic_only": True,
        "chandler_included": False,
        "split_families": {key: list(value) for key, value in SELECTION_FAMILIES.items()},
        "degradations": {key: list(value) for key, value in DEGRADATIONS.items()},
        "cases": cases,
    }


def save_sealed_public_archive(scenes: Iterable[CandidateScene], output_path: Path) -> dict[str, object]:
    ordered = tuple(scenes)
    arrays: dict[str, np.ndarray] = {}
    private_cases: list[dict[str, object]] = []
    for index, scene in enumerate(ordered):
        prefix = f"scene_{index:03d}"
        arrays[f"{prefix}_tensor"] = scene.tensor.numpy().astype("<f4", copy=False)
        arrays[f"{prefix}_centers"] = np.asarray(scene.centers, dtype="<f4")
        arrays[f"{prefix}_radii"] = np.asarray(scene.radii, dtype="<f4")
        arrays[f"{prefix}_prohibited_xy"] = np.asarray(
            [(item.x, item.y) for item in scene.prohibited], dtype="<f4"
        )
        private_cases.append(scene_manifest(scene, expose_truth=True))
    metadata = {
        "schema": "graphreader.marker-center-sealed-public-fixtures.v1",
        "dataset_revision": DATASET_REVISION,
        "case_order": [scene.scene_id for scene in ordered],
        "cases": private_cases,
    }
    arrays["metadata_json_utf8"] = np.frombuffer(
        (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        dtype=np.uint8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return metadata


def load_sealed_public_archive(path: Path) -> tuple[CandidateScene, ...]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_json_utf8"].tolist()).decode("utf-8"))
        scenes: list[CandidateScene] = []
        for index, case in enumerate(metadata["cases"]):
            prefix = f"scene_{index:03d}"
            prohibited_xy = archive[f"{prefix}_prohibited_xy"]
            prohibited = tuple(
                ProhibitedPoint(item["kind"], float(x), float(y))
                for item, (x, y) in zip(case["prohibited"], prohibited_xy, strict=True)
            )
            scenes.append(
                CandidateScene(
                    scene_id=case["scene_id"],
                    split=case["split"],
                    family=case["family"],
                    degradation=case["degradation"],
                    seed=int(case["seed"]),
                    tensor=torch.from_numpy(archive[f"{prefix}_tensor"].copy()),
                    centers=tuple(tuple(map(float, row)) for row in archive[f"{prefix}_centers"]),
                    radii=tuple(map(float, archive[f"{prefix}_radii"])),
                    prohibited=prohibited,
                )
            )
    return tuple(scenes)


__all__ = [
    "CandidateScene",
    "DATASET_REVISION",
    "DEGRADATIONS",
    "HEIGHT",
    "ProhibitedPoint",
    "SEALED_PUBLIC_FAMILIES",
    "SEALED_PUBLIC_VARIANTS",
    "SELECTION_FAMILIES",
    "WIDTH",
    "build_sealed_public_scenes",
    "build_selection_scenes",
    "load_sealed_public_archive",
    "save_sealed_public_archive",
    "scene_manifest",
    "selection_manifest",
]
