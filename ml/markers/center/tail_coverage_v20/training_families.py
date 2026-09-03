# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic synthetic-only V20 train families."""

from __future__ import annotations

from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from ml.markers.center.line_aware_v1.dataset import LineAwareScene, ProhibitedPoint
from ml.markers.center.proposal_geometry_v13 import dataset as v13_dataset
from ml.markers.center.train_family_v19.training_families import build_train_scenes as build_v19_train_scenes

from .protocol import (
    TAIL_TRAIN_SEED_BASE,
    TRAIN_FAMILY_SPECS,
    TRAIN_VARIANTS_PER_FAMILY,
)


def _roundtrip_channel(channel: np.ndarray, scale: float, *, quality: int, blur: float) -> np.ndarray:
    image = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8), mode="L")
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    width, height = image.size
    reduced = image.resize((max(8, round(width * scale)), max(8, round(height * scale))), Image.Resampling.BILINEAR)
    restored = reduced.resize((width, height), Image.Resampling.BICUBIC)
    if quality < 90:
        encoded = BytesIO()
        restored.convert("RGB").save(encoded, format="JPEG", quality=quality, optimize=False, subsampling=2)
        restored = Image.open(BytesIO(encoded.getvalue())).convert("L")
    return np.asarray(restored, dtype=np.float32) / 255.0


def _augment_scene(scene: LineAwareScene, *, scale: float, quality: int, rgba: bool, seed: int) -> LineAwareScene:
    rng = np.random.default_rng(seed)
    ink = 1.0 - scene.tensor[0].numpy()
    ink = _roundtrip_channel(ink, scale, quality=quality, blur=0.10 if scale < 0.5 else 0.0)
    if rgba:
        alpha = 0.82 + 0.12 * float(rng.random())
        ink = 1.0 - ((1.0 - ink) * alpha + (1.0 - alpha))
    ink = np.clip(ink + rng.normal(0.0, 0.0025, ink.shape), 0.0, 1.0)
    text = _roundtrip_channel(scene.tensor[1].numpy(), scale, quality=100, blur=0.0)
    artifact = _roundtrip_channel(scene.tensor[2].numpy(), scale, quality=100, blur=0.0)
    tensor = torch.from_numpy(np.stack((1.0 - ink, text, artifact), axis=0).astype(np.float32, copy=False))
    return LineAwareScene(
        scene_id=scene.scene_id, split="train", family=scene.family, degradation=scene.degradation,
        seed=scene.seed, tensor=tensor, centers=scene.centers, radii=scene.radii,
        prohibited=scene.prohibited,
    )


def _draw_marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, shape: str, index: int) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    width = max(1, radius // 4)
    if shape == "open_square":
        draw.rectangle(box, fill=246, outline=12, width=width)
    elif index % 2:
        draw.ellipse(box, fill=18, outline=8, width=width)
    else:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), fill=18, outline=8)


def _clear_marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: float) -> None:
    x, y = center
    extent = int(math.ceil(radius)) + 2
    draw.rectangle((x - extent, y - extent, x + extent, y + extent), fill=255)


def _tail_overlay(scene: LineAwareScene, *, tail_variant: str, variant: int) -> LineAwareScene:
    ink = np.clip(1.0 - scene.tensor[0].numpy(), 0.0, 1.0)
    image = Image.fromarray(np.rint(ink * 255.0).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    centers = tuple((int(round(x)), int(round(y))) for x, y in scene.centers)
    radii = list(scene.radii)
    prohibited = list(scene.prohibited)
    if tail_variant == "open_square":
        radii = [float(3 + ((index + variant) % 3)) for index in range(len(centers))]
        for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
            _clear_marker(draw, center, max(float(scene.radii[index]), radius))
            _draw_marker(draw, center, int(radius), "open_square", index)
    elif tail_variant == "radius_11_12":
        radii = [float(11 + ((index + variant) % 2)) for index in range(len(centers))]
        for index, (center, radius) in enumerate(zip(centers, radii, strict=True)):
            _clear_marker(draw, center, max(float(scene.radii[index]), radius))
            _draw_marker(draw, center, int(radius), "mixed", index + variant)
    elif tail_variant == "intersection_heavy":
        intersections = tuple((x, y) for y in (42, 126) for x in (60, 84, 108, 132))
        for x, y in intersections:
            draw.line((x - 8, y, x + 8, y), fill=28, width=2)
            draw.line((x, y - 8, x, y + 8), fill=28, width=2)
            prohibited.append(ProhibitedPoint("line_intersection", float(x), float(y)))
    else:
        raise ValueError(f"Unknown V20 tail variant: {tail_variant!r}")
    values = np.asarray(image, dtype=np.float32) / 255.0
    artifact = scene.tensor[2].numpy().copy()
    if tail_variant == "intersection_heavy":
        artifact_image = Image.fromarray(np.rint(np.clip(artifact, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        artifact_draw = ImageDraw.Draw(artifact_image)
        for point in prohibited:
            if point.kind == "line_intersection":
                artifact_draw.ellipse((point.x - 9, point.y - 9, point.x + 9, point.y + 9), fill=255)
        artifact = np.asarray(artifact_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.stack((1.0 - values, scene.tensor[1].numpy(), artifact), axis=0).astype(np.float32, copy=False))
    return LineAwareScene(
        scene_id=scene.scene_id, split="train", family=scene.family, degradation=scene.degradation,
        seed=scene.seed, tensor=tensor, centers=scene.centers, radii=tuple(radii),
        prohibited=tuple(prohibited),
    )


def geometry_consensus_veto_guard(scene: LineAwareScene) -> bool:
    """Keep synthetic line intersections away from truth centers."""
    intersections = [item for item in scene.prohibited if item.kind == "line_intersection"]
    return all(
        math.hypot(point.x - center[0], point.y - center[1]) > 3.0
        for point in intersections
        for center in scene.centers
    )


def _family_source(family: str) -> str:
    if family == "tail_open_square_train":
        return "geometry_small_train"
    if family == "tail_radius_11_12_train":
        return "geometry_medium_train"
    if family == "tail_intersection_heavy_train":
        return "geometry_joint_train"
    raise ValueError(f"V20 tail source is undefined for {family!r}")


def build_train_scenes() -> tuple[LineAwareScene, ...]:
    """Build V19's broad families plus fresh, disjoint V20 tail variants."""
    scenes: list[LineAwareScene] = list(build_v19_train_scenes())
    for family_index, (family, spec) in enumerate(TRAIN_FAMILY_SPECS.items()):
        tail_variant = spec["tail_variant"]
        if not tail_variant:
            continue
        scale_min, scale_max = map(float, spec["resize_long_scale_range"])
        quality_min, quality_max = map(int, spec["jpeg_quality_range"])
        for variant in range(TRAIN_VARIANTS_PER_FAMILY):
            seed = TAIL_TRAIN_SEED_BASE + family_index * 100 + variant
            base = v13_dataset.build_scene(split="train", family=_family_source(family), variant=variant, seed=seed)
            fraction = variant / max(1, TRAIN_VARIANTS_PER_FAMILY - 1)
            scale = scale_min + (scale_max - scale_min) * fraction
            quality = quality_min + round((quality_max - quality_min) * fraction)
            rgba = "RGBA" in spec["color_modes"] and (variant % 2 == 1)
            transformed = _augment_scene(base, scale=scale, quality=quality, rgba=rgba, seed=seed + 1)
            transformed = _tail_overlay(transformed, tail_variant=tail_variant, variant=variant)
            scene = LineAwareScene(
                scene_id=f"train-{family}-{variant}", split="train", family=family,
                degradation=f"resize_{scale:.2f}_{'rgba' if rgba else 'rgb'}_jpeg_{quality}",
                seed=seed, tensor=transformed.tensor, centers=transformed.centers,
                radii=transformed.radii, prohibited=transformed.prohibited,
            )
            if not geometry_consensus_veto_guard(scene):
                raise RuntimeError(f"V20 geometry-consensus veto guard failed for {scene.scene_id}")
            scenes.append(scene)
    return tuple(scenes)


__all__ = ["build_train_scenes", "geometry_consensus_veto_guard"]
