# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed, procedural, family/template-disjoint marker patch data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch


SHAPE_NAMES = (
    "circle",
    "square",
    "triangle_up",
    "triangle_down",
    "diamond",
    "star",
    "asterisk",
    "cross",
    "other",
)
FILL_NAMES = ("filled", "open", "unknown")
ARTIFACT_KINDS = (
    "text",
    "axis",
    "tick",
    "divider",
    "arrow",
    "bracket",
    "intersection",
    "legend",
)
SCENARIOS = (
    "isolated",
    "line_contact_horizontal",
    "line_contact_diagonal",
    "mixed_series_neighbor",
    "minority_probe",
)
SPLIT_FAMILIES = {
    "train": ("vector_thin", "press_heavy"),
    "validation": ("scan_soft",),
    "test": ("photocopy_oblique", "halftone_rough"),
}
SPLIT_TEMPLATES = {
    "train": ("compact_center", "wide_center"),
    "validation": ("high_offset",),
    "test": ("low_offset", "slender_offset"),
}
DATASET_REVISION = "marker-classifier-procedural-v1"
PATCH_SIZE = 32
REPEATS = {"train": 4, "validation": 4, "test": 3}


@dataclass(frozen=True)
class PatchSample:
    sample_id: str
    split: str
    family: str
    template: str
    scenario: str
    tensor: torch.Tensor
    shape_index: int
    fill_index: int
    artifact: float
    artifact_kind: str | None

    @property
    def marker_identity(self) -> str | None:
        if self.artifact >= 0.5:
            return None
        return f"{SHAPE_NAMES[self.shape_index]}:{FILL_NAMES[self.fill_index]}"


def _regular_polygon(cx: float, cy: float, radius: float, sides: int, rotation: float) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * math.cos(rotation + index * math.tau / sides),
            cy + radius * math.sin(rotation + index * math.tau / sides),
        )
        for index in range(sides)
    ]


def _star(cx: float, cy: float, outer: float, inner: float, rotation: float) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = rotation + index * math.pi / 5.0
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _template_geometry(template: str, repeat: int) -> tuple[float, float, float, float]:
    geometries = {
        "compact_center": (16.0, 16.0, 6.3, 1.00),
        "wide_center": (15.5, 16.5, 7.4, 0.88),
        "high_offset": (16.7, 14.7, 6.8, 0.94),
        "low_offset": (14.8, 17.4, 7.0, 1.04),
        "slender_offset": (17.2, 16.8, 7.5, 0.78),
    }
    cx, cy, radius, aspect = geometries[template]
    return cx + (repeat % 2) * 0.28, cy - ((repeat // 2) % 2) * 0.24, radius, aspect


def _draw_context(draw: ImageDraw.ImageDraw, scenario: str, cx: float, cy: float, scale: int) -> None:
    x = int(round(cx * scale))
    y = int(round(cy * scale))
    if scenario == "line_contact_horizontal":
        draw.line((1 * scale, y, 31 * scale, y), fill=52, width=max(2, scale))
    elif scenario == "line_contact_diagonal":
        draw.line((2 * scale, y + 8 * scale, 30 * scale, y - 7 * scale), fill=64, width=max(2, scale))
    elif scenario == "mixed_series_neighbor":
        draw.line((0, y + 6 * scale, x - 4 * scale, y + 1 * scale), fill=76, width=max(2, scale))
        draw.rectangle((1 * scale, (cy + 3) * scale, 4 * scale, (cy + 6) * scale), outline=80, width=scale)


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    shape: str,
    fill: str,
    geometry: tuple[float, float, float, float],
    scale: int,
    repeat: int,
) -> None:
    cx, cy, radius, aspect = geometry
    cx *= scale
    cy *= scale
    rx = radius * scale
    ry = radius * aspect * scale
    outline_width = max(3, int(round((1.25 + 0.18 * (repeat % 3)) * scale)))
    black = 10
    interior = black if fill == "filled" else (245 if fill == "open" else 132)
    box = (cx - rx, cy - ry, cx + rx, cy + ry)
    rotation = -math.pi / 2.0 + (repeat % 3 - 1) * 0.045

    if shape == "circle":
        draw.ellipse(box, fill=interior, outline=black, width=outline_width)
    elif shape == "square":
        draw.rectangle(box, fill=interior, outline=black, width=outline_width)
    elif shape in ("triangle_up", "triangle_down"):
        angle = rotation if shape == "triangle_up" else rotation + math.pi
        points = _regular_polygon(cx, cy, max(rx, ry), 3, angle)
        draw.polygon(points, fill=interior)
        draw.line(points + [points[0]], fill=black, width=outline_width, joint="curve")
    elif shape == "diamond":
        points = [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
        draw.polygon(points, fill=interior)
        draw.line(points + [points[0]], fill=black, width=outline_width, joint="curve")
    elif shape == "star":
        points = _star(cx, cy, max(rx, ry), max(rx, ry) * 0.43, rotation)
        draw.polygon(points, fill=interior)
        draw.line(points + [points[0]], fill=black, width=max(scale, outline_width - scale), joint="curve")
    elif shape == "asterisk":
        for arm in range(3):
            angle = repeat * 0.02 + arm * math.pi / 3.0
            dx = rx * math.cos(angle)
            dy = ry * math.sin(angle)
            draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=black, width=outline_width)
        if fill == "open":
            draw.ellipse((cx - 1.8 * scale, cy - 1.8 * scale, cx + 1.8 * scale, cy + 1.8 * scale), fill=245)
        elif fill == "unknown":
            draw.ellipse((cx - 1.5 * scale, cy - 1.5 * scale, cx + 1.5 * scale, cy + 1.5 * scale), fill=132)
    elif shape == "cross":
        half = outline_width * (1.25 if fill == "filled" else 0.85)
        points = [
            (cx - half, cy - ry), (cx + half, cy - ry), (cx + half, cy - half),
            (cx + rx, cy - half), (cx + rx, cy + half), (cx + half, cy + half),
            (cx + half, cy + ry), (cx - half, cy + ry), (cx - half, cy + half),
            (cx - rx, cy + half), (cx - rx, cy - half), (cx - half, cy - half),
        ]
        draw.polygon(points, fill=interior)
        draw.line(points + [points[0]], fill=black, width=max(scale, outline_width // 2))
    elif shape == "other":
        points = _regular_polygon(cx, cy, max(rx, ry), 6, rotation + math.pi / 6.0)
        draw.polygon(points, fill=interior)
        draw.line(points + [points[0]], fill=black, width=outline_width, joint="curve")
    else:
        raise ValueError(shape)

    if fill == "unknown":
        draw.line((cx - rx * 0.65, cy + ry * 0.45, cx + rx * 0.65, cy - ry * 0.45), fill=218, width=scale)


def _draw_artifact(draw: ImageDraw.ImageDraw, kind: str, scale: int, repeat: int) -> None:
    c = 16 * scale
    w = max(2, scale + repeat % 2)
    if kind == "text":
        draw.text((10 * scale, 8 * scale), "ab", fill=8, stroke_width=repeat % 2)
    elif kind == "axis":
        draw.line((5 * scale, 5 * scale, 5 * scale, 26 * scale, 28 * scale, 26 * scale), fill=8, width=w)
    elif kind == "tick":
        draw.line((4 * scale, c, 28 * scale, c), fill=8, width=w)
        draw.line((c, 12 * scale, c, 20 * scale), fill=8, width=w)
    elif kind == "divider":
        for y in range(3, 29, 5):
            draw.line((c, y * scale, c, (y + 2) * scale), fill=8, width=w)
    elif kind == "arrow":
        draw.line((4 * scale, 24 * scale, 22 * scale, 9 * scale), fill=8, width=w)
        draw.polygon(((22 * scale, 9 * scale), (15 * scale, 10 * scale), (21 * scale, 16 * scale)), fill=8)
    elif kind == "bracket":
        draw.line((8 * scale, 5 * scale, 8 * scale, 26 * scale, 24 * scale, 26 * scale), fill=8, width=w)
    elif kind == "intersection":
        draw.line((5 * scale, 5 * scale, 27 * scale, 27 * scale), fill=8, width=w)
        draw.line((5 * scale, 27 * scale, 27 * scale, 5 * scale), fill=8, width=w)
    elif kind == "legend":
        draw.rectangle((4 * scale, 6 * scale, 28 * scale, 25 * scale), outline=8, width=w)
        draw.ellipse((7 * scale, 12 * scale, 14 * scale, 19 * scale), fill=8)
        draw.line((16 * scale, 13 * scale, 25 * scale, 13 * scale), fill=8, width=w)
        draw.line((16 * scale, 18 * scale, 24 * scale, 18 * scale), fill=8, width=w)
    else:
        raise ValueError(kind)


def _degrade(image: Image.Image, family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "press_heavy":
        image = image.filter(ImageFilter.MinFilter(3))
    elif family == "scan_soft":
        image = image.filter(ImageFilter.GaussianBlur(0.48))
    elif family == "photocopy_oblique":
        image = image.transform(image.size, Image.Transform.AFFINE, (1.0, 0.05, -0.8, -0.025, 1.0, 0.4), resample=Image.Resampling.BICUBIC, fillcolor=255)
    elif family == "halftone_rough":
        image = image.resize((26, 26), Image.Resampling.BILINEAR).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    sigma = {"vector_thin": 0.004, "press_heavy": 0.006, "scan_soft": 0.010, "photocopy_oblique": 0.012, "halftone_rough": 0.014}[family]
    array = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
    if family == "photocopy_oblique":
        array = 0.08 + 0.84 * array
    elif family == "halftone_rough":
        selector = rng.random(array.shape)
        array = np.where(selector < 0.003, 0.0, np.where(selector > 0.997, 1.0, array))
    return (1.0 - array).astype(np.float32)


def _render_marker(split: str, family: str, template: str, shape_index: int, fill_index: int, repeat: int, seed: int) -> PatchSample:
    scale = 3
    image = Image.new("L", (PATCH_SIZE * scale, PATCH_SIZE * scale), 255)
    draw = ImageDraw.Draw(image)
    scenario = SCENARIOS[(shape_index * len(FILL_NAMES) + fill_index + repeat) % len(SCENARIOS)]
    if shape_index in (5, 6, 7) and repeat == 0:
        scenario = "minority_probe"
    geometry = _template_geometry(template, repeat)
    _draw_context(draw, scenario, geometry[0], geometry[1], scale)
    _draw_marker(draw, SHAPE_NAMES[shape_index], FILL_NAMES[fill_index], geometry, scale, repeat)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"{split}-{family}-{template}-{SHAPE_NAMES[shape_index]}-{FILL_NAMES[fill_index]}-{repeat}"
    return PatchSample(sample_id, split, family, template, scenario, tensor, shape_index, fill_index, 0.0, None)


def _render_artifact(split: str, family: str, template: str, kind: str, repeat: int, seed: int) -> PatchSample:
    scale = 3
    image = Image.new("L", (PATCH_SIZE * scale, PATCH_SIZE * scale), 255)
    draw = ImageDraw.Draw(image)
    _draw_artifact(draw, kind, scale, repeat)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"{split}-{family}-{template}-artifact-{kind}-{repeat}"
    return PatchSample(sample_id, split, family, template, kind, tensor, SHAPE_NAMES.index("other"), FILL_NAMES.index("unknown"), 1.0, kind)


def build_fixed_dataset(split: str) -> tuple[PatchSample, ...]:
    if split not in SPLIT_FAMILIES:
        raise ValueError(f"Unknown split {split!r}")
    samples: list[PatchSample] = []
    base = {"train": 110_000, "validation": 220_000, "test": 330_000}[split]
    ordinal = 0
    for family in SPLIT_FAMILIES[split]:
        for template in SPLIT_TEMPLATES[split]:
            for repeat in range(REPEATS[split]):
                for shape_index in range(len(SHAPE_NAMES)):
                    for fill_index in range(len(FILL_NAMES)):
                        samples.append(_render_marker(split, family, template, shape_index, fill_index, repeat, base + ordinal))
                        ordinal += 1
                for artifact_kind in ARTIFACT_KINDS:
                    samples.append(_render_artifact(split, family, template, artifact_kind, repeat, base + ordinal))
                    ordinal += 1
    return tuple(samples)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def dataset_manifest(*, include_test: bool = False) -> dict[str, object]:
    splits: Iterable[str] = ("train", "validation", "test") if include_test else ("train", "validation")
    cases = []
    for split in splits:
        for sample in build_fixed_dataset(split):
            cases.append(
                {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "family": sample.family,
                    "template": sample.template,
                    "scenario": sample.scenario,
                    "shape": SHAPE_NAMES[sample.shape_index],
                    "fill": FILL_NAMES[sample.fill_index],
                    "artifact": sample.artifact,
                    "artifact_kind": sample.artifact_kind,
                    "tensor_sha256": _tensor_sha256(sample.tensor),
                }
            )
    return {
        "revision": DATASET_REVISION,
        "patch_size": PATCH_SIZE,
        "included_splits": list(splits),
        "families": {split: SPLIT_FAMILIES[split] for split in splits},
        "templates": {split: SPLIT_TEMPLATES[split] for split in splits},
        "cases": cases,
    }


def seal_dataset_manifest(output_dir: Path) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "selection-dataset-manifest.json"
    payload = json.dumps(dataset_manifest(include_test=False), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ARTIFACT_KINDS",
    "DATASET_REVISION",
    "FILL_NAMES",
    "PATCH_SIZE",
    "PatchSample",
    "REPEATS",
    "SCENARIOS",
    "SHAPE_NAMES",
    "SPLIT_FAMILIES",
    "SPLIT_TEMPLATES",
    "build_fixed_dataset",
    "dataset_manifest",
    "seal_dataset_manifest",
]
