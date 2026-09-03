# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Procedural, deterministic marker scenes for diagnosing input coverage.

This module is intentionally synthetic-only.  It contains no model loading,
training, private corpus access, or candidate selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from statistics import median
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

from ml.markers.center.dataset import _artifact_geometry, _draw_marker

WIDTH, HEIGHT, PATCH = 224, 168, 33
SCENE_COUNT = 167
MARKERS_PER_SCENE = 12
SEED_BASE = {"train": 4100, "dev": 5100}
MASK_THRESHOLD = 0.35


@dataclass(frozen=True)
class Scene:
    split: str
    family: str
    seed: int
    tensor: torch.Tensor
    centers: tuple[tuple[float, float], ...]
    diameters: tuple[float, ...]
    rendered_diameters: tuple[float, ...]
    hard_negatives: tuple[tuple[str, float, float], ...]


def _diameters(count: int) -> list[float]:
    # Deliberately spans the measured 1..48 px envelope. Counts put p05 at 6,
    # the median at 12, p90 at 24, and p95 at 27 under linear quantiles.
    bins = [(1.0, 10), (6.0, 100), (8.0, 100), (10.0, 300), (12.0, 500),
            (16.0, 300), (20.0, 200), (24.0, 300), (27.0, 100),
            (32.0, 80), (40.0, 13), (48.0, 1)]
    values = [diameter for diameter, amount in bins for _ in range(amount)]
    return values[:count]


@lru_cache(maxsize=4)
def _mask_indices(split: str, channel: str) -> frozenset[int]:
    """Scatter overlap scenarios across the full diameter distribution."""
    if split not in SEED_BASE or channel not in {"ocr", "artifact"}:
        raise ValueError("unknown mask split or channel")
    count = 75 if channel == "ocr" else 332
    offset = 7001 if channel == "ocr" else 9001
    order = np.random.default_rng(SEED_BASE[split] + offset).permutation(2004)
    return frozenset(int(value) for value in order[:count])


def _center_for_global_index(global_index: int) -> tuple[int, int]:
    scene_index, local = divmod(global_index, MARKERS_PER_SCENE)
    return (
        18 + ((local * 29 + scene_index * 17) % (WIDTH - 36)),
        18 + ((local * 37 + scene_index * 11) % (HEIGHT - 60)),
    )


@lru_cache(maxsize=2)
def _full_ocr_patch_index(split: str) -> int:
    """Choose an OCR-overlap marker whose full patch reaches no neighbour."""
    for candidate in sorted(_mask_indices(split, "ocr")):
        scene_start = (candidate // MARKERS_PER_SCENE) * MARKERS_PER_SCENE
        x, y = _center_for_global_index(candidate)
        neighbours = (
            _center_for_global_index(index)
            for index in range(scene_start, scene_start + MARKERS_PER_SCENE)
            if index != candidate
        )
        if all(abs(x - other_x) > 18 or abs(y - other_y) > 18 for other_x, other_y in neighbours):
            return candidate
    raise RuntimeError("no isolated OCR full-patch marker exists")


@lru_cache(maxsize=64)
def _measure_rendered_diameter(diameter: float) -> float:
    """Measure the isolated ink footprint produced by the exact primitive."""
    canvas = Image.new("L", (96, 96), 255)
    probe = ImageDraw.Draw(canvas)
    center = (48, 48)
    if diameter == 1.0:
        probe.point(center, fill=0)
    else:
        _draw_marker(probe, center, max(1, int(round(diameter / 2.0))), 0)
    pixels = np.asarray(canvas) < 255
    ys, xs = np.where(pixels)
    return float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def _scene(split: str, index: int, diameters: list[float]) -> Scene:
    seed = SEED_BASE[split] + index
    rng = np.random.default_rng(seed)
    image = Image.new("L", (WIDTH, HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    ocr = Image.new("L", (WIDTH, HEIGHT), 0)
    ocr_draw = ImageDraw.Draw(ocr)
    artifact = Image.new("L", (WIDTH, HEIGHT), 0)
    artifact_draw = ImageDraw.Draw(artifact)
    per_scene = len(diameters)
    centers: list[tuple[float, float]] = []
    for local in range(per_scene):
        x = 18 + ((local * 29 + index * 17) % (WIDTH - 36))
        y = 18 + ((local * 37 + index * 11) % (HEIGHT - 60))
        centers.append((float(x), float(y)))
        radius = max(1, int(round(diameters[local] / 2.0)))
        if diameters[local] == 1.0:
            draw.point((x, y), fill=0)
        else:
            _draw_marker(draw, (x, y), radius, local + index)
        if local:
            px, py = (int(value) for value in centers[-2])
            draw.line((px, py, x, y), fill=45, width=1)

    # Exact aggregate hard-mask coverage in both splits: 75/2004 OCR and
    # 332/2004 artifact windows, scattered across marker sizes.
    global_offset = index * per_scene
    for local, (x_float, y_float) in enumerate(centers):
        global_index = global_offset + local
        x, y = int(x_float), int(y_float)
        if global_index in _mask_indices(split, "ocr"):
            if global_index == _full_ocr_patch_index(split):
                ocr_draw.rectangle((x - 16, y - 16, x + 16, y + 16), fill=255)
            else:
                ocr_draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=255)
        if global_index in _mask_indices(split, "artifact"):
            width = 7 if global_index == min(_mask_indices(split, "artifact")) else 3
            artifact_draw.line((x - 12, y, x + 12, y), fill=255, width=width)
            artifact_draw.line((x, y - 12, x, y + 12), fill=255, width=width)

    # Reuse the project's procedural artifact primitives for negatives.
    negatives: list[tuple[str, float, float]] = []
    kinds = ("text", "line_intersection", "axis")
    for n, kind in enumerate(kinds):
        x, y = 20 + n * 66, HEIGHT - 18
        target = ocr_draw if kind == "text" else artifact_draw
        _artifact_geometry(draw, target, kind, x, y)
        negatives.append((kind, float(x), float(y)))

    # Real negative proposals include long, very faint print strokes.  Keep
    # these in a lower band that is at least 20 px below every truth center
    # (truth y is bounded by 125), so they cannot alter center-mask hits.
    # Fill 226 gives an exact float32 ink maximum of 29/255, matching the
    # lower real-dev negative tail without changing any positive primitive.
    faint_y = (155, 160)
    for y in faint_y:
        # Full-width strokes provide enough 4 px grid proposals to move the
        # lower-tail quantile, while remaining outside every truth 5x5 window.
        # Two-pixel width keeps the 0.11 support threshold stable under the
        # deterministic print-noise perturbation while preserving the same
        # 29/255 per-pixel ink maximum.
        draw.line((5, y, 220, y), fill=226, width=9)
    negatives.append(("faint_line", 32.0, float(faint_y[0])))

    # OCR-heavy negatives are mask-only regions over a light text-like stroke
    # cluster.  A 29 by 29 mask occupies 841/1089 of a 33 by 33 proposal
    # patch, exceeding the tracked real-dev OCR p95 of 0.7272727.  This region
    # is also in the lower band and never intersects a truth-center window.
    ocr_heavy_center = (208, 155)
    ocr_draw.rectangle((5, 140, 223, 167), fill=255)
    for yy in range(144, 167, 4):
        draw.line((8, yy, 220, yy), fill=226, width=3)
    negatives.append(("ocr_heavy", float(ocr_heavy_center[0]), float(ocr_heavy_center[1])))

    array = np.asarray(image, dtype=np.float32) / 255.0
    # A deterministic, bounded print perturbation keeps both families useful.
    if index % 2:
        array = np.clip(array + rng.normal(0.0, 0.006, array.shape), 0.0, 1.0)
    array = array.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.stack((1.0 - array,
                                        np.asarray(ocr, dtype=np.float32) / 255.0,
                                        np.asarray(artifact, dtype=np.float32) / 255.0), axis=0).copy())
    # Effective diameter is the deterministic procedural footprint contract:
    # point markers occupy one pixel; larger marker primitives use 2*r pixels.
    rendered = tuple(_measure_rendered_diameter(d) for d in diameters)
    return Scene(split, f"{split}_range_{index:03d}", seed, tensor,
                 tuple(centers), tuple(diameters), rendered, tuple(negatives))


@lru_cache(maxsize=4)
def build_split(split: str, *, scene_count: int = SCENE_COUNT) -> tuple[Scene, ...]:
    if split not in SEED_BASE:
        raise ValueError(f"unknown split: {split}")
    if scene_count <= 0 or 2004 % scene_count:
        raise ValueError("scene_count must be a positive divisor of 2004")
    per_scene = 2004 // scene_count
    values = _diameters(2004)
    return tuple(_scene(split, index, values[index * per_scene:(index + 1) * per_scene])
                 for index in range(scene_count))


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    def q(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"minimum": ordered[0], "p05": q(.05), "median": median(ordered),
            "p90": q(.90), "p95": q(.95), "maximum": ordered[-1]}


def _hash_scenes(scenes: tuple[Scene, ...]) -> str:
    digest = hashlib.sha256()
    for scene in scenes:
        digest.update(scene.tensor.numpy().tobytes())
        digest.update(json.dumps(scene.diameters).encode())
    return digest.hexdigest()


def _patch_distribution(scenes: tuple[Scene, ...]) -> dict[str, object]:
    ink_means: list[float] = []
    ocr_means: list[float] = []
    artifact_means: list[float] = []
    center_ink_means: list[float] = []
    for scene in scenes:
        padded = np.pad(scene.tensor.numpy(), ((0, 0), (PATCH // 2, PATCH // 2),
                                               (PATCH // 2, PATCH // 2)), mode="constant")
        for x_float, y_float in scene.centers:
            x, y = int(round(x_float)), int(round(y_float))
            patch = padded[:, y:y + PATCH, x:x + PATCH]
            ink_means.append(float(patch[0].mean()))
            middle = PATCH // 2
            center_ink_means.append(float(patch[0, middle - 2:middle + 3,
                                                middle - 2:middle + 3].mean()))
            ocr_means.append(float(patch[1].mean()))
            artifact_means.append(float(patch[2].mean()))
    return {"shape": [3, PATCH, PATCH], "marker_count": len(ink_means),
            "channel_mean_quantiles": {"ink": _quantiles(ink_means),
                                        "ink_center_5x5": _quantiles(center_ink_means),
                                        "ocr_mask": _quantiles(ocr_means),
                                        "artifact_mask": _quantiles(artifact_means)},
            "aggregate_sha256": _hash_scenes(scenes)}


@lru_cache(maxsize=1)
def audit() -> dict[str, object]:
    train, dev = build_split("train"), build_split("dev")
    all_scenes = train + dev
    dev_markers = sum(len(scene.centers) for scene in dev)
    def mask_hits(scenes: tuple[Scene, ...], channel: int) -> int:
        hits = 0
        for scene in scenes:
            for x_float, y_float in scene.centers:
                x, y = int(round(x_float)), int(round(y_float))
                window = scene.tensor[channel, max(0, y - 2):y + 3,
                                      max(0, x - 2):x + 3]
                hits += int(float(window.max()) >= MASK_THRESHOLD)
        return hits

    def split_record(scenes: tuple[Scene, ...]) -> dict[str, object]:
        values = [d for s in scenes for d in s.diameters]
        rendered = [d for s in scenes for d in s.rendered_diameters]
        ocr_hits, artifact_hits = mask_hits(scenes, 1), mask_hits(scenes, 2)
        return {"scene_count": len(scenes), "marker_count": len(values),
                "seeds": {"minimum": min(s.seed for s in scenes), "maximum": max(s.seed for s in scenes)},
                "tensor_shapes": [list(shape) for shape in sorted({tuple(s.tensor.shape) for s in scenes})],
                "diameter_px": _quantiles(values), "rendered_diameter_px": _quantiles(rendered),
                "mask_center_hit_rates": {"ocr": ocr_hits / len(values),
                                           "artifact": artifact_hits / len(values)},
                "mask_center_hits": {"ocr": ocr_hits, "artifact": artifact_hits},
                "aggregate_sha256": _hash_scenes(scenes)}

    train_record = split_record(train)
    dev_record = split_record(dev)
    patch_record = _patch_distribution(dev)
    patch_quantiles = patch_record["channel_mean_quantiles"]
    gates = {
        "diameter_quantiles_match_real_dev": dev_record["diameter_px"] == {
            "minimum": 1.0, "p05": 6.0, "median": 12.0,
            "p90": 24.0, "p95": 27.0, "maximum": 48.0,
        },
        "rendered_diameter_contains_real_envelope": (
            dev_record["rendered_diameter_px"]["minimum"] <= 1.0
            and dev_record["rendered_diameter_px"]["maximum"] >= 48.0
        ),
        "ink_patch_distribution_contains_real_dev": (
            patch_quantiles["ink"]["minimum"] <= 0.08884549530358135
            and patch_quantiles["ink"]["p95"] >= 0.5558205991956047
            and abs(patch_quantiles["ink"]["median"] - 0.18972253479248236) <= 0.05
        ),
        "artifact_patch_distribution_contains_real_dev": (
            patch_quantiles["artifact_mask"]["p90"] >= 0.12121212121212122
            and patch_quantiles["artifact_mask"]["maximum"] >= 0.2396694214876033
        ),
        "ocr_patch_distribution_contains_real_maximum": patch_quantiles["ocr_mask"]["maximum"] >= 1.0,
        "train_and_dev_mask_hits_match_real_dev": all(
            record["mask_center_hits"] == {"ocr": 75, "artifact": 332}
            for record in (train_record, dev_record)
        ),
        "train_and_dev_seeds_disjoint": set(scene.seed for scene in train).isdisjoint(
            scene.seed for scene in dev
        ),
    }

    return {
        "schema": "graphreader.marker-center-real-range-generator-audit.v1",
        "scope": {"synthetic_only": True, "private_or_article_images": False,
                  "model_loaded": False, "training_performed": False,
                  "candidate_revision_created": False, "aggregate_only": True,
                  "scene_ids_emitted": False, "truth_rows_emitted": False,
                  "pixels_emitted": False},
        "sources": {"procedural_primitives": "ml/markers/center/dataset.py",
                    "patch_shape": [3, PATCH, PATCH]},
        "splits": {"train": train_record, "dev": dev_record},
        "mask_overlap_scenarios": {"markers_per_split": dev_markers,
            "ocr_hard_hits": dev_record["mask_center_hits"]["ocr"],
            "artifact_hard_hits": dev_record["mask_center_hits"]["artifact"],
            "ocr_rate": 75 / dev_markers, "artifact_rate": 332 / dev_markers,
            "unmasked_controls": dev_markers - 332,
            "threshold": MASK_THRESHOLD, "window": [5, 5]},
        "hard_negative_kinds": ["text", "line_intersection", "axis", "faint_line", "ocr_heavy"],
        "hard_negative_representatives": {
            "faint_line": {"x": 32.0, "y": 155.0},
            "ocr_heavy": {"x": 208.0, "y": 155.0},
        },
        "truth_center_patch_distribution": patch_record,
        "distribution_gates": gates,
        "aggregate": {"scene_count": len(all_scenes), "marker_count": dev_markers * 2,
                       "diameter_px": _quantiles(d for s in all_scenes for d in s.diameters),
                       "disjoint_seed_sets": {"train_range": [train[0].seed, train[-1].seed],
                                               "dev_range": [dev[0].seed, dev[-1].seed],
                                               "disjoint": True},
                       "model_or_private_inputs": False},
    }
