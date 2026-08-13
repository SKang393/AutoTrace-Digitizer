# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Disjoint eight-role procedural scenes for the V11 C# composition gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml.markers.center.runtime_consistency_v2.dataset import build_scene as build_marker_scene
from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.production_composition_v1.dataset import _text_mask
from .protocol import (
    DEGRADATION_FAMILIES,
    PLOT_BOUNDS,
    RENDERER_FAMILIES,
    REVISION,
    SCENE_COUNT,
    SCENE_HEIGHT,
    SCENE_WIDTH,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FONT_PATHS = (
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)
SOURCE_GRAPH_FAMILIES = (
    "alternating_rise_v3_composed",
    "stepped_fall_v3_composed",
    "paired_reversal_v3_composed",
    "sparse_probe_v3_composed",
    "dense_cycle_v3_composed",
    "offset_plateau_v3_composed",
)
Y_LABELS = (
    ("5", "5"), ("15", "15"), ("35", "35"), ("55", "55"),
    ("75", "75"), ("90", "90"), ("105", "105"), ("-10", "-10"),
    ("2.0", "2.0"), ("50%", "50%"),
)
X_LABELS = (
    ("2", "2"), ("5", "5"), ("9", "9"), ("14", "14"),
    ("18", "18"), ("22", "22"), ("26", "26"), ("32", "32"),
)
AXIS_TITLES = ("Session count", "Trial number", "Visit index", "Observation")
PHASE_WORDS = ("Acquisition", "Practice", "Review", "Sustainment", "Instruction")
LEGEND_WORDS = ("Correct", "Latency", "Attempts", "Percent", "Frequency", "Duration")
PARTICIPANT_WORDS = ("Subject A", "Case B", "Pupil C", "Learner D", "Student E")
ANNOTATION_WORDS = ("Target met", "Review point", "Level change", "Rule check", "Probe result")
OTHER_WORDS = ("Weekly", "Daily", "Outcome", "Summary", "Measure")
AMBIGUITY_WORDS = ("l I O o", "I l o O", "o O I l")
GRAPH_OFFSET = (104, 48)


@dataclass(frozen=True)
class TextTruth:
    display_text: str
    truth_text: str
    role: str
    family: str
    box: Box


@dataclass(frozen=True)
class ProhibitedPoint:
    kind: str
    x: float
    y: float


@dataclass(frozen=True)
class CompositeScene:
    scene_id: str
    renderer_family: str
    degradation_family: str
    raster: np.ndarray
    artifact_mask: np.ndarray
    text_truths: tuple[TextTruth, ...]
    marker_centers: tuple[tuple[float, float], ...]
    prohibited: tuple[ProhibitedPoint, ...]


def _derived_seed(secret_seed: int, index: int) -> int:
    material = f"{REVISION}:{secret_seed}:{index}:sealed-public-v3".encode()
    return int.from_bytes(sha256(material).digest()[:8], "little")


def _font(index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(REPO_ROOT / FONT_PATHS[index % len(FONT_PATHS)]), size=size)


def _apply_degradation(
    pixels: np.ndarray,
    family: str,
    rng: np.random.Generator,
) -> np.ndarray:
    value = pixels.astype(np.float32)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, SCENE_HEIGHT, dtype=np.float32),
        np.linspace(-1.0, 1.0, SCENE_WIDTH, dtype=np.float32),
        indexing="ij",
    )
    if family == "crossfield-gamma-ramp-v3-public":
        gamma = float(rng.uniform(0.97, 1.035))
        value = 255.0 * np.power(np.clip(value / 255.0, 0.0, 1.0), gamma)
        value += 0.9 * np.sin((1.35 * xx + 0.75 * yy) * np.pi)
    elif family == "elliptic-paper-shelf-v3-public":
        radius = np.square((xx + 0.24) / 1.18) + np.square((yy - 0.16) / 0.86)
        value += np.where(radius < 0.62, 1.25, -0.85)
    elif family == "staggered-row-quantization-v3-public":
        value += ((np.floor((yy + 1.0) * 7.0) % 2.0) * 1.1) - 0.55
        value = np.rint(value / 2.0) * 2.0
    elif family == "counterphase-corner-fade-v3-public":
        value -= np.clip((np.abs(xx - 0.31) + np.abs(yy + 0.22)) * 1.15, 0.0, 2.8)
        value += 0.55 * np.cos((1.8 * xx - 1.1 * yy) * np.pi)
    else:
        raise ValueError(f"Unsupported degradation family {family!r}")
    noise = rng.normal(0.0, 0.23, size=value.shape)
    return np.rint(np.clip(value + noise, 0, 255)).astype(np.uint8)


def build_scene(secret_seed: int, index: int) -> CompositeScene:
    if index < 0 or index >= SCENE_COUNT:
        raise ValueError("Composite scene index leaves the frozen public split")
    seed = _derived_seed(secret_seed, index)
    rng = np.random.default_rng(seed)
    source_family = SOURCE_GRAPH_FAMILIES[index % len(SOURCE_GRAPH_FAMILIES)]
    source = build_marker_scene(
        split="sealed_public",
        family=source_family,
        degradation="none",
        variant=431 + index,
        seed=seed ^ 0xD341A5,
    )

    background = int(rng.integers(249, 256))
    pixels = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.uint8)
    artifact_mask = np.zeros((SCENE_HEIGHT, SCENE_WIDTH), dtype=np.uint8)
    graph_height, graph_width = source.tensor.shape[1:]
    graph_left, graph_top = GRAPH_OFFSET
    graph = np.rint((1.0 - source.tensor[0].numpy()) * 255.0).clip(0, 255).astype(np.uint8)
    pixels[graph_top:graph_top + graph_height, graph_left:graph_left + graph_width] = graph
    source_artifact = np.maximum(source.tensor[1].numpy(), source.tensor[2].numpy())
    artifact_mask[graph_top:graph_top + graph_height, graph_left:graph_left + graph_width] = (
        source_artifact >= 0.5
    ).astype(np.uint8) * 255

    image = Image.fromarray(pixels, mode="L")
    draw = ImageDraw.Draw(image)
    ink = int(rng.integers(10, 44))
    y_display, y_truth = Y_LABELS[(index * 7 + 3) % len(Y_LABELS)]
    x_display, x_truth = X_LABELS[(index * 5 + 1) % len(X_LABELS)]
    labels = (
        (y_display, y_truth, "y_tick", "numeric", (18, 112 + (index * 23) % 76)),
        (x_display, x_truth, "x_tick", "numeric", (190 + (index * 37) % 210, 278)),
        (AXIS_TITLES[(index * 3 + 1) % len(AXIS_TITLES)], None, "axis_title", "word", (274 + (index * 11) % 45, 299)),
        (PHASE_WORDS[(index * 2 + 1) % len(PHASE_WORDS)], None, "phase_heading", "word", (168 + (index * 41) % 170, 4)),
        (LEGEND_WORDS[(index * 5 + 2) % len(LEGEND_WORDS)], None, "legend_text", "word", (538, 166 + (index * 2) % 8)),
        (PARTICIPANT_WORDS[(index * 3 + 2) % len(PARTICIPANT_WORDS)], None, "participant", "word", (534, 278)),
        (ANNOTATION_WORDS[(index * 4 + 2) % len(ANNOTATION_WORDS)], None, "annotation", "word", (380 + (index * 19) % 20, 105 + (index * 13) % 35)),
        (AMBIGUITY_WORDS[index % len(AMBIGUITY_WORDS)] if index % 3 == 0 else OTHER_WORDS[(index * 3 + 1) % len(OTHER_WORDS)], None, "other", "ambiguity" if index % 3 == 0 else "word", (10, 4)),
    )
    truths: list[TextTruth] = []
    for label_index, (display, truth_override, role, family, position) in enumerate(labels):
        size = 16 + int(rng.integers(0, 3))
        if role in {"x_tick", "y_tick"}:
            size += 1
        font = _font(index + label_index, size)
        mask, bounds = _text_mask(image.size, display, position, font)
        if role == "legend_text":
            clear = (520, 154, SCENE_WIDTH - 1, 202)
        elif role == "annotation":
            clear = (370, max(0, bounds.top - 10), 520, min(SCENE_HEIGHT - 1, bounds.bottom + 11))
        else:
            clear = (
                max(0, bounds.left - 3), max(0, bounds.top - 2),
                min(SCENE_WIDTH - 1, bounds.right + 3), min(SCENE_HEIGHT - 1, bounds.bottom + 2),
            )
        draw.rectangle(clear, fill=background)
        image.paste(ink, mask=mask)
        truths.append(TextTruth(display, truth_override or display, role, family, bounds))

    degradation = DEGRADATION_FAMILIES[index % len(DEGRADATION_FAMILIES)]
    raster = _apply_degradation(np.asarray(image, dtype=np.uint8), degradation, rng)
    marker_centers = tuple((x + graph_left, y + graph_top) for x, y in source.centers)
    prohibited = tuple(
        ProhibitedPoint(item.kind, item.x + graph_left, item.y + graph_top)
        for item in source.prohibited
    )
    return CompositeScene(
        f"ocr-marker-csharp-v3-public-{index:05d}",
        RENDERER_FAMILIES[index % len(RENDERER_FAMILIES)],
        degradation,
        raster,
        artifact_mask,
        tuple(truths),
        marker_centers,
        prohibited,
    )


def build_sealed_split(secret_seed: int) -> tuple[CompositeScene, ...]:
    return tuple(build_scene(secret_seed, index) for index in range(SCENE_COUNT))


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)


def save_sealed_archive(scenes: tuple[CompositeScene, ...], path: Path) -> dict[str, object]:
    if len(scenes) != SCENE_COUNT:
        raise ValueError("The sealed split must contain the preregistered scene count")
    path.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    resources: list[tuple[str, bytes]] = []
    for scene in scenes:
        image_path = f"images/{scene.scene_id}.png"
        artifact_path = f"artifact-masks/{scene.scene_id}.png"
        image_bytes = _png_bytes(scene.raster)
        artifact_bytes = _png_bytes(scene.artifact_mask)
        resources.extend(((image_path, image_bytes), (artifact_path, artifact_bytes)))
        cases.append({
            "scene_id": scene.scene_id,
            "image_path": image_path,
            "image_sha256": sha256_bytes(image_bytes),
            "raster_sha256": sha256_bytes(scene.raster.tobytes(order="C")),
            "artifact_mask_path": artifact_path,
            "artifact_mask_png_sha256": sha256_bytes(artifact_bytes),
            "artifact_mask_raster_sha256": sha256_bytes(scene.artifact_mask.tobytes(order="C")),
            "renderer_family": scene.renderer_family,
            "degradation_family": scene.degradation_family,
            "text_truths": [
                {
                    "display_text": item.display_text,
                    "truth_text": item.truth_text,
                    "role": item.role,
                    "family": item.family,
                    "bbox": [item.box.left, item.box.top, item.box.right, item.box.bottom],
                }
                for item in scene.text_truths
            ],
            "marker_centers": [[x, y] for x, y in scene.marker_centers],
            "prohibited": [
                {"kind": item.kind, "x": item.x, "y": item.y}
                for item in scene.prohibited
            ],
        })
    manifest = {
        "schema": "graphreader.ocr-marker-production-composition-fixtures.v1",
        "revision": REVISION,
        "split": "sealed_public",
        "scene_count": len(scenes),
        "text_truth_count": sum(len(scene.text_truths) for scene in scenes),
        "marker_truth_count": sum(len(scene.marker_centers) for scene in scenes),
        "renderer_families": list(RENDERER_FAMILIES),
        "degradation_families": list(DEGRADATION_FAMILIES),
        "plot_bounds": list(PLOT_BOUNDS),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_fixture_bytes_reused": False,
        "predecessor_truth_or_scene_ids_reused": False,
        "secret_seed_serialized": False,
        "cases": cases,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    with ZipFile(path, "x") as archive:
        _zip_write(archive, "manifest.json", manifest_bytes)
        for name, payload in sorted(resources):
            _zip_write(archive, name, payload)
    return {
        "schema": manifest["schema"],
        "revision": REVISION,
        "split": "sealed_public",
        "scene_count": len(scenes),
        "text_truth_count": manifest["text_truth_count"],
        "marker_truth_count": manifest["marker_truth_count"],
        "fixture_archive_sha256": sha256_file(path),
        "fixture_manifest_sha256": sha256_bytes(manifest_bytes),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "secret_seed_serialized": False,
    }


def read_manifest(path: Path) -> dict[str, object]:
    with ZipFile(path, "r") as archive:
        return json.loads(archive.read("manifest.json"))


__all__ = [
    "CompositeScene", "build_scene", "build_sealed_split", "read_manifest", "save_sealed_archive",
]
