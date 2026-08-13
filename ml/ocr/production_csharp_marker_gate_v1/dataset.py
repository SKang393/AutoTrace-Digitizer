# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""New procedural composite scenes for direct C# OCR and marker execution."""

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
    "double_rebound_public",
    "descending_echo_public",
    "interleaved_plateau_public",
    "late_surge_public",
    "wide_cycle_public",
)
NUMERIC_LABELS = (
    ("0", "0"), ("5", "5"), ("10", "10"), ("20", "20"),
    ("40", "40"), ("60", "60"), ("75", "75"), ("90", "90"),
    ("100", "100"), ("-2", "-2"), ("2.5", "2.5"), ("33%", "33%"),
)
PHASE_WORDS = ("Baseline", "Treatment", "Maintenance", "Followup", "Intervention")
ANNOTATION_WORDS = ("O o l I", "A B C", "Low High", "Probe Set", "Data Check", "Phase Note")
LEGEND_WORDS = ("Target", "Series", "Rate", "Data", "Level", "Plan", "Probe")
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
    material = f"{REVISION}:{secret_seed}:{index}".encode()
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
    if family == "asymmetric-luma-ramp-public":
        value += 4.2 * xx - 2.7 * yy + 1.1 * xx * yy
    elif family == "cross-axis-scan-public":
        value += 1.8 * np.sin((2.1 * xx + 0.35 * yy) * np.pi)
        value += 1.3 * np.cos((0.4 * xx - 2.5 * yy) * np.pi)
    elif family == "gamma-quantized-paper-public":
        gamma = float(rng.uniform(0.965, 1.035))
        value = 255.0 * np.power(np.clip(value / 255.0, 0.0, 1.0), gamma)
        value = np.rint(value).astype(np.int32) // 2 * 2
    elif family == "corner-falloff-public":
        distance = np.sqrt(np.square(xx + 0.62) + np.square(yy - 0.48))
        value -= np.clip(distance * 2.8, 0.0, 5.0)
    else:
        raise ValueError(f"Unsupported degradation family {family!r}")
    noise = rng.normal(0.0, 0.32, size=value.shape)
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
        variant=101 + index,
        seed=seed ^ 0x5A17D3,
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
    ink = int(rng.integers(10, 48))
    y_display, y_truth = NUMERIC_LABELS[(index * 7 + 5) % len(NUMERIC_LABELS)]
    x_display, x_truth = NUMERIC_LABELS[(index * 11 + 3) % len(NUMERIC_LABELS)]
    phase = PHASE_WORDS[(index * 2 + 3) % len(PHASE_WORDS)]
    annotation = ANNOTATION_WORDS[(index * 5 + 1) % len(ANNOTATION_WORDS)]
    legend = LEGEND_WORDS[(index * 3 + 2) % len(LEGEND_WORDS)]
    annotation_y = 98 + (index * 37) % 53
    labels = (
        (y_display, y_truth, "y_tick", "numeric", (13, 71 + (index * 47) % 99)),
        (x_display, x_truth, "x_tick", "numeric", (169 + (index * 71) % 222, 282)),
        (phase, phase, "phase_heading", "word", (137 + (index * 59) % 184, 7)),
        (
            annotation,
            annotation,
            "annotation",
            "ambiguity" if annotation == "O o l I" else "word",
            (398 + (index * 29) % 60, annotation_y),
        ),
        (legend, legend, "legend_text", "word", (576, 174 + (index * 9) % 12)),
    )
    truths: list[TextTruth] = []
    for label_index, (display, truth, role, family, position) in enumerate(labels):
        size = int(rng.integers(16, 21)) if role in {"x_tick", "y_tick"} else int(rng.integers(19, 23))
        mask, bounds = _text_mask(image.size, display, position, _font(index + label_index, size))
        image.paste(ink, mask=mask)
        truths.append(TextTruth(display, truth, role, family, bounds))

    degradation = DEGRADATION_FAMILIES[index % len(DEGRADATION_FAMILIES)]
    raster = _apply_degradation(np.asarray(image, dtype=np.uint8), degradation, rng)
    marker_centers = tuple(
        (x + graph_left, y + graph_top)
        for x, y in source.centers
    )
    prohibited = tuple(
        ProhibitedPoint(item.kind, item.x + graph_left, item.y + graph_top)
        for item in source.prohibited
    )
    renderer = RENDERER_FAMILIES[index % len(RENDERER_FAMILIES)]
    return CompositeScene(
        f"ocr-marker-csharp-public-{index:05d}",
        renderer,
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
    "CompositeScene",
    "build_scene",
    "build_sealed_split",
    "read_manifest",
    "save_sealed_archive",
]
