# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh visible structure-collision selection scenes for OCR V9 P1."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes
from ml.ocr.production_csharp_marker_gate_v4.dataset import CompositeScene, build_scene as build_v4_scene
from .protocol import (
    DEGRADATION_FAMILIES,
    RENDERER_FAMILIES,
    REVISION,
    SCENE_COUNT,
    SCENE_HEIGHT,
    SCENE_WIDTH,
)


def _derived_secret(secret_seed: int, index: int) -> int:
    material = f"{REVISION}:{secret_seed}:{index}:visible-selection-v9".encode()
    return int.from_bytes(sha256(material).digest()[:16], "little")


def _add_structure_collisions(pixels: np.ndarray, index: int) -> np.ndarray:
    image = Image.fromarray(pixels, mode="L")
    draw = ImageDraw.Draw(image)
    ink = 16 + (index % 19)
    shift = index % 9
    y = 211 + (index % 3) * 7
    draw.line((176 + shift, y, 190 + shift, y - 11, 204 + shift, y), fill=ink, width=3)
    draw.ellipse((306 - shift, y - 10, 320 - shift, y + 4), outline=ink, width=3)
    draw.line((423 + shift, y - 12, 423 + shift, y + 5, 448 + shift, y + 5), fill=ink, width=3)
    return np.asarray(image, dtype=np.uint8)


def _apply_selection_degradation(
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
    if family == "cross-axis-sinusoid-v9-selection":
        value += 0.54 * np.sin((1.3 * xx + 0.7 * yy) * np.pi)
    elif family == "local-shelf-quantization-v9-selection":
        value += np.where((yy > -0.1) & (yy < 0.42), 0.83, -0.21)
        value = np.rint(value / 2.0) * 2.0
    elif family == "asymmetric-paper-fade-v9-selection":
        value += (0.46 * xx) - (0.62 * yy) + 0.29 * xx * yy
    elif family == "row-column-ripple-v9-selection":
        value += 0.37 * np.cos(11.0 * yy) + 0.31 * np.sin(9.0 * xx)
    else:
        raise ValueError(f"Unknown V9 degradation family: {family}")
    value += rng.normal(0.0, 0.12, size=value.shape)
    return np.rint(np.clip(value, 0, 255)).astype(np.uint8)


def build_scene(secret_seed: int, index: int) -> CompositeScene:
    if index < 0 or index >= SCENE_COUNT:
        raise ValueError("V9 selection scene index is out of range")
    derived = _derived_secret(secret_seed, index)
    base = build_v4_scene(derived, index % 64)
    pixels = _add_structure_collisions(base.raster, index)
    family = DEGRADATION_FAMILIES[index % len(DEGRADATION_FAMILIES)]
    pixels = _apply_selection_degradation(pixels, family, np.random.default_rng(derived ^ 0x91A7C3))
    return replace(
        base,
        scene_id=f"ocr-recognizer-confirmed-v9-selection-{index:05d}",
        renderer_family=RENDERER_FAMILIES[index % len(RENDERER_FAMILIES)],
        degradation_family=family,
        raster=pixels,
    )


def build_selection_split(secret_seed: int) -> tuple[CompositeScene, ...]:
    return tuple(build_scene(secret_seed, index) for index in range(SCENE_COUNT))


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)


def save_selection_archive(scenes: tuple[CompositeScene, ...], path: Path) -> dict[str, object]:
    if len(scenes) != SCENE_COUNT:
        raise ValueError("V9 selection scene count changed")
    resources: list[tuple[str, bytes]] = []
    cases: list[dict[str, object]] = []
    for scene in scenes:
        image_path = f"images/{scene.scene_id}.png"
        image_bytes = _png_bytes(scene.raster)
        resources.append((image_path, image_bytes))
        cases.append({
            "scene_id": scene.scene_id,
            "image_path": image_path,
            "image_sha256": sha256_bytes(image_bytes),
            "raster_sha256": sha256_bytes(scene.raster.tobytes(order="C")),
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
            "structure_collision_count": 3,
        })
    manifest = {
        "schema": "graphreader.ocr-recognizer-confirmed-selection-fixtures.v9",
        "revision": REVISION,
        "split": "visible_selection",
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.text_truths) for scene in scenes),
        "cases": cases,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_fixture_bytes_reused": False,
        "predecessor_truth_or_scene_ids_reused": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "x") as archive:
        _write(archive, "manifest.json", manifest_bytes)
        for name, payload in sorted(resources):
            _write(archive, name, payload)
    return {
        "fixture_archive_sha256": sha256_bytes(path.read_bytes()),
        "fixture_manifest_sha256": sha256_bytes(manifest_bytes),
        "scene_count": len(scenes),
        "truth_region_count": manifest["truth_region_count"],
        "resource_count": len(resources),
    }
