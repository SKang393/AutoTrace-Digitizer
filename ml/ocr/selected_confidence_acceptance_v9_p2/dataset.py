# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh visible shape-confound selection scenes for OCR V9 P2."""

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
    material = f"{REVISION}:{secret_seed}:{index}:visible-selection-v9-p2".encode()
    return int.from_bytes(sha256(material).digest()[:16], "little")


def _add_shape_confounds(pixels: np.ndarray, index: int) -> np.ndarray:
    image = Image.fromarray(pixels, mode="L")
    draw = ImageDraw.Draw(image)
    ink = 13 + (index % 23)
    shift = index % 11
    y = 202 + (index % 4) * 9
    draw.arc((158 + shift, y - 10, 178 + shift, y + 10), 25, 315, fill=ink, width=3)
    draw.line((264 - shift, y - 12, 276 - shift, y + 3, 288 - shift, y - 12), fill=ink, width=3)
    draw.rectangle((372 + shift, y - 7, 390 + shift, y + 5), outline=ink, width=3)
    draw.line((453 - shift, y - 11, 470 - shift, y + 6), fill=ink, width=3)
    draw.line((453 - shift, y + 6, 470 - shift, y - 11), fill=ink, width=3)
    return np.asarray(image, dtype=np.uint8)


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
    if family == "oblique-paper-ripple-v9-p2-selection":
        value += 0.43 * np.sin((7.0 * xx) + (5.0 * yy)) + 0.18 * xx
    elif family == "piecewise-contrast-shelf-v9-p2-selection":
        value = np.where(yy < -0.16, value * 0.998, value * 1.002) + 0.37 * xx
    elif family == "radial-fade-quantization-v9-p2-selection":
        value += 0.56 * np.sqrt((xx * xx) + (yy * yy))
        value = np.rint(value / 2.0) * 2.0
    elif family == "row-drift-speckle-v9-p2-selection":
        value += 0.39 * np.cos(13.0 * yy) - 0.24 * np.sin(8.0 * xx)
    else:
        raise ValueError(f"Unknown V9 P2 degradation family: {family}")
    value += rng.normal(0.0, 0.11, size=value.shape)
    return np.rint(np.clip(value, 0, 255)).astype(np.uint8)


def build_scene(secret_seed: int, index: int) -> CompositeScene:
    if index < 0 or index >= SCENE_COUNT:
        raise ValueError("V9 P2 selection scene index is out of range")
    derived = _derived_secret(secret_seed, index)
    base = build_v4_scene(derived, index % 64)
    pixels = _add_shape_confounds(base.raster, index)
    family = DEGRADATION_FAMILIES[index % len(DEGRADATION_FAMILIES)]
    pixels = _apply_degradation(pixels, family, np.random.default_rng(derived ^ 0x2A71D5))
    return replace(
        base,
        scene_id=f"ocr-selected-confidence-v9-p2-selection-{index:05d}",
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
        raise ValueError("V9 P2 selection scene count changed")
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
            "structure_collision_count": 4,
        })
    manifest = {
        "schema": "graphreader.ocr-selected-confidence-selection-fixtures.v9-p2",
        "revision": REVISION,
        "split": "visible_selection",
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.text_truths) for scene in scenes),
        "cases": cases,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "p1_fixture_bytes_reused": False,
        "p1_truth_or_scene_ids_reused": False,
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
