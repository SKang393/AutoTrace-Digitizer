# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh truth-hidden eight-role public scenes for selected-confidence P2."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes
from ml.ocr.production_csharp_marker_gate_v3.dataset import (
    CompositeScene,
    build_scene as build_v3_scene,
)
from .protocol import (
    DEGRADATION_FAMILIES,
    RENDERER_FAMILIES,
    REQUIRED_ROLES,
    REVISION,
    SCENE_COUNT,
    SCENE_HEIGHT,
    SCENE_WIDTH,
)


def _derived_secret(secret_seed: int, index: int) -> int:
    material = f"{REVISION}:{secret_seed}:{index}:sealed-public-v9-p2".encode()
    return int.from_bytes(sha256(material).digest()[:16], "little")


def _add_structure_confounds(pixels: np.ndarray, index: int) -> np.ndarray:
    image = Image.fromarray(pixels, mode="L")
    draw = ImageDraw.Draw(image)
    ink = 12 + (index % 25)
    shift = index % 9
    draw.arc((142 + shift, 198, 164 + shift, 220), 35, 320, fill=ink, width=3)
    draw.line((264 - shift, 208, 278 - shift, 194, 292 - shift, 208), fill=ink, width=3)
    draw.line((356 + shift, 192, 356 + shift, 224), fill=ink, width=2)
    draw.line((416 - shift, 205, 432 - shift, 188), fill=ink, width=3)
    draw.line((416 - shift, 205, 435 - shift, 210), fill=ink, width=3)
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
    if family == "diagonal-fiber-drift-v9-p2-public":
        value += 0.23 * np.sin((8.0 * xx) + (5.0 * yy))
    elif family == "low-amplitude-ring-fade-v9-p2-public":
        value += 0.29 * np.sqrt((xx * xx) + (yy * yy))
    elif family == "offset-row-shelf-v9-p2-public":
        value += np.where(yy < 0.13, -0.19, 0.22)
    elif family == "subpixel-paper-wave-v9-p2-public":
        value += 0.21 * np.cos((11.0 * yy) - (4.0 * xx))
    else:
        raise ValueError(f"Unknown V9 P2 public degradation family: {family}")
    value += rng.normal(0.0, 0.08, size=value.shape)
    return np.rint(np.clip(value, 0, 255)).astype(np.uint8)


def build_scene(secret_seed: int, index: int) -> CompositeScene:
    if index < 0 or index >= SCENE_COUNT:
        raise ValueError("V9 P2 public scene index is out of range")
    derived = _derived_secret(secret_seed, index)
    base = build_v3_scene(derived, index % 48)
    family = DEGRADATION_FAMILIES[index % len(DEGRADATION_FAMILIES)]
    raster = _add_structure_confounds(base.raster, index)
    raster = _apply_degradation(raster, family, np.random.default_rng(derived ^ 0x71C4A9))
    roles = tuple(item.role for item in base.text_truths)
    if set(roles) != set(REQUIRED_ROLES) or len(roles) != len(REQUIRED_ROLES):
        raise RuntimeError("The fresh public scene does not contain every role exactly once")
    return replace(
        base,
        scene_id=f"ocr-selected-confidence-v9-p2-public-{index:05d}",
        renderer_family=RENDERER_FAMILIES[index % len(RENDERER_FAMILIES)],
        degradation_family=family,
        raster=raster,
    )


def build_public_split(secret_seed: int) -> tuple[CompositeScene, ...]:
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


def save_public_archive(scenes: tuple[CompositeScene, ...], path: Path) -> dict[str, object]:
    if len(scenes) != SCENE_COUNT:
        raise ValueError("V9 P2 public scene count changed")
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
        "schema": "graphreader.ocr-selected-confidence-public-fixtures.v9-p2",
        "revision": REVISION,
        "split": "sealed_public",
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.text_truths) for scene in scenes),
        "required_roles": list(REQUIRED_ROLES),
        "cases": cases,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "p1_or_p2_selection_fixture_bytes_reused": False,
        "predecessor_truth_or_scene_ids_reused": False,
        "secret_seed_serialized": False,
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


__all__ = ["CompositeScene", "build_public_split", "build_scene", "save_public_archive"]
