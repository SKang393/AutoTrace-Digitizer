# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh selection and public fixtures for P3 cross-model consensus."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes
from ml.ocr.production_csharp_marker_gate_v3.dataset import (
    CompositeScene,
    build_scene as build_base_scene,
)
from .protocol import (
    PUBLIC_DEGRADATION_FAMILIES,
    PUBLIC_RENDERER_FAMILIES,
    PUBLIC_SCENE_COUNT,
    REQUIRED_ROLES,
    REVISION,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    SELECTION_DEGRADATION_FAMILIES,
    SELECTION_RENDERER_FAMILIES,
    SELECTION_SCENE_COUNT,
)


Split = Literal["selection", "sealed_public"]


def _count(split: Split) -> int:
    return SELECTION_SCENE_COUNT if split == "selection" else PUBLIC_SCENE_COUNT


def _families(split: Split) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if split == "selection":
        return SELECTION_RENDERER_FAMILIES, SELECTION_DEGRADATION_FAMILIES
    return PUBLIC_RENDERER_FAMILIES, PUBLIC_DEGRADATION_FAMILIES


def _derived_secret(split: Split, secret_seed: int, index: int) -> int:
    material = f"{REVISION}:{split}:{secret_seed}:{index}:fresh-v9-p3".encode()
    return int.from_bytes(sha256(material).digest()[:16], "little")


def _add_structure_confounds(pixels: np.ndarray, split: Split, index: int) -> np.ndarray:
    image = Image.fromarray(pixels, mode="L")
    draw = ImageDraw.Draw(image)
    ink = 16 + ((index * 7) % 31)
    offset = index % 13
    if split == "selection":
        draw.line((132 + offset, 188, 151 + offset, 207, 169 + offset, 188), fill=ink, width=2)
        draw.rectangle((241 - offset, 195, 258 - offset, 212), outline=ink, width=2)
        draw.line((383 + offset, 187, 383 + offset, 219), fill=ink, width=2)
        draw.arc((444 - offset, 193, 469 - offset, 218), 10, 340, fill=ink, width=2)
    else:
        draw.arc((125 + offset, 190, 151 + offset, 216), 28, 326, fill=ink, width=3)
        draw.line((232 - offset, 211, 247 - offset, 190, 263 - offset, 211), fill=ink, width=2)
        draw.line((369 + offset, 188, 369 + offset, 221), fill=ink, width=3)
        draw.line((438 - offset, 207, 454 - offset, 190, 471 - offset, 207), fill=ink, width=2)
    return np.asarray(image, dtype=np.uint8)


def _degrade(pixels: np.ndarray, family: str, rng: np.random.Generator) -> np.ndarray:
    value = pixels.astype(np.float32)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, SCENE_HEIGHT, dtype=np.float32),
        np.linspace(-1.0, 1.0, SCENE_WIDTH, dtype=np.float32),
        indexing="ij",
    )
    if family == "fine-column-lilt-v9-p3-selection":
        value += 0.20 * np.sin((13.0 * xx) + (2.0 * yy))
    elif family == "shallow-corner-fade-v9-p3-selection":
        value += 0.22 * ((xx + 0.63) ** 2 + (yy - 0.41) ** 2)
    elif family == "low-rank-paper-slope-v9-p3-selection":
        value += (0.18 * xx) - (0.12 * yy)
    elif family == "alternating-row-haze-v9-p3-selection":
        value += 0.17 * np.cos(9.0 * yy)
    elif family == "oblique-paper-ripple-v9-p3-public":
        value += 0.24 * np.sin((7.0 * xx) - (9.0 * yy))
    elif family == "off-center-radial-wash-v9-p3-public":
        value += 0.27 * np.sqrt(((xx - 0.31) ** 2) + ((yy + 0.22) ** 2))
    elif family == "paired-column-drift-v9-p3-public":
        value += 0.19 * np.cos(15.0 * xx) + 0.11 * np.sin(3.0 * yy)
    elif family == "asymmetric-row-shelf-v9-p3-public":
        value += np.where(yy < -0.17, -0.18, 0.21)
    else:
        raise ValueError(f"Unknown P3 degradation family: {family}")
    value += rng.normal(0.0, 0.075, size=value.shape)
    return np.rint(np.clip(value, 0, 255)).astype(np.uint8)


def build_scene(split: Split, secret_seed: int, index: int) -> CompositeScene:
    if index < 0 or index >= _count(split):
        raise ValueError("P3 scene index is out of range")
    derived = _derived_secret(split, secret_seed, index)
    base = build_base_scene(derived, index % 48)
    renderers, degradations = _families(split)
    family = degradations[index % len(degradations)]
    raster = _add_structure_confounds(base.raster, split, index)
    raster = _degrade(raster, family, np.random.default_rng(derived ^ 0x39C7D2))
    roles = tuple(item.role for item in base.text_truths)
    if set(roles) != set(REQUIRED_ROLES) or len(roles) != len(REQUIRED_ROLES):
        raise RuntimeError("P3 fixture does not contain every role exactly once")
    labels = {item.display_text for item in base.text_truths}
    if "Chandler" in labels or "Generalization" in labels:
        raise RuntimeError("P3 fixture contains a prohibited private-example label")
    return replace(
        base,
        scene_id=f"ocr-cross-model-consensus-v9-p3-{split}-{index:05d}",
        renderer_family=renderers[index % len(renderers)],
        degradation_family=family,
        raster=raster,
    )


def build_split(split: Split, secret_seed: int) -> tuple[CompositeScene, ...]:
    return tuple(build_scene(split, secret_seed, index) for index in range(_count(split)))


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)


def save_archive(scenes: tuple[CompositeScene, ...], split: Split, path: Path) -> dict[str, object]:
    if len(scenes) != _count(split):
        raise ValueError("P3 split scene count changed")
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
        "schema": "graphreader.ocr-cross-model-consensus-fixtures.v9-p3",
        "revision": REVISION,
        "split": split,
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.text_truths) for scene in scenes),
        "required_roles": list(REQUIRED_ROLES),
        "cases": cases,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_fixture_bytes_truth_or_scene_ids_reused": False,
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


__all__ = ["CompositeScene", "Split", "build_scene", "build_split", "save_archive"]
