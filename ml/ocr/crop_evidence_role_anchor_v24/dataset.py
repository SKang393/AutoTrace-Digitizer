# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh stored procedural scenes for OCR V24."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from threading import RLock
from typing import Iterator, Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.relational_scene_proposal_role_v21 import dataset as v21
from ml.ocr.role_anchor_set_v23 import dataset as v23
from .protocol import REVISION, ROLE_ORDER, split_registration


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]
RoleTruth = v21.RoleTruth
SceneSample = v21.SceneSample
proposals = v21.proposals
proposal_targets = v21.proposal_targets
encode_scene = v21.encode_scene
_RENDER_LOCK = RLock()


@contextmanager
def _registration_scope() -> Iterator[None]:
    with _RENDER_LOCK:
        original = v23.split_registration
        v23.split_registration = split_registration
        try:
            yield
        finally:
            v23.split_registration = original


def _apply_degradation(raster: np.ndarray, split: Split, index: int) -> np.ndarray:
    """Apply a fresh photometric family without moving any truth geometry."""
    values = raster.astype(np.float32)
    source_ink = raster < 128
    height, width = raster.shape
    if split == "train":
        column = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        gamma = 0.96 + 0.01 * float(index % 9)
        values = 255.0 * np.power(np.clip(values / 255.0, 0.0, 1.0), gamma)
        values += column * float(1 + index % 5)
        level = float(2 + (index * 5) % 6)
        values = np.rint(values / level) * level
    elif split == "validation":
        padded = np.pad(values, ((1, 1), (0, 0)), mode="edge")
        values = (
            padded[:-2] + 6.0 * padded[1:-1] + padded[2:]
        ) / 8.0
        center = width // 2
        half = 41 + index % 37
        values[:, max(0, center - half):min(width, center + half)] += 1.0 + index % 5
    else:
        row = np.linspace(-2.5, 2.5, height, dtype=np.float32)[:, None]
        column = np.linspace(2.0, -2.0, width, dtype=np.float32)[None, :]
        values += row + column
        band = 17 + (index * 37) % max(18, height - 18)
        values[band:band + 1] -= 2.0
    values = np.where(source_ink, np.minimum(values, 120.0), np.maximum(values, 136.0))
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def render_scene(split: Split, index: int) -> SceneSample:
    registered = split_registration(split)
    if index < 0 or index >= registered.scene_count:
        raise IndexError(f"OCR V24 scene index outside frozen {split} split: {index}")
    for attempt in range(48):
        source_index = (index + attempt * 61) % registered.scene_count
        with _registration_scope():
            source = v23.render_scene(split, source_index)
        candidate = SceneSample(
            f"crop-evidence-role-anchor-v24-{split}-{index:05d}",
            split,
            source.renderer_family,
            source.degradation_family,
            _apply_degradation(source.raster, split, index),
            source.plot,
            source.truths,
        )
        try:
            v21.proposal_summary((candidate,))
        except RuntimeError:
            continue
        return candidate
    raise RuntimeError(f"OCR V24 could not render one production proposal per truth: {split}/{index}")


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registered = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registered.scene_count))


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, object]:
    return v21.proposal_summary(scenes)


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    return v21.split_fingerprint(scenes)


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(
        stream, format="PNG", optimize=False, compress_level=9,
    )
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(2026, 8, 16, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def save_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"OCR V24 fixture identity already exists: {path}")
    if not scenes or len({scene.split for scene in scenes}) != 1:
        raise RuntimeError("OCR V24 archive requires one nonempty split")
    cases: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    for scene in scenes:
        payload = _png_bytes(scene.raster)
        source_path = f"images/{scene.scene_id}.png"
        payloads.append((source_path, payload))
        cases.append({
            "scene_id": scene.scene_id,
            "renderer_family": scene.renderer_family,
            "degradation_family": scene.degradation_family,
            "source_path": source_path,
            "source_sha256": sha256(payload).hexdigest(),
            "plot": [scene.plot.left, scene.plot.top, scene.plot.right, scene.plot.bottom],
            "truths": [
                {
                    "box": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom],
                    "role": truth.role,
                    "text": truth.text,
                }
                for truth in scene.truths
            ],
        })
    manifest: dict[str, object] = {
        "schema": "graphreader.ocr-crop-evidence-role-anchor-fixtures.v1",
        "revision": REVISION,
        "split": scenes[0].split,
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "required_roles": list(ROLE_ORDER),
        "cases": cases,
        "synthetic_only": True,
        "chandler_included": False,
        "generalization_label_included": False,
        "private_or_article_images": False,
        "production_approval": False,
        "release_eligible": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        _zip_write(archive, "manifest.json", manifest_bytes)
        for name, payload in payloads:
            _zip_write(archive, name, payload)
    return {
        "archive_path": path.relative_to(REPO_ROOT).as_posix(),
        "archive_sha256": sha256_file(path),
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "split_fingerprint": split_fingerprint(scenes),
        "proposal_summary": proposal_summary(scenes),
    }


def load_archive(path: Path) -> tuple[SceneSample, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("schema") != "graphreader.ocr-crop-evidence-role-anchor-fixtures.v1"
            or manifest.get("revision") != REVISION
            or manifest.get("synthetic_only") is not True
            or manifest.get("chandler_included") is not False
            or manifest.get("private_or_article_images") is not False
        ):
            raise RuntimeError("OCR V24 stored fixture contract changed")
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["source_path"])
            if sha256(payload).hexdigest() != case["source_sha256"]:
                raise RuntimeError(f"OCR V24 fixture hash mismatch: {case['scene_id']}")
            raster = np.asarray(Image.open(BytesIO(payload)).convert("L"), dtype=np.uint8).copy()
            scenes.append(SceneSample(
                case["scene_id"],
                manifest["split"],
                case["renderer_family"],
                case["degradation_family"],
                raster,
                Box(*case["plot"]),
                tuple(
                    RoleTruth(Box(*truth["box"]), truth["role"], truth["text"])
                    for truth in case["truths"]
                ),
            ))
    result = tuple(scenes)
    registered = split_registration(manifest["split"])
    if len(result) != registered.scene_count:
        raise RuntimeError("OCR V24 stored fixture scene count changed")
    proposal_summary(result)
    return result


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_scene", "load_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene",
    "save_archive", "split_fingerprint",
]
