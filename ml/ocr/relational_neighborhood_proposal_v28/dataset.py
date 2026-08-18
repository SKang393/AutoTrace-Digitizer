# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh stored procedural scenes for OCR relational-neighborhood V28."""

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
from ml.ocr.crop_evidence_role_anchor_v24 import dataset as v24
from ml.ocr.relational_scene_proposal_role_v21 import dataset as v21

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
        original = v24.split_registration
        v24.split_registration = split_registration
        try:
            yield
        finally:
            v24.split_registration = original


def _apply_degradation(raster: np.ndarray, split: Split, index: int) -> np.ndarray:
    """Apply a V28-only photometric transform without moving truth geometry."""
    values = raster.astype(np.float32)
    source_ink = raster < 128
    height, width = raster.shape
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    if split == "train":
        checker = ((np.indices((height, width)).sum(axis=0) + index) % 2).astype(
            np.float32,
        )
        values += (checker * 2.0 - 1.0) * float(0.25 + 0.25 * (index % 4))
        values += np.cos((x + 1.0) * np.pi * float(1 + index % 6)) * float(
            1 + index % 3,
        )
        gamma = 0.97 + 0.005 * float(index % 11)
        values = 255.0 * np.power(np.clip(values / 255.0, 0.0, 1.0), gamma)
        quantum = float(2 + (index * 3) % 5)
        values = np.rint(values / quantum) * quantum
    elif split == "validation":
        radius = np.sqrt(np.square(x * 0.75) + np.square(y))
        values += np.clip(radius, 0.0, 1.0) * float(1 + index % 5)
        values[index % 6::6, :] -= 0.75
        values[(index + 1) % 6::6, :] += 0.75
        diagonal = (x + y) * float(0.5 + 0.25 * (index % 5))
        values += diagonal
        quantum = float(3 + index % 4)
        values = np.floor(values / quantum + 0.5) * quantum
    else:
        gamma_left = 0.975 + 0.005 * float(index % 7)
        gamma_right = 1.015 - 0.005 * float(index % 5)
        normalized = np.clip(values / 255.0, 0.0, 1.0)
        left = 255.0 * np.power(normalized, gamma_left)
        right = 255.0 * np.power(normalized, gamma_right)
        blend = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
        values = left * (1.0 - blend) + right * blend
        values += (x - 0.5 * y) * float(1 + index % 4)
        column = 17 + (index * 29) % max(18, width - 18)
        values[:, max(0, column - 1):column + 1] += 1.0
        row = 11 + (index * 37) % max(12, height - 12)
        values[max(0, row - 1):row + 1, :] -= 0.75
    values = np.where(source_ink, np.minimum(values, 120.0), np.maximum(values, 136.0))
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def render_scene(split: Split, index: int) -> SceneSample:
    registered = split_registration(split)
    if index < 0 or index >= registered.scene_count:
        raise IndexError(f"OCR V28 scene index outside frozen {split} split: {index}")
    for attempt in range(64):
        source_index = (index + attempt * 103) % registered.scene_count
        with _registration_scope():
            source = v24.render_scene(split, source_index)
        candidate = SceneSample(
            f"relational-neighborhood-v28-{split}-{index:05d}",
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
    raise RuntimeError(
        f"OCR V28 could not render one production proposal per truth: {split}/{index}"
    )


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
    info = ZipInfo(name, date_time=(2026, 8, 18, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def save_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"OCR V28 fixture identity already exists: {path}")
    if not scenes or len({scene.split for scene in scenes}) != 1:
        raise RuntimeError("OCR V28 archive requires one nonempty split")
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
        "schema": "graphreader.ocr-relational-neighborhood-fixtures.v1",
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
            manifest.get("schema") != "graphreader.ocr-relational-neighborhood-fixtures.v1"
            or manifest.get("revision") != REVISION
            or manifest.get("synthetic_only") is not True
            or manifest.get("chandler_included") is not False
            or manifest.get("private_or_article_images") is not False
        ):
            raise RuntimeError("OCR V28 stored fixture contract changed")
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["source_path"])
            if sha256(payload).hexdigest() != case["source_sha256"]:
                raise RuntimeError(f"OCR V28 fixture hash mismatch: {case['scene_id']}")
            raster = np.asarray(
                Image.open(BytesIO(payload)).convert("L"), dtype=np.uint8,
            ).copy()
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
        raise RuntimeError("OCR V28 stored fixture scene count changed")
    proposal_summary(result)
    return result


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_scene", "load_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene",
    "save_archive", "split_fingerprint",
]
