# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh stored procedural scenes for OCR robust quorum recall V31."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.relational_scene_proposal_role_v21 import dataset as v21
from ml.ocr.unanimous_structure_veto_v30 import dataset as v30

from .protocol import REVISION, ROLE_ORDER, split_registration


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]
RoleTruth = v21.RoleTruth
SceneSample = v21.SceneSample
proposals = v21.proposals
proposal_targets = v21.proposal_targets
encode_scene = v21.encode_scene


def _apply_v31_degradation(
    raster: np.ndarray, split: Split, index: int,
) -> np.ndarray:
    """Apply a V31-only bounded nuisance field without changing truth geometry."""
    values = raster.astype(np.float32)
    source_ink = raster < 128
    height, width = raster.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(1.0, float(width - 1))
    y = yy / max(1.0, float(height - 1))
    if split == "train":
        field = np.sin((3 + index % 5) * np.pi * x) * np.cos(
            (2 + index % 4) * np.pi * y,
        )
        values += field * (0.5 + 0.25 * float(index % 5))
        quantum = float(2 + (index * 5) % 6)
    elif split == "validation":
        field = np.cos((4 + index % 6) * np.pi * (x + 0.3 * y))
        values += field * (0.75 + 0.25 * float(index % 4))
        quantum = float(3 + (index * 7) % 5)
    else:
        field = np.sin((5 + index % 7) * np.pi * (0.4 * x + y))
        values += field * (0.5 + 0.25 * float(index % 6))
        quantum = float(2 + (index * 11) % 7)
    values = np.rint(values / quantum) * quantum
    values = np.where(
        source_ink, np.minimum(values, 120.0), np.maximum(values, 136.0),
    )
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def render_scene(split: Split, index: int) -> SceneSample:
    registered = split_registration(split)
    if index < 0 or index >= registered.scene_count:
        raise IndexError(f"OCR V31 scene index outside frozen {split} split: {index}")
    for attempt in range(64):
        source_index = (index + attempt * 149) % registered.scene_count
        with v30._RENDER_LOCK:  # pylint: disable=protected-access
            original = v30.split_registration
            v30.split_registration = split_registration
            try:
                source = v30.render_scene(split, source_index)
            finally:
                v30.split_registration = original
        candidate = SceneSample(
            f"robust-quorum-recall-v31-{split}-{index:05d}",
            split,
            source.renderer_family,
            source.degradation_family,
            _apply_v31_degradation(source.raster, split, index),
            source.plot,
            source.truths,
        )
        try:
            v21.proposal_summary((candidate,))
        except RuntimeError:
            continue
        return candidate
    raise RuntimeError(
        f"OCR V31 could not render one production proposal per truth: {split}/{index}"
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


def save_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"OCR V31 fixture identity already exists: {path}")
    if not scenes or len({scene.split for scene in scenes}) != 1:
        raise RuntimeError("OCR V31 archive requires one nonempty split")
    summary = proposal_summary(scenes)
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
        "schema": "graphreader.ocr-robust-quorum-recall-fixtures.v1",
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
        v30._zip_write(archive, "manifest.json", manifest_bytes)  # pylint: disable=protected-access
        for name, payload in payloads:
            v30._zip_write(archive, name, payload)  # pylint: disable=protected-access
    return {
        "archive_path": path.relative_to(REPO_ROOT).as_posix(),
        "archive_sha256": sha256_file(path),
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "split_fingerprint": split_fingerprint(scenes),
        "proposal_summary": summary,
    }


def load_archive(path: Path | BytesIO) -> tuple[SceneSample, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("schema") != "graphreader.ocr-robust-quorum-recall-fixtures.v1"
            or manifest.get("revision") != REVISION
            or manifest.get("synthetic_only") is not True
            or manifest.get("chandler_included") is not False
            or manifest.get("generalization_label_included") is not False
            or manifest.get("private_or_article_images") is not False
        ):
            raise RuntimeError("OCR V31 stored fixture contract changed")
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["source_path"])
            if sha256(payload).hexdigest() != case["source_sha256"]:
                raise RuntimeError(f"OCR V31 fixture hash mismatch: {case['scene_id']}")
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
        raise RuntimeError("OCR V31 stored fixture scene count changed")
    proposal_summary(result)
    return result


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_scene", "load_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene",
    "save_archive", "split_fingerprint",
]
