# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh V8 splits from unexecuted V6 renderer indices."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals
from ml.ocr.production_composition_v6.dataset import CompositionScene, TextTruth, _render_scene
from .protocol import REVISION, TRUTH_MATCH_IOU_MINIMUM, split_registration


def build_split(split: str) -> tuple[CompositionScene, ...]:
    registration = split_registration(split)
    scenes = []
    for local_index in range(registration.scene_count):
        source_index = registration.source_index_offset + local_index
        source = _render_scene(split, source_index)
        scenes.append(replace(
            source,
            scene_id=f"ocr-production-composition-v8-{split}-{local_index:05d}",
            renderer_family=registration.renderer_family,
            degradation_family=registration.degradation_family,
        ))
    return tuple(scenes)


def split_fingerprint(scenes: tuple[CompositionScene, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(f"{truth.display_text}\0{truth.truth_text}\0{truth.role}\0{truth.family}\0{truth.box}\n".encode())
    return digest.hexdigest()


def proposal_summary(scenes: tuple[CompositionScene, ...]) -> dict[str, int]:
    proposal_count = positive_count = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        proposal_count += len(candidates)
        positive_count += sum(
            any(box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM for truth in scene.truths)
            for candidate in candidates
        )
        for truth in scene.truths:
            matches = sum(box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM for candidate in candidates)
            if matches != 1:
                raise RuntimeError(f"{scene.scene_id} truth {truth.role}:{truth.truth_text} has {matches} proposals")
    return {
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "proposal_count": proposal_count,
        "positive_proposal_count": positive_count,
        "negative_proposal_count": proposal_count - positive_count,
    }


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)


def save_sealed_archive(scenes: tuple[CompositionScene, ...], path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases, images = [], []
    for scene in scenes:
        name, payload = f"images/{scene.scene_id}.png", _png_bytes(scene.raster)
        images.append((name, payload))
        cases.append({
            "scene_id": scene.scene_id,
            "image_path": name,
            "image_sha256": sha256_bytes(payload),
            "raster_sha256": sha256_bytes(scene.raster.tobytes()),
            "renderer_family": scene.renderer_family,
            "degradation_family": scene.degradation_family,
            "truths": [
                {
                    "display_text": truth.display_text,
                    "truth_text": truth.truth_text,
                    "role": truth.role,
                    "family": truth.family,
                    "bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom],
                }
                for truth in scene.truths
            ],
        })
    manifest = {
        "schema": "graphreader.ocr-production-composition-fixtures.v8",
        "revision": REVISION,
        "split": scenes[0].split,
        "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "split_fingerprint": split_fingerprint(scenes),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "predecessor_fixture_bytes_reused": False,
        "cases": cases,
    }
    with ZipFile(path, "x") as archive:
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
        for name, payload in sorted(images):
            _zip_write(archive, name, payload)
    return {key: manifest[key] for key in (
        "schema", "revision", "split", "scene_count", "truth_region_count", "split_fingerprint",
        "synthetic_only", "private_or_article_images", "chandler_included", "generalization_label_included",
    )} | {"fixture_archive_sha256": sha256_file(path)}


def load_sealed_archive(path: Path) -> tuple[CompositionScene, ...]:
    from ml.ocr.component_region_detector_v6.dataset import Box

    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["revision"] != REVISION:
            raise RuntimeError("OCR composition V8 fixture revision changed")
        scenes = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256_bytes(payload) != case["image_sha256"]:
                raise RuntimeError("OCR composition V8 PNG changed")
            with Image.open(BytesIO(payload)) as image:
                raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            if sha256_bytes(raster.tobytes()) != case["raster_sha256"]:
                raise RuntimeError("OCR composition V8 raster changed")
            truths = tuple(
                TextTruth(item["display_text"], item["truth_text"], item["role"], item["family"], Box(*item["bbox"]))
                for item in case["truths"]
            )
            scenes.append(CompositionScene(
                case["scene_id"], manifest["split"], case["renderer_family"],
                case["degradation_family"], raster, truths,
            ))
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR composition V8 fingerprint changed")
    return result


__all__ = ["CompositionScene", "TextTruth", "build_split", "load_sealed_archive", "proposal_summary", "save_sealed_archive", "split_fingerprint"]
