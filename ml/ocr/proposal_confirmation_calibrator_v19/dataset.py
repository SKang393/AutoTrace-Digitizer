# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh stored-fixture graph scenes for OCR V19."""

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
from ml.ocr.layout_conditioned_proposal_role_v15 import dataset as v15
from .protocol import ROLE_ORDER, split_registration


Split = Literal["train", "validation", "sealed_public"]
RoleTruth = v15.RoleTruth
SceneSample = v15.SceneSample
proposals = v15.proposals
encode_proposal = v15.encode_proposal
proposal_summary = v15.proposal_summary
proposal_targets = v15.proposal_targets
split_fingerprint = v15.split_fingerprint
_RENDER_LOCK = RLock()


@contextmanager
def _registration_scope() -> Iterator[None]:
    with _RENDER_LOCK:
        original = v15.split_registration
        v15.split_registration = split_registration
        try:
            yield
        finally:
            v15.split_registration = original


def render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    if index < 0 or index >= registration.scene_count:
        raise IndexError(f"OCR V19 scene index outside frozen {split} split: {index}")
    with _registration_scope():
        source = v15.render_scene(split, index)
    return SceneSample(
        f"proposal-confirmation-calibrator-v19-{split}-{index:05d}",
        split,
        registration.renderer_family,
        registration.degradation_family,
        source.raster,
        source.plot,
        source.truths,
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registration.scene_count))


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def save_split_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    if not scenes or len({scene.split for scene in scenes}) != 1:
        raise RuntimeError("OCR V19 archive requires one nonempty frozen split")
    split = scenes[0].split
    manifest: dict[str, object] = {
        "schema": "graphreader.ocr-proposal-confirmation-calibrator-fixtures.v1",
        "split": split,
        **proposal_summary(scenes),
        "cases": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        cases: list[dict[str, object]] = []
        for scene in scenes:
            payload = _png_bytes(scene.raster)
            image_path = f"images/{scene.scene_id}.png"
            _zip_write(archive, image_path, payload)
            cases.append({
                "scene_id": scene.scene_id,
                "renderer_family": scene.renderer_family,
                "degradation_family": scene.degradation_family,
                "image_path": image_path,
                "source_sha256": sha256(payload).hexdigest(),
                "plot_box": [scene.plot.left, scene.plot.top, scene.plot.right, scene.plot.bottom],
            })
        manifest["cases"] = cases
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
    return {
        "schema": "graphreader.ocr-proposal-confirmation-calibrator-private-manifest.v1",
        "split": split,
        "fixture_archive_sha256": sha256_file(path),
        "truths": [
            {
                "scene_id": scene.scene_id,
                "items": [
                    {
                        "box": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom],
                        "role": truth.role,
                        "text": truth.text,
                    }
                    for truth in scene.truths
                ],
            }
            for scene in scenes
        ],
    }


def load_split_archive(
    path: Path, private_manifest_path: Path, *, expected_split: Split,
) -> tuple[SceneSample, ...]:
    private = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    if (
        private.get("schema") != "graphreader.ocr-proposal-confirmation-calibrator-private-manifest.v1"
        or private.get("split") != expected_split
        or private.get("fixture_archive_sha256") != sha256_file(path)
    ):
        raise RuntimeError("OCR V19 private manifest or archive identity mismatch")
    truth_by_id = {case["scene_id"]: case["items"] for case in private["truths"]}
    scenes: list[SceneSample] = []
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("schema") != "graphreader.ocr-proposal-confirmation-calibrator-fixtures.v1"
            or manifest.get("split") != expected_split
        ):
            raise RuntimeError("OCR V19 stored fixture identity mismatch")
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256(payload).hexdigest() != case["source_sha256"]:
                raise RuntimeError("OCR V19 stored fixture image checksum mismatch")
            raster = np.asarray(Image.open(BytesIO(payload)).convert("L"), dtype=np.uint8).copy()
            truths = tuple(
                RoleTruth(Box(*item["box"]), item["role"], item["text"])
                for item in truth_by_id[case["scene_id"]]
            )
            scenes.append(SceneSample(
                case["scene_id"], expected_split, case["renderer_family"],
                case["degradation_family"], raster, Box(*case["plot_box"]), truths,
            ))
    result = tuple(scenes)
    registration = split_registration(expected_split)
    if len(result) != registration.scene_count:
        raise RuntimeError("OCR V19 stored fixture scene count changed")
    if proposal_summary(result)["role_truth_counts"] != {role: len(result) for role in ROLE_ORDER}:
        raise RuntimeError("OCR V19 stored fixture role balance changed")
    return result


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_proposal", "load_split_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene", "save_split_archive",
    "split_fingerprint",
]
