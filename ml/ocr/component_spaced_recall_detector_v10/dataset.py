# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh spaced-label graph scenes for OCR detector V10."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.component_context_detector_v7.dataset import (
    SceneSample, box_iou, encode_proposal as encode_v9_proposal, proposals,
)
from ml.ocr.production_composition_v1.dataset import _draw_structures, _text_mask

from .protocol import SCENE_HEIGHT, SCENE_WIDTH, TRUTH_MATCH_IOU_MINIMUM, split_registration


REPO_ROOT = Path(__file__).resolve().parents[3]
_NUMBERS = ("0", "5", "10", "20", "25", "40", "50", "75", "80", "100", "-2", "2.5", "33%")
_PHASES = ("Baseline", "Treatment", "Maintenance", "Followup", "Intervention")
_ANNOTATIONS = ("O o l I", "A B C", "Low High", "Probe Set", "Data Check", "Phase Note", "Rule One")
_LEGENDS = ("Target", "Series", "Rate", "Data", "Level", "Plan", "Probe")


def _rng(split: str, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"component-spaced-recall-v10:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    return ImageFont.truetype(str(REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]), size=size)


def _render_scene(split: str, index: int) -> SceneSample:
    registration = split_registration(split)
    rng = _rng(split, index)
    background, ink = int(rng.integers(248, 256)), int(rng.integers(9, 59))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(0.0, float(rng.integers(0, 6)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    _draw_structures(draw, rng, index + 503, ink)
    labels = (
        _NUMBERS[(index * 7 + 2) % len(_NUMBERS)], _NUMBERS[(index * 11 + 4) % len(_NUMBERS)],
        _PHASES[(index * 3 + 1) % len(_PHASES)], _ANNOTATIONS[(index * 5 + 2) % len(_ANNOTATIONS)],
        _LEGENDS[(index * 4 + 1) % len(_LEGENDS)],
    )
    y_slot = 72 + (index * 29) % 104
    x_slot = 174 + (index * 43) % 220
    phase_slot = 137 + (index * 47) % 185
    annotation_x = 342 + (index * 31) % 105
    annotation_y = 100 + (index * 19) % 53
    if split == "validation":
        y_slot = 79 + (index * 37) % 93
        x_slot = 184 + (index * 53) % 207
        phase_slot = 149 + (index * 59) % 167
        annotation_x = 350 + (index * 41) % 94
    elif split == "sealed_public":
        y_slot = 85 + (index * 41) % 83
        x_slot = 193 + (index * 61) % 193
        phase_slot = 157 + (index * 67) % 151
        annotation_x = 358 + (index * 43) % 84
        annotation_y = 105 + (index * 31) % 43
    draw.rectangle((324, annotation_y - 5, 638, annotation_y + 35), fill=background)
    draw.rectangle((568, 161, 639, 222), fill=background)
    positions = ((17, y_slot), (x_slot, 282), (phase_slot, 7), (annotation_x, annotation_y), (582, 176 + (index * 5) % 9))
    truths = []
    for label_index, (label, position) in enumerate(zip(labels, positions, strict=True)):
        font = _font(split, index + label_index, int(rng.integers(19, 23)))
        mask, bounds = _text_mask(image.size, label, position, font)
        image.paste(ink, mask=mask)
        truths.append(bounds)
    if split == "train":
        if index % 5 == 0:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        pixels = np.asarray(image, dtype=np.float32)
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(0.95, 1.07)) + 128.0, 0, 255)
    elif split == "validation":
        reduced = image.resize((624, 312), resample=Image.Resampling.BOX)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32)
        column = 53 + (index * 47) % 530
        pixels[:, max(0, column - 1) : min(SCENE_WIDTH, column + 2)] = np.maximum(
            pixels[:, max(0, column - 1) : min(SCENE_WIDTH, column + 2)], 172
        )
    else:
        pixels = np.asarray(image, dtype=np.float32)
        pixels = 255.0 * np.power(pixels / 255.0, float(rng.uniform(0.94, 1.06)))
        pixels = np.rint(pixels).astype(np.uint16) // 4 * 4
        row = 41 + (index * 37) % 237
        pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :] = np.maximum(
            pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :], 173
        )
    raster = np.rint(pixels).clip(0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 7))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(188, 241))
    return SceneSample(
        f"component-spaced-recall-v10-{split}-{index:05d}", split, registration.renderer_family,
        registration.degradation_family, raster, tuple(truths),
    )


def build_split(split: str) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(_render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: object) -> np.ndarray:
    return encode_v9_proposal(gray, proposal)


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(f"{truth.left},{truth.top},{truth.right},{truth.bottom}\n".encode())
    return digest.hexdigest()


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, int]:
    proposal_count = positive_count = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        proposal_count += len(candidates)
        positive_count += sum(any(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for truth in scene.truths) for candidate in candidates)
        for truth in scene.truths:
            matches = sum(box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM for candidate in candidates)
            if matches != 1:
                raise RuntimeError(f"{scene.scene_id} truth {truth} has {matches} proposals")
    return {
        "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "proposal_count": proposal_count, "positive_proposal_count": positive_count,
        "negative_proposal_count": proposal_count - positive_count,
    }


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def save_sealed_public_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "graphreader.ocr-spaced-component-recall-fixtures.v1", "split": "sealed_public",
        "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "split_fingerprint": split_fingerprint(scenes), "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False, "generalization_label_included": False,
        "cases": [],
    }
    images: list[tuple[str, bytes]] = []
    for scene in scenes:
        name, payload = f"images/{scene.scene_id}.png", _png_bytes(scene.raster)
        images.append((name, payload))
        manifest["cases"].append({
            "scene_id": scene.scene_id, "image_path": name, "image_sha256": sha256(payload).hexdigest(),
            "raster_sha256": sha256(scene.raster.tobytes(order="C")).hexdigest(),
            "renderer_family": scene.renderer_family, "degradation_family": scene.degradation_family,
            "truths": [[box.left, box.top, box.right, box.bottom] for box in scene.truths],
        })
    with ZipFile(path, "x") as archive:
        for name, payload in [("manifest.json", canonical_json_bytes(manifest)), *sorted(images)]:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = ZIP_DEFLATED, 0o100644 << 16
            archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)
    return {
        "schema": "graphreader.ocr-spaced-component-recall-private-manifest.v1",
        "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "split_fingerprint": split_fingerprint(scenes), "fixture_archive_sha256": sha256_file(path),
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False,
    }


def load_sealed_public_archive(path: Path) -> tuple[SceneSample, ...]:
    from ml.ocr.component_region_detector_v6.dataset import Box
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        scenes = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256(payload).hexdigest() != case["image_sha256"]:
                raise RuntimeError("OCR V10 fixture PNG changed")
            with Image.open(BytesIO(payload)) as image:
                raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            if sha256(raster.tobytes(order="C")).hexdigest() != case["raster_sha256"]:
                raise RuntimeError("OCR V10 fixture raster changed")
            scenes.append(SceneSample(
                case["scene_id"], "sealed_public", case["renderer_family"], case["degradation_family"],
                raster, tuple(Box(*values) for values in case["truths"]),
            ))
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR V10 fixture fingerprint changed")
    return result


__all__ = [
    "build_split", "encode_proposal", "load_sealed_public_archive", "proposal_summary", "proposals",
    "save_sealed_public_archive", "split_fingerprint",
]
