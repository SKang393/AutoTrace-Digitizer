# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh procedural graph scenes for OCR production-composition V3."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals
from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.production_composition_v1.dataset import _draw_structures, _text_mask

from .protocol import REVISION, SCENE_HEIGHT, SCENE_WIDTH, TRUTH_MATCH_IOU_MINIMUM, split_registration


REPO_ROOT = Path(__file__).resolve().parents[3]
NUMERIC_LABELS = (
    ("0", "0"), ("5", "5"), ("10", "10"), ("20", "20"), ("25", "25"),
    ("40", "40"), ("50", "50"), ("75", "75"), ("80", "80"), ("100", "100"),
    ("-2", "-2"), ("2.5", "2.5"), ("33%", "33%"),
)
PHASE_WORDS = ("Baseline", "Treatment", "Maintenance", "Followup", "Intervention")
ANNOTATION_WORDS = ("O o l I", "A B C", "Low High", "Probe Set", "Data Check", "Phase Note", "Rule One")
LEGEND_WORDS = ("Target", "Series", "Rate", "Data", "Level", "Plan", "Probe")


@dataclass(frozen=True)
class TextTruth:
    display_text: str
    truth_text: str
    role: str
    family: str
    box: Box


@dataclass(frozen=True)
class CompositionScene:
    scene_id: str
    split: str
    renderer_family: str
    degradation_family: str
    raster: np.ndarray
    truths: tuple[TextTruth, ...]


def _rng(split: str, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"production-composition-v3:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    return ImageFont.truetype(str(REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]), size=size)


def _render_scene(split: str, index: int) -> CompositionScene:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(248, 256))
    ink = int(rng.integers(9, 58))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(float(rng.integers(0, 3)), float(rng.integers(3, 7)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(float(rng.integers(0, 3)), float(rng.integers(2, 6)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    _draw_structures(draw, rng, index + 809, ink)

    y_display, y_truth = NUMERIC_LABELS[(index * 5 + 7) % len(NUMERIC_LABELS)]
    x_display, x_truth = NUMERIC_LABELS[(index * 9 + 4) % len(NUMERIC_LABELS)]
    phase = PHASE_WORDS[(index * 2 + 3) % len(PHASE_WORDS)]
    annotation = ANNOTATION_WORDS[(index * 3 + 2) % len(ANNOTATION_WORDS)]
    legend = LEGEND_WORDS[(index * 5 + 4) % len(LEGEND_WORDS)]
    y_slot = 78 + (index * 43) % 91
    x_slot = 185 + (index * 57) % 204
    phase_slot = 151 + (index * 53) % 166
    annotation_x = 352 + (index * 47) % 92
    annotation_y = 102 + (index * 31) % 47
    if split == "sealed_public":
        y_slot = 87 + (index * 47) % 79
        x_slot = 197 + (index * 67) % 185
        phase_slot = 161 + (index * 71) % 145
        annotation_x = 362 + (index * 53) % 80
        annotation_y = 108 + (index * 37) % 38
    draw.rectangle((326, annotation_y - 5, 638, annotation_y + 35), fill=background)
    draw.rectangle((568, 161, 639, 223), fill=background)
    labels = (
        (y_display, y_truth, "y_tick", "numeric", (17, y_slot)),
        (x_display, x_truth, "x_tick", "numeric", (x_slot, 282)),
        (phase, phase, "phase_heading", "word", (phase_slot, 7)),
        (annotation, annotation, "annotation", "ambiguity" if annotation == "O o l I" else "word", (annotation_x, annotation_y)),
        (legend, legend, "legend_text", "word", (582, 176 + (index * 3) % 9)),
    )
    truths: list[TextTruth] = []
    for label_index, (display, truth, role, family, position) in enumerate(labels):
        font = _font(split, index + label_index, int(rng.integers(19, 23)))
        mask, bounds = _text_mask(image.size, display, position, font)
        image.paste(ink, mask=mask)
        truths.append(TextTruth(display, truth, role, family, bounds))

    if split == "validation":
        reduced = image.resize((632, 316), resample=Image.Resampling.BILINEAR)
        image = reduced.resize((SCENE_WIDTH, SCENE_HEIGHT), resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32)
        row = 49 + (index * 41) % 222
        pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :] = np.maximum(
            pixels[max(0, row - 1) : min(SCENE_HEIGHT, row + 2), :], 176
        )
    else:
        pixels = np.asarray(image, dtype=np.float32)
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(0.96, 1.045)) + 128.0, 0, 255)
        pixels = np.rint(pixels).astype(np.uint16) // 3 * 3
        column = 67 + (index * 61) % 506
        pixels[:, max(0, column - 1) : min(SCENE_WIDTH, column + 2)] = np.maximum(
            pixels[:, max(0, column - 1) : min(SCENE_WIDTH, column + 2)], 175
        )
    raster = np.rint(pixels).clip(0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 7))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(188, 241))
    return CompositionScene(
        f"ocr-production-composition-v3-{split}-{index:05d}", split,
        registration.renderer_family, registration.degradation_family, raster, tuple(truths),
    )


def build_split(split: str) -> tuple[CompositionScene, ...]:
    registration = split_registration(split)
    return tuple(_render_scene(split, index) for index in range(registration.scene_count))


def split_fingerprint(scenes: tuple[CompositionScene, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(
                f"{truth.display_text}\0{truth.truth_text}\0{truth.role}\0{truth.family}\0"
                f"{truth.box.left},{truth.box.top},{truth.box.right},{truth.box.bottom}\n".encode()
            )
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
        "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "proposal_count": proposal_count, "positive_proposal_count": positive_count,
        "negative_proposal_count": proposal_count - positive_count,
    }


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _zip_write(archive: ZipFile, name: str, payload: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)


def save_sealed_archive(scenes: tuple[CompositionScene, ...], path: Path) -> dict[str, object]:
    if path.exists():
        raise RuntimeError(f"OCR composition V3 archive exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    images: list[tuple[str, bytes]] = []
    for scene in scenes:
        name = f"images/{scene.scene_id}.png"
        payload = _png_bytes(scene.raster)
        images.append((name, payload))
        cases.append({
            "scene_id": scene.scene_id, "image_path": name, "image_sha256": sha256_bytes(payload),
            "raster_sha256": sha256_bytes(scene.raster.tobytes(order="C")),
            "renderer_family": scene.renderer_family, "degradation_family": scene.degradation_family,
            "truths": [{
                "display_text": truth.display_text, "truth_text": truth.truth_text, "role": truth.role,
                "family": truth.family,
                "bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom],
            } for truth in scene.truths],
        })
    manifest = {
        "schema": "graphreader.ocr-production-composition-fixtures.v3", "revision": REVISION,
        "split": scenes[0].split, "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "split_fingerprint": split_fingerprint(scenes), "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "predecessor_fixture_bytes_reused": False, "cases": cases,
    }
    with ZipFile(path, "x") as archive:
        _zip_write(archive, "manifest.json", canonical_json_bytes(manifest))
        for name, payload in sorted(images):
            _zip_write(archive, name, payload)
    return {
        "schema": "graphreader.ocr-production-composition-private-manifest.v3", "revision": REVISION,
        "split": scenes[0].split, "scene_count": len(scenes),
        "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "split_fingerprint": split_fingerprint(scenes), "fixture_archive_sha256": sha256_file(path),
        "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False,
    }


def load_sealed_archive(path: Path) -> tuple[CompositionScene, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("revision") != REVISION:
            raise RuntimeError("OCR composition V3 fixture revision changed")
        scenes: list[CompositionScene] = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256_bytes(payload) != case["image_sha256"]:
                raise RuntimeError("OCR composition V3 fixture PNG changed")
            with Image.open(BytesIO(payload)) as image:
                raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            if sha256_bytes(raster.tobytes(order="C")) != case["raster_sha256"]:
                raise RuntimeError("OCR composition V3 raster changed")
            scenes.append(CompositionScene(
                case["scene_id"], manifest["split"], case["renderer_family"], case["degradation_family"], raster,
                tuple(TextTruth(
                    item["display_text"], item["truth_text"], item["role"], item["family"], Box(*item["bbox"]),
                ) for item in case["truths"]),
            ))
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR composition V3 fixture fingerprint changed")
    return result


__all__ = [
    "CompositionScene", "TextTruth", "build_split", "load_sealed_archive", "proposal_summary",
    "save_sealed_archive", "split_fingerprint",
]
