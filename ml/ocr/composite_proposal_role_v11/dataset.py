# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh structure-dense eight-role scenes for OCR V11."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.component_context_detector_v7.dataset import (
    box_iou,
    encode_proposal as encode_v7_proposal,
    proposals,
)
from ml.ocr.component_region_detector_v6.dataset import Box, Component
from ml.ocr.production_composition_v1.dataset import _draw_structures, _text_mask
from .protocol import (
    ENCODED_WIDTH,
    GEOMETRY_FEATURE_COUNT,
    INPUT_CHANNELS,
    ROLE_ORDER,
    SCENE_HEIGHT,
    SCENE_WIDTH,
    TRUTH_MATCH_IOU_MINIMUM,
    split_registration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
Split = Literal["train", "validation", "sealed_public"]
_Y_TICKS = ("0", "20", "40", "60", "80", "100", "-5", "2.5")
_X_TICKS = ("1", "4", "8", "12", "16", "20", "24", "30")
_AXIS_TITLES = ("Session", "Observation", "Trials", "Visits")
_PHASES = ("Baseline", "Treatment", "Followup", "Maintenance", "Intervention")
_LEGENDS = ("Target", "Rate", "Level", "Probe", "Data")
_PARTICIPANTS = ("Learner", "Student", "Client", "Observer", "Case")
_ANNOTATIONS = ("Probe note", "Level shift", "Check point", "Rule change", "Data note")
_OTHERS = ("Weekly", "Daily", "Measure", "Outcome", "Summary")


@dataclass(frozen=True)
class RoleTruth:
    box: Box
    role: str
    text: str


@dataclass(frozen=True)
class SceneSample:
    scene_id: str
    split: Split
    renderer_family: str
    degradation_family: str
    raster: np.ndarray
    truths: tuple[RoleTruth, ...]


def _rng(split: Split, index: int) -> np.random.Generator:
    registration = split_registration(split)
    material = f"composite-proposal-role-v11:{registration.seed_offset}:{split}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _font(split: Split, index: int, size: int) -> ImageFont.FreeTypeFont:
    registration = split_registration(split)
    path = REPO_ROOT / registration.font_paths[index % len(registration.font_paths)]
    return ImageFont.truetype(str(path), size=size)


def render_scene(split: Split, index: int) -> SceneSample:
    registration = split_registration(split)
    rng = _rng(split, index)
    background = int(rng.integers(247, 256))
    ink = int(rng.integers(9, 54))
    base = np.full((SCENE_HEIGHT, SCENE_WIDTH), background, dtype=np.float32)
    base -= np.linspace(0.0, float(rng.integers(0, 7)), SCENE_WIDTH, dtype=np.float32)[None, :]
    base -= np.linspace(0.0, float(rng.integers(0, 5)), SCENE_HEIGHT, dtype=np.float32)[:, None]
    image = Image.fromarray(np.rint(base).clip(0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    _draw_structures(draw, rng, index + 1103, ink)
    labels = (
        (_Y_TICKS[(index * 3 + 1) % len(_Y_TICKS)], "YTick", (18, 112 + (index * 23) % 76)),
        (_X_TICKS[(index * 5 + 2) % len(_X_TICKS)], "XTick", (190 + (index * 37) % 210, 278)),
        (_AXIS_TITLES[(index * 7 + 1) % len(_AXIS_TITLES)], "AxisTitle", (274 + (index * 11) % 45, 299)),
        (_PHASES[(index * 2 + 1) % len(_PHASES)], "PhaseHeading", (168 + (index * 41) % 170, 4)),
        (_LEGENDS[(index * 4 + 1) % len(_LEGENDS)], "LegendText", (538, 166 + (index * 2) % 8)),
        (_PARTICIPANTS[(index * 3 + 2) % len(_PARTICIPANTS)], "Participant", (534, 278)),
        (_ANNOTATIONS[(index * 4 + 2) % len(_ANNOTATIONS)], "Annotation", (329 + (index * 29) % 70, 114 + (index * 17) % 42)),
        (_OTHERS[(index * 3 + 1) % len(_OTHERS)], "Other", (10, 4)),
    )
    truths: list[RoleTruth] = []
    for label_index, (text, role, position) in enumerate(labels):
        size = 16 + int(rng.integers(0, 3))
        if role in {"YTick", "XTick"}:
            size += 1
        font = _font(split, index + label_index, size)
        mask, bounds = _text_mask(image.size, text, position, font)
        if role == "LegendText":
            clear = (520, 154, SCENE_WIDTH - 1, 202)
        elif role == "Annotation":
            clear = (300, max(0, bounds.top - 10), 510, min(SCENE_HEIGHT - 1, bounds.bottom + 11))
        else:
            clear = (
                max(0, bounds.left - 3), max(0, bounds.top - 2),
                min(SCENE_WIDTH - 1, bounds.right + 3), min(SCENE_HEIGHT - 1, bounds.bottom + 2),
            )
        draw.rectangle(clear, fill=background)
        image.paste(ink, mask=mask)
        truths.append(RoleTruth(bounds, role, text))
    if split == "train":
        pixels = np.asarray(image, dtype=np.float32)
        pixels = np.clip((pixels - 128.0) * float(rng.uniform(1.00, 1.08)) + 128.0, 0, 255)
    elif split == "validation":
        pixels = np.asarray(image, dtype=np.float32)
        row = 43 + (index * 31) % 232
        pixels[max(0, row - 1):min(SCENE_HEIGHT, row + 2), :] = np.maximum(
            pixels[max(0, row - 1):min(SCENE_HEIGHT, row + 2), :], 177
        )
    else:
        pixels = np.asarray(image, dtype=np.float32)
        pixels = 255.0 * np.power(pixels / 255.0, float(rng.uniform(0.93, 1.07)))
        pixels = np.rint(pixels).astype(np.uint16) // 4 * 4
        column = 47 + (index * 43) % 548
        pixels[:, max(0, column - 1):min(SCENE_WIDTH, column + 2)] = np.maximum(
            pixels[:, max(0, column - 1):min(SCENE_WIDTH, column + 2)], 179
        )
    raster = np.rint(pixels).clip(0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 8))):
        raster[int(rng.integers(0, SCENE_HEIGHT)), int(rng.integers(0, SCENE_WIDTH))] = int(rng.integers(186, 242))
    return SceneSample(
        f"composite-proposal-role-v11-{split}-{index:05d}", split,
        registration.renderer_family, registration.degradation_family, raster, tuple(truths),
    )


def build_split(split: Split) -> tuple[SceneSample, ...]:
    registration = split_registration(split)
    return tuple(render_scene(split, index) for index in range(registration.scene_count))


def encode_proposal(gray: np.ndarray, proposal: Component) -> np.ndarray:
    base = encode_v7_proposal(gray, proposal)
    if base.shape != (INPUT_CHANNELS, 32, ENCODED_WIDTH - 4):
        raise RuntimeError("OCR V11 predecessor encoding changed")
    result = np.zeros((INPUT_CHANNELS, 32, ENCODED_WIDTH), dtype=np.float32)
    result[:, :, :base.shape[2]] = base
    position = np.asarray(
        (
            (proposal.left + proposal.right + 1.0) / (2.0 * gray.shape[1]),
            (proposal.top + proposal.bottom + 1.0) / (2.0 * gray.shape[0]),
            proposal.left / gray.shape[1],
            proposal.top / gray.shape[0],
        ),
        dtype=np.float32,
    )
    result[:, :, -4:] = position[None, None, :]
    if result.shape[2] != 128 + GEOMETRY_FEATURE_COUNT:
        raise RuntimeError("OCR V11 geometry contract changed")
    return result


def proposal_targets(scene: SceneSample, items: tuple[Component, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    candidates = proposals(scene.raster) if items is None else items
    accepted: list[int] = []
    roles: list[int] = []
    for candidate in candidates:
        matches = [
            (index, box_iou(candidate.box, truth.box))
            for index, truth in enumerate(scene.truths)
            if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
        ]
        if not matches:
            accepted.append(0)
            roles.append(-1)
            continue
        best = max(matches, key=lambda item: item[1])[0]
        accepted.append(1)
        roles.append(ROLE_ORDER.index(scene.truths[best].role))
    return np.asarray(accepted, dtype=np.int64), np.asarray(roles, dtype=np.int64)


def training_examples(scenes: tuple[SceneSample, ...], *, negative_cap_per_scene: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    values: list[np.ndarray] = []
    proposal_labels: list[int] = []
    role_labels: list[int] = []
    stream = sha256()
    positives = negatives = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        accept, roles = proposal_targets(scene, candidates)
        positive_indices = np.flatnonzero(accept == 1).tolist()
        negative_indices = np.flatnonzero(accept == 0).tolist()[:negative_cap_per_scene]
        for index in positive_indices + negative_indices:
            encoded = encode_proposal(scene.raster, candidates[index])
            values.append(encoded)
            proposal_labels.append(int(accept[index]))
            role_labels.append(int(roles[index]))
            stream.update(scene.scene_id.encode())
            stream.update(encoded.tobytes(order="C"))
            stream.update(bytes((int(accept[index]), int(roles[index]) + 1)))
        positives += len(positive_indices)
        negatives += len(negative_indices)
    if not values or positives == 0 or negatives == 0:
        raise RuntimeError("OCR V11 training examples require positive and negative proposals")
    evidence: dict[str, object] = {
        "scene_count": len(scenes), "negative_cap_per_scene": negative_cap_per_scene,
        "proposal_count": len(values), "positive_proposal_count": positives,
        "negative_proposal_count": negatives, "tensor_label_stream_sha256": stream.hexdigest(),
        "validation_or_public_pixels_used": False, "v2_bytes_used": False,
    }
    return (
        np.stack(values).astype(np.float32), np.asarray(proposal_labels, dtype=np.int64),
        np.asarray(role_labels, dtype=np.int64), evidence,
    )


def split_fingerprint(scenes: tuple[SceneSample, ...]) -> str:
    digest = sha256()
    for scene in scenes:
        digest.update(scene.scene_id.encode())
        digest.update(scene.renderer_family.encode())
        digest.update(scene.degradation_family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(f"{truth.box.left},{truth.box.top},{truth.box.right},{truth.box.bottom}|{truth.role}|{truth.text}\n".encode())
    return digest.hexdigest()


def proposal_summary(scenes: tuple[SceneSample, ...]) -> dict[str, object]:
    proposal_count = positive_count = 0
    role_counts = {role: 0 for role in ROLE_ORDER}
    for scene in scenes:
        candidates = proposals(scene.raster)
        accept, _ = proposal_targets(scene, candidates)
        proposal_count += len(candidates)
        positive_count += int(accept.sum())
        for truth in scene.truths:
            matches = sum(box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM for candidate in candidates)
            if matches != 1:
                raise RuntimeError(f"{scene.scene_id} {truth.role} truth has {matches} proposals")
            role_counts[truth.role] += 1
    return {
        "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "proposal_count": proposal_count, "positive_proposal_count": positive_count,
        "negative_proposal_count": proposal_count - positive_count, "role_truth_counts": role_counts,
    }


def _png_bytes(raster: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(raster, mode="L").save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def save_sealed_public_archive(scenes: tuple[SceneSample, ...], path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "graphreader.ocr-composite-proposal-role-fixtures.v1", "split": "sealed_public",
        **proposal_summary(scenes), "split_fingerprint": split_fingerprint(scenes), "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False, "generalization_label_included": False,
        "v2_bytes_used": False, "cases": [],
    }
    images: list[tuple[str, bytes]] = []
    cases = manifest["cases"]
    assert isinstance(cases, list)
    for scene in scenes:
        name, payload = f"images/{scene.scene_id}.png", _png_bytes(scene.raster)
        images.append((name, payload))
        cases.append({
            "scene_id": scene.scene_id, "image_path": name, "image_sha256": sha256(payload).hexdigest(),
            "raster_sha256": sha256(scene.raster.tobytes(order="C")).hexdigest(),
            "renderer_family": scene.renderer_family, "degradation_family": scene.degradation_family,
            "truths": [
                {"box": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom], "role": truth.role, "text": truth.text}
                for truth in scene.truths
            ],
        })
    with ZipFile(path, "x") as archive:
        for name, payload in [("manifest.json", canonical_json_bytes(manifest)), *sorted(images)]:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = ZIP_DEFLATED, 0o100644 << 16
            archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)
    return {
        "schema": "graphreader.ocr-composite-proposal-role-private-manifest.v1",
        **proposal_summary(scenes), "split_fingerprint": split_fingerprint(scenes),
        "fixture_archive_sha256": sha256_file(path), "synthetic_only": True,
        "private_or_article_images": False, "chandler_included": False,
        "generalization_label_included": False, "v2_bytes_used": False,
    }


def load_sealed_public_archive(path: Path) -> tuple[SceneSample, ...]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        scenes: list[SceneSample] = []
        for case in manifest["cases"]:
            payload = archive.read(case["image_path"])
            if sha256(payload).hexdigest() != case["image_sha256"]:
                raise RuntimeError("OCR V11 fixture PNG changed")
            with Image.open(BytesIO(payload)) as image:
                raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            if sha256(raster.tobytes(order="C")).hexdigest() != case["raster_sha256"]:
                raise RuntimeError("OCR V11 fixture raster changed")
            truths = tuple(RoleTruth(Box(*truth["box"]), truth["role"], truth["text"]) for truth in case["truths"])
            scenes.append(SceneSample(
                case["scene_id"], "sealed_public", case["renderer_family"],
                case["degradation_family"], raster, truths,
            ))
    result = tuple(scenes)
    if split_fingerprint(result) != manifest["split_fingerprint"]:
        raise RuntimeError("OCR V11 fixture fingerprint changed")
    return result


__all__ = [
    "RoleTruth", "SceneSample", "build_split", "encode_proposal", "load_sealed_public_archive",
    "proposal_summary", "proposal_targets", "proposals", "render_scene", "save_sealed_public_archive",
    "split_fingerprint", "training_examples",
]
