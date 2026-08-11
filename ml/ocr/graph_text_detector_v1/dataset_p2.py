# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""P2 training patches sampled after exact production-scale resizing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .dataset import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GENERIC_TEXT,
    _degrade_frame,
    _draw_frame_structures,
    _font,
)
from .protocol import TRAIN_SAMPLE_COUNT, split_registration


P2_SEED = 20260902
P2_PATCH_WIDTH = 512
P2_PATCH_HEIGHT = 192
P2_RENDERER_FAMILY = "production-scale-context-crops-v2"
P2_DEGRADATION_FAMILY = "source-render-then-production-resize-v2"


@dataclass(frozen=True)
class ProductionScalePatch:
    sample_id: str
    kind: str
    bgr: np.ndarray
    target: np.ndarray
    renderer_family: str
    degradation_family: str


def _rng(index: int) -> np.random.Generator:
    material = f"graph-text-detector-v1:P2:{P2_SEED}:{index}".encode()
    return np.random.default_rng(int.from_bytes(sha256(material).digest()[:8], "little"))


def _production_resize(values: np.ndarray, interpolation: int) -> np.ndarray:
    ratio = 960.0 / max(FRAME_WIDTH, FRAME_HEIGHT)
    resized_width = int(FRAME_WIDTH * ratio)
    resized_height = int(FRAME_HEIGHT * ratio)
    target_width = ((resized_width + 127) // 128) * 128
    target_height = ((resized_height + 127) // 128) * 128
    return cv2.resize(values, (target_width, target_height), interpolation=interpolation)


def _render_source(index: int) -> tuple[np.ndarray, np.ndarray, str, str]:
    registration = split_registration("train")
    rng = _rng(index)
    image = Image.new(
        "RGB",
        (FRAME_WIDTH, FRAME_HEIGHT),
        tuple(int(value) for value in rng.integers(247, 256, size=3)),
    )
    draw = ImageDraw.Draw(image)
    structure_family, masks = _draw_frame_structures(draw, index + 8192, sealed=False)
    target = Image.new("L", (FRAME_WIDTH, FRAME_HEIGHT), 0)
    target_draw = ImageDraw.Draw(target)
    kind = "text" if index < registration.text_count else "exclusion"
    if kind == "text":
        text = GENERIC_TEXT[(index * 11 + 3) % len(GENERIC_TEXT)]
        size = int(rng.integers(18, 32))
        if len(text) > 6:
            size = min(size, 24)
        font = _font(rng, size)
        raw = draw.textbbox((0, 0), text, font=font)
        width = raw[2] - raw[0]
        height = raw[3] - raw[1]
        anchors = ((74, 9), (258, 105), (8, 84), (132, 10), (239, 127), (75, 132))
        anchor_x, anchor_y = anchors[(index * 7 + 2) % len(anchors)]
        x = min(max(4, anchor_x + int(rng.integers(-8, 9))), FRAME_WIDTH - width - 5)
        y = min(max(3, anchor_y + int(rng.integers(-5, 6))), FRAME_HEIGHT - height - 5)
        draw.text(
            (x, y),
            text,
            font=font,
            fill=tuple(int(value) for value in rng.integers(7, 55, size=3)),
        )
        box = draw.textbbox((x, y), text, font=font)
        target_draw.rounded_rectangle(
            (max(0, box[0] - 1), max(0, box[1] - 1), min(FRAME_WIDTH - 1, box[2] + 1), min(FRAME_HEIGHT - 1, box[3] + 1)),
            radius=1,
            fill=255,
        )
    image, degradation = _degrade_frame(image, index + 8192, sealed=False)
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    for rectangle in masks:
        bgr[rectangle["top"] : rectangle["bottom"], rectangle["left"] : rectangle["right"], :] = 255
    return bgr, np.asarray(target, dtype=np.uint8), kind, f"{structure_family}:{degradation}"


def render_production_scale_patch(index: int) -> ProductionScalePatch:
    if not 0 <= index < TRAIN_SAMPLE_COUNT:
        raise ValueError("P2 training patch index is out of range")
    rng = _rng(index)
    source_bgr, source_target, kind, degradation = _render_source(index)
    detector_bgr = _production_resize(source_bgr, cv2.INTER_LINEAR)
    detector_target = _production_resize(source_target, cv2.INTER_NEAREST)
    maximum_left = detector_bgr.shape[1] - P2_PATCH_WIDTH
    maximum_top = detector_bgr.shape[0] - P2_PATCH_HEIGHT
    if kind == "text":
        ys, xs = np.nonzero(detector_target)
        if len(xs) == 0:
            raise RuntimeError("P2 text target disappeared during production resize")
        if int(xs.max() - xs.min() + 1) > P2_PATCH_WIDTH - 16:
            raise RuntimeError("P2 text target does not fit the preregistered crop")
        center_x = int(round((float(xs.min()) + float(xs.max())) / 2.0)) + int(rng.integers(-32, 33))
        center_y = int(round((float(ys.min()) + float(ys.max())) / 2.0)) + int(rng.integers(-20, 21))
        left = min(max(0, center_x - (P2_PATCH_WIDTH // 2)), maximum_left)
        top = min(max(0, center_y - (P2_PATCH_HEIGHT // 2)), maximum_top)
        if xs.min() < left or xs.max() >= left + P2_PATCH_WIDTH:
            left = min(max(0, int(xs.max()) - P2_PATCH_WIDTH + 8), maximum_left)
        if ys.min() < top or ys.max() >= top + P2_PATCH_HEIGHT:
            top = min(max(0, int(ys.max()) - P2_PATCH_HEIGHT + 8), maximum_top)
        if (
            xs.min() < left
            or xs.max() >= left + P2_PATCH_WIDTH
            or ys.min() < top
            or ys.max() >= top + P2_PATCH_HEIGHT
        ):
            raise RuntimeError("P2 text target is not fully contained by its deterministic crop")
    else:
        left = int(rng.integers(0, maximum_left + 1))
        top = int(rng.integers(0, maximum_top + 1))
    right = left + P2_PATCH_WIDTH
    bottom = top + P2_PATCH_HEIGHT
    return ProductionScalePatch(
        sample_id=f"graph-text-detector-v1-p2-train-{index:05d}",
        kind=kind,
        bgr=np.ascontiguousarray(detector_bgr[top:bottom, left:right, :]),
        target=np.ascontiguousarray(detector_target[top:bottom, left:right]),
        renderer_family=P2_RENDERER_FAMILY,
        degradation_family=f"{P2_DEGRADATION_FAMILY}:{degradation}",
    )


def build_p2_training_arrays() -> tuple[np.ndarray, np.ndarray]:
    samples = [render_production_scale_patch(index) for index in range(TRAIN_SAMPLE_COUNT)]
    return (
        np.stack([sample.bgr for sample in samples]).astype(np.uint8),
        np.stack([sample.target for sample in samples])[:, None, :, :].astype(np.uint8),
    )


def p2_training_split_fingerprint() -> str:
    records: list[dict[str, object]] = []
    for index in range(TRAIN_SAMPLE_COUNT):
        sample = render_production_scale_patch(index)
        records.append(
            {
                "sample_id": sample.sample_id,
                "kind": sample.kind,
                "bgr_sha256": sha256(sample.bgr.tobytes(order="C")).hexdigest(),
                "target_sha256": sha256(sample.target.tobytes(order="C")).hexdigest(),
                "renderer_family": sample.renderer_family,
                "degradation_family": sample.degradation_family,
            }
        )
    return sha256((json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


__all__ = [
    "P2_DEGRADATION_FAMILY",
    "P2_PATCH_HEIGHT",
    "P2_PATCH_WIDTH",
    "P2_RENDERER_FAMILY",
    "P2_SEED",
    "ProductionScalePatch",
    "build_p2_training_arrays",
    "p2_training_split_fingerprint",
    "render_production_scale_patch",
]
