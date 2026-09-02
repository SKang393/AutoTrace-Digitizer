# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Family-disjoint corrected real-range proposal data for V32."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from ml.ocr.component_context_detector_v7.dataset import Box, box_iou, encode_proposal, proposals
from ml.synthetic.dataset import CaseSpec, _build_scenes
from ml.synthetic.renderer import render_scene

from .protocol import DEV_SEED, NEGATIVE_CAP_PER_SCENE, TRAIN_SEED


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SceneSample:
    scene_id: str
    split: str
    renderer_family: str
    font_family: str
    degradation_family: str
    template_family: str
    marker_family: str
    raster: np.ndarray
    truths: tuple[Box, ...]


_TRAIN_SPECS = (
    CaseSpec("ab", "vector_clean", 1, 24, canvas_width=361, panel_height=160, marker_radius=3.0, stroke_width=1, presentation={"font_size_px": 17}),
    CaseSpec("multiple_probe", "vector_clean", 1, 24, ("sparse_probes",), canvas_width=863, panel_height=315, marker_radius=4.4, stroke_width=2, presentation={"font_size_px": 18, "dense_tick_labels": True}),
    CaseSpec("multiple_baseline", "print_monochrome", 1, 24, canvas_width=1338, panel_height=412, marker_radius=12.0, stroke_width=2, presentation={"font_size_px": 18}),
    CaseSpec("ab", "print_monochrome", 1, 24, canvas_width=6352, panel_height=520, marker_radius=6.2, stroke_width=3, presentation={"font_size_px": 18}),
    CaseSpec("abab", "print_monochrome", 1, 24, canvas_width=600, panel_height=4404, marker_radius=5.0, stroke_width=2, presentation={"font_size_px": 18}),
)
_DEV_SPECS = (
    CaseSpec("ab", "scan_rough", 1, 24, canvas_width=361, panel_height=160, marker_radius=3.0, stroke_width=1, presentation={"font_size_px": 17}),
    CaseSpec("multiple_probe", "scan_rough", 1, 24, ("sparse_probes",), canvas_width=863, panel_height=315, marker_radius=4.4, stroke_width=2, presentation={"font_size_px": 18, "dense_tick_labels": True}),
    CaseSpec("multiple_baseline", "hand_drawn", 1, 24, canvas_width=1338, panel_height=412, marker_radius=12.0, stroke_width=2, presentation={"font_size_px": 18}),
    CaseSpec("ab", "hand_drawn", 1, 24, canvas_width=6352, panel_height=520, marker_radius=6.2, stroke_width=3, presentation={"font_size_px": 18}),
    CaseSpec("abab", "hand_drawn", 1, 24, canvas_width=600, panel_height=4404, marker_radius=5.0, stroke_width=2, presentation={"font_size_px": 18}),
)


def _split_specs(split: str) -> tuple[CaseSpec, ...]:
    if split == "train":
        return _TRAIN_SPECS
    if split == "dev":
        return _DEV_SPECS
    raise ValueError(f"V32 exposes only train and dev, not {split}")


def build_split(split: str) -> tuple[SceneSample, ...]:
    seed = TRAIN_SEED if split == "train" else DEV_SEED
    built = _build_scenes(_split_specs(split), seed, require_complete_style_catalog=False)
    result: list[SceneSample] = []
    for scene in built:
        image, annotation, _ = render_scene(scene)
        truths: list[Box] = []
        for panel in annotation.get("panels", []):
            for text in panel.get("texts", []):
                if text.get("visible", True):
                    left, top, width, height = (float(value) for value in text["rendered_pixel_box"])
                    truths.append(Box(left, top, left + width, top + height))
        raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        families = scene["families"]
        result.append(SceneSample(
            str(scene["scene_id"]), split,
            str(families["renderer"]["key"]), str(families["font"]["key"]),
            str(families["degradation"]["key"]), str(families["template"]["key"]),
            str(families["marker"]["key"]), raster, tuple(truths),
        ))
    if len(result) != 5:
        raise RuntimeError(f"V32 {split} split did not produce five scenes")
    return tuple(result)


def proposal_examples(split: str = "train") -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    digest = sha256()
    scenes = build_split(split)
    proposal_count = 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        proposal_count += len(candidates)
        candidate_labels = [int(any(box_iou(candidate.box, truth) >= 0.5 for truth in scene.truths)) for candidate in candidates]
        indices = [index for index, label in enumerate(candidate_labels) if label]
        negatives = [index for index, label in enumerate(candidate_labels) if not label]
        indices.extend(negatives[:NEGATIVE_CAP_PER_SCENE])
        for index in sorted(set(indices)):
            encoded = encode_proposal(scene.raster, candidates[index]).astype(np.float32)
            values.append(encoded)
            labels.append(candidate_labels[index])
            digest.update(encoded.tobytes(order="C"))
            digest.update(bytes((candidate_labels[index],)))
    if not values or not any(labels) or all(labels):
        raise RuntimeError(f"V32 {split} proposal split lacks both classes")
    encoded = np.stack(values).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    return encoded, label_array, {
        "split": split,
        "scene_count": len(scenes),
        "proposal_count_before_cap": proposal_count,
        "proposal_count": len(labels),
        "positive_proposal_count": int(label_array.sum()),
        "negative_proposal_count": int(len(labels) - label_array.sum()),
        "tensor_label_stream_sha256": digest.hexdigest(),
        "truth_match_iou_minimum": 0.5,
        "validation_or_public_pixels_used": False,
        "sealed_public_archive_opened": False,
    }


def split_fingerprint(split: str) -> str:
    digest = sha256()
    for scene in build_split(split):
        digest.update(scene.scene_id.encode())
        for family in (
            scene.renderer_family,
            scene.font_family,
            scene.degradation_family,
            scene.template_family,
            scene.marker_family,
        ):
            digest.update(family.encode())
        digest.update(scene.raster.tobytes(order="C"))
        for truth in scene.truths:
            digest.update(f"{truth.left},{truth.top},{truth.right},{truth.bottom}\n".encode())
    return digest.hexdigest()


__all__ = ["SceneSample", "build_split", "proposal_examples", "split_fingerprint"]
