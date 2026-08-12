# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Training-only scale-degraded hard negatives for V9 candidate P2."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
from PIL import Image

from .dataset import SceneSample, encode_proposal, proposal_examples, proposal_labels, proposals


P2_AUGMENTED_NEGATIVE_CAP_PER_SCENE = 16


def _disjoint_from_truths(candidate: object, scene: SceneSample) -> bool:
    box = candidate.box
    return all(
        box.right <= truth.left
        or box.left >= truth.right
        or box.bottom <= truth.top
        or box.top >= truth.bottom
        for truth in scene.truths
    )


def _scale_degraded_training_scene(scene: SceneSample) -> SceneSample:
    if scene.split != "train":
        raise ValueError("OCR V9 P2 augmentation accepts training scenes only")
    image = Image.fromarray(scene.raster, mode="L")
    reduced = image.resize((624, 312), resample=Image.Resampling.BOX)
    restored = reduced.resize((640, 320), resample=Image.Resampling.BICUBIC)
    raster = np.asarray(restored, dtype=np.uint8).copy()
    row = 35 + (int.from_bytes(sha256(scene.scene_id.encode()).digest()[:4], "little") % 248)
    raster[max(0, row - 1) : min(320, row + 2)] = np.maximum(
        raster[max(0, row - 1) : min(320, row + 2)], 170
    )
    return SceneSample(
        f"{scene.scene_id}-p2-scale-negative",
        "train",
        scene.renderer_family,
        "training-only-box-bicubic-scale-row-fade-v9-p2",
        raster,
        scene.truths,
    )


def p2_proposal_examples(scenes: tuple[SceneSample, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values, labels = proposal_examples(scenes)
    augmented: list[np.ndarray] = []
    per_scene_counts: list[int] = []
    for original in scenes:
        scene = _scale_degraded_training_scene(original)
        candidates = proposals(scene.raster)
        candidate_labels = proposal_labels(scene, candidates)
        negative_indices = [
            index for index, value in enumerate(candidate_labels)
            if value == 0 and _disjoint_from_truths(candidates[index], scene)
        ]
        selected = negative_indices[:P2_AUGMENTED_NEGATIVE_CAP_PER_SCENE]
        augmented.extend(encode_proposal(scene.raster, candidates[index]) for index in selected)
        per_scene_counts.append(len(selected))
    if not augmented:
        raise RuntimeError("OCR V9 P2 scale degradation produced no hard-negative proposals")
    augmented_values = np.stack(augmented).astype(np.float32)
    combined_values = np.concatenate((values, augmented_values), axis=0)
    combined_labels = np.concatenate((labels, np.zeros(len(augmented_values), dtype=np.int64)))
    evidence = {
        "augmentation": "training-only-box-bicubic-scale-row-fade-v9-p2",
        "scene_count": len(scenes),
        "negative_cap_per_scene": P2_AUGMENTED_NEGATIVE_CAP_PER_SCENE,
        "augmented_negative_count": len(augmented_values),
        "minimum_augmented_negatives_per_scene": min(per_scene_counts),
        "maximum_augmented_negatives_per_scene": max(per_scene_counts),
        "augmented_tensor_stream_sha256": sha256(augmented_values.tobytes(order="C")).hexdigest(),
        "truth_overlap_allowed": False,
        "validation_or_public_pixels_used": False,
    }
    return combined_values, combined_labels, evidence


__all__ = ["P2_AUGMENTED_NEGATIVE_CAP_PER_SCENE", "p2_proposal_examples"]
