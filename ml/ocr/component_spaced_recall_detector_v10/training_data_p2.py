# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic V10 P2 training-only proposal selection."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals

from .dataset import build_split, encode_proposal
from .protocol import TRUTH_MATCH_IOU_MINIMUM


NEGATIVE_CAP_PER_SCENE = 32


def training_examples() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    digest = sha256()
    scenes = build_split("train")
    for scene in scenes:
        candidates = proposals(scene.raster)
        scene_labels = np.asarray(
            [
                int(
                    any(
                        box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM
                        for truth in scene.truths
                    )
                )
                for candidate in candidates
            ],
            dtype=np.int64,
        )
        positive = [index for index, value in enumerate(scene_labels) if value == 1]
        negative = [index for index, value in enumerate(scene_labels) if value == 0][:NEGATIVE_CAP_PER_SCENE]
        for index in sorted((*positive, *negative)):
            encoded = encode_proposal(scene.raster, candidates[index])
            values.append(encoded)
            labels.append(int(scene_labels[index]))
            digest.update(encoded.tobytes(order="C"))
            digest.update(bytes((int(scene_labels[index]),)))
    result_values = np.stack(values).astype(np.float32)
    result_labels = np.asarray(labels, dtype=np.int64)
    return result_values, result_labels, {
        "scene_count": len(scenes),
        "negative_cap_per_scene": NEGATIVE_CAP_PER_SCENE,
        "proposal_count": len(result_labels),
        "positive_proposal_count": int(result_labels.sum()),
        "negative_proposal_count": int(len(result_labels) - result_labels.sum()),
        "tensor_label_stream_sha256": digest.hexdigest(),
        "validation_or_public_pixels_used": False,
    }


__all__ = ["NEGATIVE_CAP_PER_SCENE", "training_examples"]
