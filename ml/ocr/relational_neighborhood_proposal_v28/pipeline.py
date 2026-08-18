# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Checksum-bound proposal, crop, and relation inputs for OCR V28."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ml.ocr.crop_evidence_role_anchor_v24.pipeline import extract_crop_evidence

from .dataset import SceneSample
from .relations import scene_relation_stream


Runner = Callable[[np.ndarray], np.ndarray]


def extract_relational_evidence(
    scenes: tuple[SceneSample, ...],
    detector_runner: Runner,
    recognizer_runner: Runner,
    alphabet: str,
    *,
    mode: str,
    negative_cap_per_scene: int = 4,
    recognition_batch_size: int = 64,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[object, ...],
    tuple[np.ndarray, ...],
    tuple[slice, ...],
    dict[str, object],
]:
    values, crops, labels, records, direct = extract_crop_evidence(
        scenes,
        detector_runner,
        recognizer_runner,
        alphabet,
        mode=mode,
        negative_cap_per_scene=negative_cap_per_scene,
        recognition_batch_size=recognition_batch_size,
    )
    relations, scene_slices, relation_hash = scene_relation_stream(scenes, records)
    direct.update({
        "proposal_relation_tensor_stream_sha256": relation_hash,
        "proposal_relation_scene_shapes": [list(item.shape) for item in relations],
        "proposal_relation_truth_independent": True,
    })
    return values, crops, labels, records, relations, scene_slices, direct


__all__ = ["extract_relational_evidence"]
