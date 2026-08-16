# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Checksum-bound evidence and crop extraction for OCR V24."""

from __future__ import annotations

from hashlib import sha256
from typing import Callable

import numpy as np

from ml.ocr.margin_calibrator_v20.pipeline import ProposalRecord, extract_features
from .dataset import SceneSample, encode_scene
from .protocol import CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH


Runner = Callable[[np.ndarray], np.ndarray]


def proposal_crops(
    scenes: tuple[SceneSample, ...], records: tuple[ProposalRecord, ...],
) -> np.ndarray:
    """Select the exact production tight/context crop for each evidence record."""
    encoded_scenes: list[np.ndarray] = []
    for scene in scenes:
        encoded, _, _, _ = encode_scene(scene)
        if encoded.ndim != 4 or encoded.shape[1:3] != (CROP_CHANNELS, CROP_HEIGHT):
            raise RuntimeError("OCR V24 production crop tensor contract changed")
        if encoded.shape[3] < CROP_WIDTH:
            raise RuntimeError("OCR V24 production crop encoding is too narrow")
        encoded_scenes.append(np.ascontiguousarray(encoded[:, :, :, :CROP_WIDTH]))
    selected: list[np.ndarray] = []
    for record in records:
        if record.scene_index < 0 or record.scene_index >= len(encoded_scenes):
            raise RuntimeError("OCR V24 proposal record scene index is invalid")
        scene_values = encoded_scenes[record.scene_index]
        if record.candidate_index < 0 or record.candidate_index >= len(scene_values):
            raise RuntimeError("OCR V24 proposal record candidate index is invalid")
        selected.append(scene_values[record.candidate_index])
    if not selected:
        raise RuntimeError("OCR V24 proposal crop stream is empty")
    result = np.stack(selected).astype(np.float32)
    if result.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
        raise RuntimeError("OCR V24 selected crop tensor shape changed")
    if not np.isfinite(result).all():
        raise RuntimeError("OCR V24 selected crop tensor is nonfinite")
    return result


def extract_crop_evidence(
    scenes: tuple[SceneSample, ...], detector_runner: Runner,
    recognizer_runner: Runner, alphabet: str, *, mode: str,
    negative_cap_per_scene: int = 4, recognition_batch_size: int = 64,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[ProposalRecord, ...], dict[str, object],
]:
    evidence_values, labels, records, direct = extract_features(
        scenes,
        detector_runner,
        recognizer_runner,
        alphabet,
        mode=mode,  # type: ignore[arg-type]
        negative_cap_per_scene=negative_cap_per_scene,
        recognition_batch_size=recognition_batch_size,
    )
    crops = proposal_crops(scenes, records)
    if len(crops) != len(evidence_values):
        raise RuntimeError("OCR V24 evidence and crop proposal streams diverged")
    crop_stream = sha256(np.ascontiguousarray(crops).tobytes(order="C")).hexdigest()
    direct.update({
        "proposal_crop_tensor_stream_sha256": crop_stream,
        "proposal_crop_tensor_shape": list(crops.shape),
        "proposal_crop_channels": ["tight", "context"],
    })
    return evidence_values, crops, labels, records, direct


__all__ = ["extract_crop_evidence", "proposal_crops"]
