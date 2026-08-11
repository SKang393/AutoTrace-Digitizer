# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Truth-hidden renderer and archive writer for the V1 sealed public split."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from .dataset import EvaluationFrame, _render_evaluation_frame, split_fingerprint
from .protocol import SEALED_EXCLUSION_COUNT, SEALED_TEXT_COUNT


def build_sealed_public_split() -> tuple[EvaluationFrame, ...]:
    count = SEALED_TEXT_COUNT + SEALED_EXCLUSION_COUNT
    return tuple(_render_evaluation_frame("sealed_public", index) for index in range(count))


def save_sealed_public_archive(samples: tuple[EvaluationFrame, ...], path: Path) -> dict[str, object]:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite sealed graph-text fixtures: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    case_ids = np.asarray([sample.case_id for sample in samples])
    kinds = np.asarray([sample.kind for sample in samples])
    source_sha256 = np.asarray([sample.source_sha256 for sample in samples])
    detector_sha256 = np.asarray([sample.detector_bgr_sha256 for sample in samples])
    truth = np.asarray(
        [sample.truth_bbox if sample.truth_bbox is not None else (-1.0, -1.0, -1.0, -1.0) for sample in samples],
        dtype=np.float32,
    )
    detector_bgr = np.stack(
        [np.frombuffer(sample.detector_bgr, dtype=np.uint8) for sample in samples]
    )
    np.savez_compressed(
        path,
        case_ids=case_ids,
        kinds=kinds,
        source_sha256=source_sha256,
        detector_sha256=detector_sha256,
        truth_bbox=truth,
        detector_bgr=detector_bgr,
    )
    return {
        "schema": "graphreader.ocr-graph-text-detector-private-sealed-manifest.v1",
        "sample_count": len(samples),
        "text_count": sum(sample.kind == "text" for sample in samples),
        "exclusion_count": sum(sample.kind == "exclusion" for sample in samples),
        "split_fingerprint": split_fingerprint(samples),
        "archive_sha256": sha256(path.read_bytes()).hexdigest(),
        "private_data": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "truth_hidden_from_training_runner": True,
    }


__all__ = ["build_sealed_public_split", "save_sealed_public_archive"]
