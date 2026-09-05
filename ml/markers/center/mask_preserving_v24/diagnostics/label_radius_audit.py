# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Compare diagnostic label radii on one frozen synthetic proposal stream."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.center.real_range_generator_v1.generator import _hash_scenes, build_split
from .diagnose_retry import RETRY9_DEV_SPLIT_SHA256, RETRY9_ONNX_SHA256, THRESHOLD, _sha
from ..mask_preserving import extract_proposals


def radius_counts(distances: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, int]]:
    if distances.ndim != 1 or distances.shape != scores.shape:
        raise ValueError("distance and score vectors must have the same shape")
    if not np.isfinite(distances).all() or not np.isfinite(scores).all() or (distances < 0).any():
        raise ValueError("distances and scores must be finite, with nonnegative distances")
    above = scores >= THRESHOLD
    return {
        str(radius): {
            "positive": int((distances <= radius).sum()),
            "negative_above_threshold": int(((distances > radius) & above).sum()),
            "negative_below_threshold": int(((distances > radius) & ~above).sum()),
        }
        for radius in (3, 5)
    }


def run(model: Path) -> dict:
    started = time.perf_counter()
    if _sha(model) != RETRY9_ONNX_SHA256:
        raise ValueError("retry9 ONNX checksum mismatch")
    scenes = build_split("dev")
    scene_hash = _hash_scenes(scenes)
    if scene_hash != RETRY9_DEV_SPLIT_SHA256:
        raise ValueError("historical retry9 synthetic dev stream changed")
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("CPU execution provider was not selected")
    totals = {str(radius): dict(positive=0, negative_above_threshold=0,
                               negative_below_threshold=0) for radius in (3, 5)}
    for scene in scenes:
        proposals = extract_proposals(scene.tensor)
        output = session.run(None, {session.get_inputs()[0].name: proposals.patches.numpy()})[0]
        if output.shape != (len(proposals.coordinates), 4):
            raise ValueError("model output contract changed")
        distances = torch.cdist(proposals.coordinates, torch.tensor(scene.centers,
                                dtype=proposals.coordinates.dtype)).amin(dim=1).numpy()
        for radius, counts in radius_counts(distances, output[:, 0]).items():
            for key, count in counts.items():
                totals[radius][key] += count
    historical = dict(positive=3258, negative_above_threshold=1512,
                      negative_below_threshold=231275)
    if totals["3"] != historical:
        raise RuntimeError("3-pixel control does not reproduce the recorded retry9 counts")
    for counts in totals.values():
        denominator = counts["negative_above_threshold"] + counts["negative_below_threshold"]
        counts["negative_above_threshold_rate"] = counts["negative_above_threshold"] / denominator
    return {
        "schema": "graphreader.marker-v24-retry9-label-radius-audit.v1",
        "scope": {"synthetic_only": True, "scene_count": len(scenes),
                  "optimizer_steps": 0, "private_reads": 0, "sealed_reads": 0,
                  "case_ids_or_pixels_emitted": False, "model_or_threshold_selection": False},
        "binding": {"model_sha256": RETRY9_ONNX_SHA256, "dev_sha256": scene_hash,
                    "provider": session.get_providers()[0], "threshold": THRESHOLD,
                    "diagnostic_source_sha256": _sha(Path(__file__))},
        "counts_by_radius_px": totals,
        "historical_3px_control_reproduced": True,
        "proposals_reclassified_as_positive": totals["5"]["positive"] - totals["3"]["positive"],
        "above_threshold_proposals_reclassified": totals["3"]["negative_above_threshold"] - totals["5"]["negative_above_threshold"],
        "limitation": "Isolates label-radius effects on this synthetic stream only; does not establish the cause of real-image failure.",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(args.model)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
