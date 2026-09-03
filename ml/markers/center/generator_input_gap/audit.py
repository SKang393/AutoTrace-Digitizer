# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Audit synthetic marker inputs without loading models or private data.

The audit intentionally consumes only the V20 train builder and frozen V13
synthetic dev builder.  It reports aggregate geometry and the fixed tensor
shape, including evidence that V20's resize degradation is restored to the
original ``224x168`` scene before proposal extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes
from ml.markers.center.tail_coverage_v20.training_families import build_train_scenes


PATCH_SIZE_PX = 33
EXPECTED_TENSOR_SHAPE = (3, 168, 224)


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty value sequence")
    n = len(ordered)

    def percentile(fraction: float) -> float:
        position = (n - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, n - 1)
        weight = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    return {
        "minimum": round(ordered[0], 6),
        "p05": round(percentile(0.05), 6),
        "p10": round(percentile(0.10), 6),
        "median": round(float(median(ordered)), 6),
        "p90": round(percentile(0.90), 6),
        "p95": round(percentile(0.95), 6),
        "maximum": round(ordered[-1], 6),
    }


def _summarize_split(name: str, scenes: tuple[Any, ...]) -> dict[str, Any]:
    radii = [float(radius) for scene in scenes for radius in scene.radii]
    diameters = [2.0 * radius for radius in radii]
    ratios = [diameter / PATCH_SIZE_PX for diameter in diameters]
    shapes = sorted({tuple(int(value) for value in scene.tensor.shape) for scene in scenes})
    return {
        "split": name,
        "scene_count": len(scenes),
        "marker_count": len(radii),
        "tensor_shapes": [list(shape) for shape in shapes],
        "marker_radius_px": _quantiles(radii),
        "marker_diameter_px": _quantiles(diameters),
        "diameter_to_33px_patch_ratio": _quantiles(ratios),
        "fixed_tensor_shape": all(shape == EXPECTED_TENSOR_SHAPE for shape in shapes),
    }


def audit() -> dict[str, Any]:
    """Return deterministic aggregate evidence from synthetic scenes only."""
    train = build_train_scenes()
    dev = build_selection_scenes("dev")
    all_scenes = train + dev
    if not all_scenes:
        raise RuntimeError("synthetic scene builders returned no scenes")
    shapes = {tuple(int(value) for value in scene.tensor.shape) for scene in all_scenes}
    return {
        "schema": "graphreader.marker-center-generator-input-gap-audit.v1",
        "scope": {
            "synthetic_only": True,
            "private_or_article_images": False,
            "model_loaded": False,
            "training_performed": False,
            "candidate_revision_created": False,
            "scene_ids_emitted": False,
            "truth_rows_emitted": False,
            "pixels_emitted": False,
        },
        "sources": {
            "train_builder": "ml/markers/center/tail_coverage_v20/training_families.py",
            "dev_builder": "ml/markers/center/proposal_geometry_v13/dataset.py",
            "proposal_patch_size_px": PATCH_SIZE_PX,
        },
        "splits": {
            "train": _summarize_split("train", train),
            "dev": _summarize_split("dev", dev),
        },
        "aggregate": {
            "scene_count": len(all_scenes),
            "marker_count": sum(len(scene.radii) for scene in all_scenes),
            "tensor_shapes": [list(shape) for shape in sorted(shapes)],
            "fixed_224x168_tensor": shapes == {EXPECTED_TENSOR_SHAPE},
            "resize_degradation_roundtrip_restored": shapes == {EXPECTED_TENSOR_SHAPE},
            "production_marker_frame_resize_observed": False,
            "evidence": "V20 _augment_scene resizes channels down then back to the source tensor dimensions; proposal inputs remain fixed 224x168",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded.encode("utf-8"))
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
