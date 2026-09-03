# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate-only synthetic negative-proposal distribution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ml.markers.center.mask_preserving_v24.mask_preserving import extract_proposals

from .generator import PATCH, _quantiles, build_split

REAL_NEGATIVE_GATES = {
    "ink_max_minimum": 0.11372548341751099,
    "ink_max_p05": 0.1921568512916565,
    "ocr_mean_p95": 0.7272727272727273,
}


def _proposal_distribution(split: str) -> dict[str, object]:
    feature_names = (
        "ink_mean", "ink_center_5x5_mean", "ink_max",
        "ocr_mean", "ocr_max", "artifact_mean", "artifact_max",
    )
    negatives: dict[str, list[float]] = {name: [] for name in feature_names}
    proposal_count = positive_count = 0
    digest = hashlib.sha256()
    for scene in build_split(split):
        batch = extract_proposals(scene.tensor)
        proposal_count += len(batch.coordinates)
        digest.update(batch.coordinates.numpy().tobytes())
        centers = torch.tensor(scene.centers, dtype=batch.coordinates.dtype)
        positive = torch.cdist(batch.coordinates, centers).min(dim=1).values <= 3.0
        positive_count += int(positive.sum())
        patch = batch.patches[~positive]
        middle = PATCH // 2
        values = {
            "ink_mean": patch[:, 0].mean(dim=(1, 2)),
            "ink_center_5x5_mean": patch[:, 0, middle - 2:middle + 3, middle - 2:middle + 3].mean(dim=(1, 2)),
            "ink_max": patch[:, 0].amax(dim=(1, 2)),
            "ocr_mean": patch[:, 1].mean(dim=(1, 2)),
            "ocr_max": patch[:, 1].amax(dim=(1, 2)),
            "artifact_mean": patch[:, 2].mean(dim=(1, 2)),
            "artifact_max": patch[:, 2].amax(dim=(1, 2)),
        }
        for name, value in values.items():
            negatives[name].extend(float(item) for item in value.tolist())
    negative_count = proposal_count - positive_count
    return {
        "proposal_count": proposal_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rule": "proposal center within <=3 px of a truth center",
        "sampled_negative_count_max10_per_positive": min(negative_count, positive_count * 10),
        "negative_patch_feature_quantiles": {
            name: _quantiles(values) for name, values in negatives.items()
        },
        "proposal_coordinates_aggregate_sha256": digest.hexdigest(),
    }


def _distribution_gates(record: dict[str, object]) -> dict[str, bool]:
    values = record["negative_patch_feature_quantiles"]
    return {
        "ink_max_minimum_reaches_real_dev": values["ink_max"]["minimum"] <= REAL_NEGATIVE_GATES["ink_max_minimum"],
        "ink_max_p05_reaches_real_dev": values["ink_max"]["p05"] <= REAL_NEGATIVE_GATES["ink_max_p05"],
        "ocr_mean_p95_reaches_real_dev": values["ocr_mean"]["p95"] >= REAL_NEGATIVE_GATES["ocr_mean_p95"],
    }


def audit() -> dict[str, object]:
    train = _proposal_distribution("train")
    dev = _proposal_distribution("dev")
    return {
        "schema": "graphreader.marker-center-negative-proposal-audit.v1",
        "scope": {
            "synthetic_only": True,
            "aggregate_only": True,
            "model_loaded": False,
            "training_performed": False,
            "private_or_article_images": False,
            "scene_ids_emitted": False,
            "truth_rows_emitted": False,
            "pixels_emitted": False,
        },
        "extractor": "mask-preserving-v24 ink-supported full grid",
        "splits": {"train": train, "dev": dev},
        "distribution_gates": {
            split: _distribution_gates(record) for split, record in (("train", train), ("dev", dev))
        },
        "hard_negative_kinds": ["text", "line_intersection", "axis", "faint_line", "ocr_heavy"],
        "hard_negative_representatives": {
            "faint_line": {"x": 32.0, "y": 155.0},
            "ocr_heavy": {"x": 208.0, "y": 155.0},
        },
        "coordinate_streams_identical": (
            train["proposal_coordinates_aggregate_sha256"]
            == dev["proposal_coordinates_aggregate_sha256"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoded = (json.dumps(audit(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(encoded.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
