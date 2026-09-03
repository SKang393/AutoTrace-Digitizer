# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate-only synthetic negative-proposal distribution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ml.markers.center.mask_preserving_v24.mask_preserving import extract_proposals

from .generator import PATCH, TOPOLOGY_TARGETS, _quantiles, build_split
from .negative_sampler import CONNECTOR_ANCHOR_MAX_DISTANCE_PX, CONNECTOR_ENDPOINT_OFFSET_PX, TOPOLOGY_SAMPLER_RADIUS_PX, sample_negatives

REAL_NEGATIVE_GATES = {
    "ink_max_minimum": 0.11372548341751099,
    "ink_max_p05": 0.1921568512916565,
    "ocr_mean_p95": 0.7272727272727273,
}

MORPHOLOGY_KEYS = (
    "dark_fraction_ge_012", "dark_fraction_ge_05", "center5x5_mean",
    "max_row_dark_fraction_ge_012", "max_col_dark_fraction_ge_012",
    "foreground_extent_balance", "covariance_eigen_ratio",
    "border_dark_fraction_ge_012", "max_ring_support_3_12",
)


def _central_quantiles(values: list[float]) -> dict[str, float]:
    ordered = np.asarray(values, dtype=np.float64)
    q25, q50, q75 = np.quantile(ordered, (0.25, 0.50, 0.75))
    return {"p25": float(q25), "median": float(q50), "p75": float(q75)}


def _patch_morphology(patch: torch.Tensor) -> dict[str, float]:
    ink = patch[0].detach().cpu().numpy().astype("float64", copy=False)
    dark012, dark05 = ink >= .12, ink >= .5
    ys, xs = np.where(dark012)
    if len(xs):
        width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        extent = min(width, height) / max(width, height)
    else:
        extent = 0.0
    if len(xs) >= 2:
        eigen = np.linalg.eigvalsh(np.cov(np.stack((xs, ys)), bias=True))
        ratio = float(np.clip(eigen[1] / max(eigen[0], 1e-12), 1.0, 1e6))
    else:
        ratio = 1.0
    border = np.concatenate((dark012[0], dark012[-1], dark012[1:-1, 0], dark012[1:-1, -1]))
    ring = []
    for radius in range(3, 13):
        points = tuple((int(round(16 + radius * math.cos(i * math.pi / 4))),
                        int(round(16 + radius * math.sin(i * math.pi / 4)))) for i in range(8))
        ring.append(sum(0 <= x < 33 and 0 <= y < 33 and ink[y, x] >= .12 for x, y in points))
    return {
        "dark_fraction_ge_012": float(dark012.mean()),
        "dark_fraction_ge_05": float(dark05.mean()),
        "center5x5_mean": float(ink[14:19, 14:19].mean()),
        "max_row_dark_fraction_ge_012": float(dark012.mean(1).max()),
        "max_col_dark_fraction_ge_012": float(dark012.mean(0).max()),
        "foreground_extent_balance": float(extent),
        "covariance_eigen_ratio": ratio,
        "border_dark_fraction_ge_012": float(border.mean()),
        "max_ring_support_3_12": float(max(ring, default=0)),
    }


def _topology_proposal_distribution(split: str) -> dict[str, object]:
    """Measure only proposals near declared synthetic topology primitives."""
    values: dict[str, dict[str, list[float]]] = {
        "topology_junction": {key: [] for key in MORPHOLOGY_KEYS},
        "topology_fragment": {key: [] for key in MORPHOLOGY_KEYS},
    }
    counts = {kind: 0 for kind in values}
    for scene in build_split(split):
        batch = extract_proposals(scene.tensor)
        centers = torch.tensor(scene.centers, dtype=batch.coordinates.dtype)
        positive = torch.cdist(batch.coordinates, centers).min(dim=1).values <= 3.0
        for kind, x, y in scene.hard_negatives:
            if kind not in values:
                continue
            distance = torch.linalg.vector_norm(batch.coordinates - torch.tensor((x, y), dtype=batch.coordinates.dtype), dim=1)
            selected = (distance <= 16.0) & ~positive
            for patch in batch.patches[selected]:
                features = _patch_morphology(patch)
                for key in MORPHOLOGY_KEYS:
                    values[kind][key].append(features[key])
                    counts[kind] += int(key == MORPHOLOGY_KEYS[0])
    return {
        kind: {key: _quantiles(items) for key, items in features.items()}
        for kind, features in values.items()
    } | {"proposal_counts": counts}


def _positive_proposal_distribution(split: str) -> dict[str, object]:
    """Measure positive proposal morphology with the V24 five-pixel label rule."""
    values = {key: [] for key in MORPHOLOGY_KEYS}
    count = 0
    for scene in build_split(split):
        batch = extract_proposals(scene.tensor)
        centers = torch.tensor(scene.centers, dtype=batch.coordinates.dtype)
        positive = torch.cdist(batch.coordinates, centers).min(dim=1).values <= 5.0
        for patch in batch.patches[positive]:
            features = _patch_morphology(patch)
            for key in MORPHOLOGY_KEYS:
                values[key].append(features[key])
            count += 1
    return {"label_distance_px": 5.0, "proposal_count": count,
            "morphology_quantiles": {key: _quantiles(items) for key, items in values.items()},
            "central_morphology_quantiles": {key: _central_quantiles(items) for key, items in values.items()}}


def _topology_gates(topology: dict[str, object]) -> dict[str, bool]:
    junction, fragment = topology["topology_junction"], topology["topology_fragment"]
    above = TOPOLOGY_TARGETS["negative_above_025"]
    below = TOPOLOGY_TARGETS["negative_below_025"]
    # Each real-dev morphology median must lie inside the synthetic topology
    # p05..p95 envelope. This is an input-coverage audit,
    # never a candidate or threshold-selection rule.
    def covered(record: dict[str, object], key: str, target: float) -> bool:
        quantiles = record[key]
        return quantiles["p05"] <= target <= quantiles["p95"]
    return {
        "junction_dark05_covers_real_median": covered(junction, "dark_fraction_ge_05", above["dark_fraction_ge_05_median"]),
        "junction_center5_covers_real_median": covered(junction, "center5x5_mean", above["center5x5_mean_median"]),
        "junction_row_covers_real_median": covered(junction, "max_row_dark_fraction_ge_012", above["max_row_dark_fraction_ge_012_median"]),
        "junction_col_covers_real_median": covered(junction, "max_col_dark_fraction_ge_012", above["max_col_dark_fraction_ge_012_median"]),
        "junction_extent_covers_real_median": covered(junction, "foreground_extent_balance", above["foreground_extent_balance_median"]),
        "junction_covariance_covers_real_median": covered(junction, "covariance_eigen_ratio", above["covariance_eigen_ratio_median"]),
        "junction_border_covers_real_median": covered(junction, "border_dark_fraction_ge_012", above["border_dark_fraction_ge_012_median"]),
        "junction_ring_covers_real_median": covered(junction, "max_ring_support_3_12", above["max_ring_support_3_12_median"]),
        "fragment_row_covers_real_median": covered(fragment, "max_row_dark_fraction_ge_012", below["max_row_dark_fraction_ge_012_median"]),
        "fragment_col_covers_real_median": covered(fragment, "max_col_dark_fraction_ge_012", below["max_col_dark_fraction_ge_012_median"]),
        "fragment_extent_covers_real_median": covered(fragment, "foreground_extent_balance", below["foreground_extent_balance_median"]),
        "fragment_covariance_covers_real_median": covered(fragment, "covariance_eigen_ratio", below["covariance_eigen_ratio_median"]),
        "fragment_border_covers_real_median": covered(fragment, "border_dark_fraction_ge_012", below["border_dark_fraction_ge_012_median"]),
    }


def _positive_gates(positives: dict[str, object]) -> dict[str, bool]:
    values = positives["central_morphology_quantiles"]
    targets = TOPOLOGY_TARGETS["positives"]
    def covered(field: str, target_name: str) -> bool:
        quantiles = values[field]
        target = targets[target_name]
        return quantiles["p25"] <= target <= quantiles["p75"]
    return {
        "center5_covers_real_median": covered("center5x5_mean", "center5x5_mean_median"),
        "row_covers_real_median": covered("max_row_dark_fraction_ge_012", "max_row_dark_fraction_ge_012_median"),
        "col_covers_real_median": covered("max_col_dark_fraction_ge_012", "max_col_dark_fraction_ge_012_median"),
        "extent_covers_real_median": covered("foreground_extent_balance", "foreground_extent_balance_median"),
        "covariance_covers_real_median": covered("covariance_eigen_ratio", "covariance_eigen_ratio_median"),
        "ring_covers_real_median": covered("max_ring_support_3_12", "max_ring_support_3_12_median"),
    }


def _train_sampler_records() -> list[tuple[object, object, torch.Tensor, torch.Tensor]]:
    records = []
    for scene in build_split("train"):
        batch = extract_proposals(scene.tensor)
        centers = torch.tensor(scene.centers, dtype=batch.coordinates.dtype)
        labels = torch.cdist(batch.coordinates, centers).min(dim=1).values.le(3.0).float()
        hard = torch.zeros(len(batch.coordinates), dtype=torch.bool)
        for kind, x, y in scene.hard_negatives:
            if kind in {"text", "line_intersection", "axis"}:
                hard |= torch.cdist(batch.coordinates, torch.tensor(((x, y),), dtype=batch.coordinates.dtype)).squeeze(1).le(8.0)
        records.append((scene, batch, labels, hard))
    return records


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
    topology_train = _topology_proposal_distribution("train")
    topology_dev = _topology_proposal_distribution("dev")
    positive_train = _positive_proposal_distribution("train")
    positive_dev = _positive_proposal_distribution("dev")
    sampled = sample_negatives(_train_sampler_records(), split="train", seed=20260904)
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
        "hard_negative_kinds": ["text", "line_intersection", "axis", "faint_line", "ocr_heavy", "topology_junction", "topology_fragment"],
        "hard_negative_representatives": {
            "faint_line": {"x": 32.0, "y": 155.0},
            "ocr_heavy": {"x": 208.0, "y": 155.0},
        },
        "topology": {
            "target_real_dev_medians": TOPOLOGY_TARGETS,
            "proposals": {"train": topology_train, "dev": topology_dev},
        },
        "topology_distribution_gates": _topology_gates(topology_dev),
        "positive_morphology": {"train": positive_train, "dev": positive_dev},
        "positive_morphology_gates": {
            "train": _positive_gates(positive_train),
            "dev": _positive_gates(positive_dev),
        },
        "positive_morphology_gate_split": "train_and_dev",
        "sampler": {
            "split": "train",
            "seed": 20260904,
            "negative_total": sampled.total,
            "topology_radius_px": 16.0,
            "topology_sampler_radius_px": sampled.topology_sampler_radius_px,
            "topology_capacity": sampled.topology_capacity,
            "topology_selected": sampled.topology_selected,
            "topology_all_eligible_retained": sampled.topology_capacity == sampled.topology_selected,
            "selected_index_sha256": sampled.selected_index_sha256,
            "topology_selected_index_sha256": sampled.topology_selected_index_sha256,
            "connector_endpoint_offset_px": CONNECTOR_ENDPOINT_OFFSET_PX,
            "connector_anchor_max_distance_px": CONNECTOR_ANCHOR_MAX_DISTANCE_PX,
            "connector_anchor_target_count": sampled.connector_anchor_target_count,
            "connector_anchor_capacity": sampled.connector_anchor_capacity,
            "connector_anchor_selected": sampled.connector_anchor_selected,
            "connector_anchor_all_eligible_retained": sampled.connector_anchor_capacity == sampled.connector_anchor_selected,
            "connector_anchor_selected_index_sha256": sampled.connector_anchor_selected_index_sha256,
            "generic_remainder_selected": sampled.generic_remainder_selected,
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
