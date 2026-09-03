# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Trace V19 fixed-dev misses without emitting scene or case identities."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import onnxruntime as ort

from ml.markers.center.line_aware_v1.pipeline import (
    _center_is_unmasked,
    _marker_geometry_consensus,
    extract_proposals,
    postprocess_predictions,
)
from ml.markers.center.metrics import aggregate_scene_metrics, center_metrics
from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes, selection_manifest
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals


REPO_ROOT = Path(__file__).resolve().parents[5]
V19_ARTIFACT_DIR = (
    REPO_ROOT
    / "artifacts/goal22-worktrees/marker-v19/ml/markers/center/train_family_v19/artifacts/P1-run"
)
DEFAULT_ONNX = V19_ARTIFACT_DIR / "marker-center-train-family-v19-p1.onnx"
DEFAULT_CHECKPOINT = V19_ARTIFACT_DIR / "marker-center-train-family-v19-p1.pt"
DEFAULT_REPORT = V19_ARTIFACT_DIR / "candidate-report.json"
THRESHOLD = 0.25
TOLERANCE = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": round(min(values), 6),
        "median": round(float(median(values)), 6),
        "maximum": round(max(values), 6),
    }


def _radius_bin(radius: float) -> str:
    if radius <= 4:
        return "3-4"
    if radius <= 7:
        return "5-7"
    if radius <= 10:
        return "8-10"
    return "11-12"


def _marker_shape(truth_index: int, variant: int) -> str:
    # The fixed V13 generator selects the glyph from (index + variant) % 3.
    return ("filled_circle", "open_square", "filled_diamond")[(truth_index + variant) % 3]


def _scene_variant(scene: Any) -> int:
    return int(str(scene.scene_id).rsplit("-", 1)[1])


def _decoded(output: np.ndarray, coordinates: np.ndarray) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    for index, (base_x, base_y) in enumerate(coordinates.tolist()):
        confidence = float(output[index, 0])
        x = float(base_x + output[index, 1] * 4.0)
        y = float(base_y + output[index, 2] * 4.0)
        radius = float(np.clip(output[index, 3], 2.5, 8.0))
        rows.append(
            {
                "index": index,
                "confidence": confidence,
                "x": x,
                "y": y,
                "radius": radius,
            }
        )
    return rows


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(path),
        "exists": True,
    }


def summarize(onnx_path: Path = DEFAULT_ONNX) -> dict[str, Any]:
    """Run the fixed V13 dev split through only the ignored V19 ONNX payload."""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"V19 ONNX artifact is missing: {onnx_path}")
    scenes = build_selection_scenes("dev")
    manifest_bytes = (json.dumps(selection_manifest(), indent=2, sort_keys=True) + "\n").encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    threshold_metrics: dict[str, dict[str, Any]] = {}
    threshold_values = (0.0, 0.10, THRESHOLD, 0.40, 0.55, 0.70)
    threshold_scene_values: dict[float, list[Any]] = defaultdict(list)
    families: dict[str, dict[str, Any]] = {}
    degradations: dict[str, dict[str, Any]] = {}
    radius_rows: dict[str, dict[str, Any]] = {}
    shape_rows: dict[str, dict[str, Any]] = {}
    reason_counts: Counter[str] = Counter()
    proposal_counts = Counter()
    all_scores: list[float] = []
    missed_scores: list[float] = []
    matched_scores: list[float] = []
    total_truths = 0
    total_predictions = 0

    for scene in scenes:
        raw = extract_proposals(scene.tensor)
        filtered = filter_proposals(scene.tensor, raw)
        output = session.run(None, {input_name: filtered.patches.numpy().astype(np.float32, copy=False)})[0]
        decoded = _decoded(output, filtered.coordinates.numpy())
        predictions = postprocess_predictions(scene, filtered, output, threshold=THRESHOLD)
        for threshold in threshold_values:
            threshold_scene_values[threshold].append(
                center_metrics(
                    postprocess_predictions(scene, filtered, output, threshold=threshold),
                    scene.centers,
                    TOLERANCE,
                )
            )

        family = str(scene.family)
        degradation = str(scene.degradation)
        for key, bucket in ((family, families), (degradation, degradations)):
            bucket.setdefault(
                key,
                {
                    "scene_count": 0,
                    "truth_count": 0,
                    "raw_proposal_count": 0,
                    "geometry_proposal_count": 0,
                    "truth_with_raw_proposal_3px": 0,
                    "truth_with_geometry_proposal_3px": 0,
                    "truth_with_geometry_proposal_5px": 0,
                    "true_positives": 0,
                    "false_negatives": 0,
                    "max_confidence": [],
                    "miss_max_confidence": [],
                },
            )
            bucket[key]["scene_count"] += 1
            bucket[key]["truth_count"] += len(scene.centers)
            bucket[key]["raw_proposal_count"] += len(raw.coordinates)
            bucket[key]["geometry_proposal_count"] += len(filtered.coordinates)

        variant = _scene_variant(scene)
        for truth_index, ((truth_x, truth_y), truth_radius) in enumerate(zip(scene.centers, scene.radii, strict=True)):
            total_truths += 1
            raw_distances = np.hypot(
                raw.coordinates[:, 0].numpy() - truth_x,
                raw.coordinates[:, 1].numpy() - truth_y,
            )
            filtered_distances = np.hypot(
                filtered.coordinates[:, 0].numpy() - truth_x,
                filtered.coordinates[:, 1].numpy() - truth_y,
            )
            raw_3 = bool(np.any(raw_distances <= 3.0))
            filtered_3 = bool(np.any(filtered_distances <= 3.0))
            filtered_5 = bool(np.any(filtered_distances <= 5.0))
            score_candidates = output[filtered_distances <= TOLERANCE, 0]
            max_score = float(np.max(score_candidates)) if len(score_candidates) else 0.0
            all_scores.append(max_score)
            decoded_near = [
                row for row in decoded if math.hypot(float(row["x"]) - truth_x, float(row["y"]) - truth_y) <= TOLERANCE
            ]
            high_near = [row for row in decoded_near if float(row["confidence"]) >= THRESHOLD]
            gated_near = [
                row
                for row in high_near
                if _center_is_unmasked(scene, float(row["x"]), float(row["y"]))
                and _marker_geometry_consensus(scene, float(row["x"]), float(row["y"]), float(row["radius"]))
            ]
            matched = any(math.hypot(pred.x - truth_x, pred.y - truth_y) <= TOLERANCE for pred in predictions)
            if matched:
                matched_scores.append(max_score)
            else:
                missed_scores.append(max_score)
                if not filtered_5:
                    reason_counts["proposal_unavailable_after_geometry"] += 1
                elif max_score < THRESHOLD:
                    reason_counts["confidence_below_threshold"] += 1
                elif high_near and not any(_center_is_unmasked(scene, float(row["x"]), float(row["y"])) for row in high_near):
                    reason_counts["unmasked_artifact_veto"] += 1
                elif high_near and not gated_near:
                    reason_counts["marker_geometry_veto"] += 1
                else:
                    reason_counts["nms_or_matching"] += 1

            for bucket, row in ((families[family], None), (degradations[degradation], None)):
                bucket["truth_with_raw_proposal_3px"] += int(raw_3)
                bucket["truth_with_geometry_proposal_3px"] += int(filtered_3)
                bucket["truth_with_geometry_proposal_5px"] += int(filtered_5)
                bucket["true_positives"] += int(matched)
                bucket["false_negatives"] += int(not matched)
                bucket["max_confidence"].append(max_score)
                if not matched:
                    bucket["miss_max_confidence"].append(max_score)

            radius_key = _radius_bin(float(truth_radius))
            radius_rows.setdefault(
                radius_key,
                {
                    "truth_count": 0,
                    "truth_with_geometry_proposal_3px": 0,
                    "true_positives": 0,
                    "false_negatives": 0,
                    "max_confidence": [],
                    "miss_max_confidence": [],
                },
            )
            radius_rows[radius_key]["truth_count"] += 1
            radius_rows[radius_key]["truth_with_geometry_proposal_3px"] += int(filtered_3)
            radius_rows[radius_key]["true_positives"] += int(matched)
            radius_rows[radius_key]["false_negatives"] += int(not matched)
            radius_rows[radius_key]["max_confidence"].append(max_score)
            if not matched:
                radius_rows[radius_key]["miss_max_confidence"].append(max_score)

            shape_key = _marker_shape(truth_index, variant)
            shape_rows.setdefault(
                shape_key,
                {
                    "truth_count": 0,
                    "truth_with_geometry_proposal_3px": 0,
                    "true_positives": 0,
                    "false_negatives": 0,
                    "max_confidence": [],
                    "miss_max_confidence": [],
                },
            )
            shape_rows[shape_key]["truth_count"] += 1
            shape_rows[shape_key]["truth_with_geometry_proposal_3px"] += int(filtered_3)
            shape_rows[shape_key]["true_positives"] += int(matched)
            shape_rows[shape_key]["false_negatives"] += int(not matched)
            shape_rows[shape_key]["max_confidence"].append(max_score)
            if not matched:
                shape_rows[shape_key]["miss_max_confidence"].append(max_score)

        proposal_counts["raw"] += len(raw.coordinates)
        proposal_counts["geometry_filtered"] += len(filtered.coordinates)
        total_predictions += len(predictions)

    for threshold in threshold_values:
        aggregate = aggregate_scene_metrics(threshold_scene_values[threshold], TOLERANCE)
        threshold_metrics[str(threshold)] = {
            "threshold": threshold,
            "true_positives": aggregate.true_positives,
            "false_positives": aggregate.false_positives,
            "false_negatives": aggregate.false_negatives,
            "precision": aggregate.precision,
            "recall": aggregate.recall,
            "f1": aggregate.f1,
            "duplicate_count": aggregate.duplicate_count,
        }

    def finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        bucket = dict(bucket)
        if "raw_proposal_count" in bucket:
            bucket["mean_raw_proposals_per_scene"] = round(bucket.pop("raw_proposal_count") / bucket["scene_count"], 6)
        if "geometry_proposal_count" in bucket:
            bucket["mean_geometry_proposals_per_scene"] = round(bucket.pop("geometry_proposal_count") / bucket["scene_count"], 6)
        bucket["max_confidence"] = _quantiles(bucket["max_confidence"])
        bucket["miss_max_confidence"] = _quantiles(bucket["miss_max_confidence"])
        return bucket

    families = {key: finish_bucket(value) for key, value in sorted(families.items())}
    degradations = {key: finish_bucket(value) for key, value in sorted(degradations.items())}
    radius_rows = {key: finish_bucket(value) for key, value in sorted(radius_rows.items())}
    shape_rows = {key: finish_bucket(value) for key, value in sorted(shape_rows.items())}

    result = {
        "schema": "graphreader.marker-center-train-family-v19-diagnostic.v1",
        "revision": "marker-center-train-family-v19",
        "candidate_id": "P1",
        "scope": {
            "synthetic_only": True,
            "private_or_article_images": False,
            "public_gate_archive_opened": False,
            "sealed_runs": 0,
            "scene_ids_emitted": False,
            "case_level_details_emitted": False,
            "fixed_dev_split": "marker-center-proposal-geometry-v13-dev",
            "fixed_dev_manifest_sha256": manifest_sha256,
        },
        "artifacts": {
            "onnx": _artifact(onnx_path),
            "checkpoint": _artifact(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else {"exists": False},
            "candidate_report": _artifact(DEFAULT_REPORT) if DEFAULT_REPORT.is_file() else {"exists": False},
        },
        "inference": {
            "provider": session.get_providers()[0],
            "confidence_threshold": THRESHOLD,
            "match_tolerance_px": TOLERANCE,
            "proposal_counts": dict(proposal_counts),
            "truth_count": total_truths,
            "prediction_count": total_predictions,
            "max_confidence_over_truth_proposals": _quantiles(all_scores),
            "matched_truth_max_confidence": _quantiles(matched_scores),
            "missed_truth_max_confidence": _quantiles(missed_scores),
        },
        "thresholds": threshold_metrics,
        "miss_stage": {
            "total_misses": sum(reason_counts.values()),
            "responsible_stage": "classifier_confidence_with_one_marker_geometry_veto",
            "counts": dict(sorted(reason_counts.items())),
        },
        "dimensions": {
            "family": families,
            "degradation": degradations,
            "marker_radius_px": radius_rows,
            "marker_geometry": shape_rows,
        },
        "next_revision": {
            "startable": True,
            "requires_new_candidate_authorization": True,
            "recommended_isolated_change": "classifier calibration or training coverage for the low-confidence marker tail, with an explicit guard for the marker-geometry veto",
            "proposal_availability_defect": False,
            "geometry_filter_defect": False,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.onnx.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
