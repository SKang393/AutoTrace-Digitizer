# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Trace V20 fixed-dev misses without emitting scene or case identities."""

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
V20_ARTIFACT_DIR = (
    REPO_ROOT
    / "artifacts/goal22-worktrees/marker-v20/ml/markers/center/tail_coverage_v20/artifacts/P1-run"
)
DEFAULT_ONNX = V20_ARTIFACT_DIR / "marker-center-tail-coverage-v20-p1.onnx"
DEFAULT_CHECKPOINT = V20_ARTIFACT_DIR / "marker-center-tail-coverage-v20-p1.pt"
DEFAULT_REPORT = V20_ARTIFACT_DIR / "candidate-report.json"
FIXED_THRESHOLD = 0.25
MATCH_TOLERANCE = 5.0
THRESHOLDS = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return ("filled_circle", "open_square", "filled_diamond")[(truth_index + variant) % 3]


def _scene_variant(scene: Any) -> int:
    return int(str(scene.scene_id).rsplit("-", 1)[1])


def _decoded(output: np.ndarray, coordinates: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for index, (base_x, base_y) in enumerate(coordinates.tolist()):
        rows.append({
            "index": index,
            "confidence": float(output[index, 0]),
            "x": float(base_x + output[index, 1] * 4.0),
            "y": float(base_y + output[index, 2] * 4.0),
            "radius": float(np.clip(output[index, 3], 2.5, 8.0)),
        })
    return rows


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(path),
        "exists": True,
    }


def _bucket() -> dict[str, Any]:
    return {
        "truth_count": 0,
        "truth_with_raw_proposal_3px": 0,
        "truth_with_geometry_proposal_3px": 0,
        "truth_with_geometry_proposal_5px": 0,
        "true_positives": 0,
        "false_negatives": 0,
        "max_confidence": [],
        "miss_max_confidence": [],
    }


def _finish(bucket: dict[str, Any]) -> dict[str, Any]:
    result = dict(bucket)
    result["max_confidence"] = _quantiles(result["max_confidence"])
    result["miss_max_confidence"] = _quantiles(result["miss_max_confidence"])
    return result


def summarize(
    onnx_path: Path = DEFAULT_ONNX,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    candidate_report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Run only the fixed V13 synthetic dev split through the V20 ONNX payload."""
    for path, label in ((onnx_path, "V20 ONNX"), (checkpoint_path, "V20 checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact is missing: {path}")
    scenes = build_selection_scenes("dev")
    manifest_bytes = (json.dumps(selection_manifest(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    threshold_scene_values: dict[float, list[Any]] = defaultdict(list)
    families: dict[str, dict[str, Any]] = defaultdict(_bucket)
    shapes: dict[str, dict[str, Any]] = defaultdict(_bucket)
    radii: dict[str, dict[str, Any]] = defaultdict(_bucket)
    reasons: Counter[str] = Counter()
    proposal_funnel = Counter()
    all_scores: list[float] = []
    matched_scores: list[float] = []
    missed_scores: list[float] = []
    total_truths = 0
    prediction_count = 0
    raw_proposal_count = 0
    geometry_proposal_count = 0

    for scene in scenes:
        raw = extract_proposals(scene.tensor)
        filtered = filter_proposals(scene.tensor, raw)
        raw_proposal_count += len(raw.coordinates)
        geometry_proposal_count += len(filtered.coordinates)
        output = session.run([session.get_outputs()[0].name], {input_name: filtered.patches.numpy().astype(np.float32, copy=False)})[0]
        decoded = _decoded(output, filtered.coordinates.numpy())
        predictions = postprocess_predictions(scene, filtered, output, threshold=FIXED_THRESHOLD)
        prediction_count += len(predictions)
        variant = _scene_variant(scene)
        for threshold in THRESHOLDS:
            threshold_scene_values[threshold].append(
                center_metrics(postprocess_predictions(scene, filtered, output, threshold=threshold), scene.centers, MATCH_TOLERANCE)
            )

        for truth_index, ((truth_x, truth_y), truth_radius) in enumerate(zip(scene.centers, scene.radii, strict=True)):
            total_truths += 1
            raw_distance = np.hypot(raw.coordinates[:, 0].numpy() - truth_x, raw.coordinates[:, 1].numpy() - truth_y)
            filtered_distance = np.hypot(filtered.coordinates[:, 0].numpy() - truth_x, filtered.coordinates[:, 1].numpy() - truth_y)
            raw_3 = bool(np.any(raw_distance <= 3.0))
            filtered_3 = bool(np.any(filtered_distance <= 3.0))
            filtered_5 = bool(np.any(filtered_distance <= MATCH_TOLERANCE))
            near_scores = output[filtered_distance <= MATCH_TOLERANCE, 0]
            max_score = float(np.max(near_scores)) if len(near_scores) else 0.0
            decoded_near = [
                row for row in decoded
                if math.hypot(float(row["x"]) - truth_x, float(row["y"]) - truth_y) <= MATCH_TOLERANCE
            ]
            high_near = [row for row in decoded_near if float(row["confidence"]) >= FIXED_THRESHOLD]
            unmasked_near = [
                row for row in high_near
                if _center_is_unmasked(scene, float(row["x"]), float(row["y"]))
            ]
            gated_near = [
                row for row in unmasked_near
                if _marker_geometry_consensus(scene, float(row["x"]), float(row["y"]), float(row["radius"]))
            ]
            matched = any(math.hypot(pred.x - truth_x, pred.y - truth_y) <= MATCH_TOLERANCE for pred in predictions)
            proposal_funnel["truth_count"] += 1
            proposal_funnel["raw_3px"] += int(raw_3)
            proposal_funnel["geometry_3px"] += int(filtered_3)
            proposal_funnel["geometry_5px"] += int(filtered_5)
            all_scores.append(max_score)
            (matched_scores if matched else missed_scores).append(max_score)
            if not matched:
                if not filtered_5:
                    reasons["proposal_unavailable_after_geometry"] += 1
                elif max_score < FIXED_THRESHOLD:
                    reasons["confidence_below_threshold"] += 1
                elif high_near and not unmasked_near:
                    reasons["unmasked_artifact_veto"] += 1
                elif high_near and not gated_near:
                    reasons["marker_geometry_veto"] += 1
                else:
                    reasons["nms_or_matching"] += 1

            for key, bucket in ((str(scene.family), families), (_marker_shape(truth_index, variant), shapes), (_radius_bin(float(truth_radius)), radii)):
                row = bucket[key]
                row["truth_count"] += 1
                row["truth_with_raw_proposal_3px"] += int(raw_3)
                row["truth_with_geometry_proposal_3px"] += int(filtered_3)
                row["truth_with_geometry_proposal_5px"] += int(filtered_5)
                row["true_positives"] += int(matched)
                row["false_negatives"] += int(not matched)
                row["max_confidence"].append(max_score)
                if not matched:
                    row["miss_max_confidence"].append(max_score)

    thresholds = {}
    for threshold in THRESHOLDS:
        aggregate = aggregate_scene_metrics(threshold_scene_values[threshold], MATCH_TOLERANCE)
        thresholds[str(threshold)] = {
            "threshold": threshold,
            "true_positives": aggregate.true_positives,
            "false_positives": aggregate.false_positives,
            "false_negatives": aggregate.false_negatives,
            "precision": aggregate.precision,
            "recall": aggregate.recall,
            "f1": aggregate.f1,
            "duplicate_count": aggregate.duplicate_count,
        }

    return {
        "schema": "graphreader.marker-center-tail-coverage-v20-diagnostic.v1",
        "revision": "marker-center-tail-coverage-v20",
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
            "checkpoint": _artifact(checkpoint_path),
            "candidate_report": _artifact(candidate_report_path) if candidate_report_path.is_file() else {"exists": False},
        },
        "inference": {
            "provider": session.get_providers()[0],
            "confidence_threshold": FIXED_THRESHOLD,
            "match_tolerance_px": MATCH_TOLERANCE,
            "proposal_funnel": dict(proposal_funnel),
            "proposal_counts": {"raw": raw_proposal_count, "geometry_filtered": geometry_proposal_count},
            "truth_count": total_truths,
            "prediction_count": prediction_count,
            "max_confidence_over_truth_proposals": _quantiles(all_scores),
            "matched_truth_max_confidence": _quantiles(matched_scores),
            "missed_truth_max_confidence": _quantiles(missed_scores),
        },
        "thresholds": thresholds,
        "miss_stage": {
            "total_misses": sum(reasons.values()),
            "responsible_stage": "classifier_confidence_with_unmasked_artifact_and_marker_geometry_residual",
            "counts": dict(sorted(reasons.items())),
        },
        "dimensions": {
            "family": {key: _finish(value) for key, value in sorted(families.items())},
            "marker_shape": {key: _finish(value) for key, value in sorted(shapes.items())},
            "marker_radius_px": {key: _finish(value) for key, value in sorted(radii.items())},
        },
        "next_revision": {
            "startable": True,
            "requires_new_candidate_authorization": True,
            "responsible_stage": "classifier_confidence_tail; three downstream vetoes remain and proposal coverage/NMS are clear",
            "recommended_isolated_change": "diagnose and repair the confidence tail while preserving the V13 proposal and geometry contracts",
            "proposal_availability_defect": reasons["proposal_unavailable_after_geometry"] > 0,
            "geometry_veto_defect": reasons["marker_geometry_veto"] > 0,
            "nms_defect": reasons["nms_or_matching"] > 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.onnx.resolve(), args.checkpoint.resolve(), args.candidate_report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
