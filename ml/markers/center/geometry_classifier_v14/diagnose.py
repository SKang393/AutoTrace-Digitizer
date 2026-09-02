# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Evaluate fixed deterministic geometry scores on V13 synthetic proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import math
import time

import numpy as np

from ml.markers.center.metrics import aggregate_scene_metrics, center_metrics
from ml.markers.center.line_aware_v1.pipeline import MarkerPrediction, extract_proposals
from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes, seal_manifest
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals
from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from .features import score_proposals
from .protocol import ACCEPTANCE_BAR, EVALUATOR_SOURCE_PATHS, EVIDENCE_POLICY_PATH, SOURCE_BUNDLE_PATHS, THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parents[4]


def _decode(scene, threshold: float) -> tuple[MarkerPrediction, ...]:
    proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
    features = score_proposals(scene.tensor, proposals)
    candidates = [MarkerPrediction(float(coordinate[0]), float(coordinate[1]), feature.radius, feature.score) for coordinate, feature in zip(proposals.coordinates.tolist(), features, strict=True) if feature.score >= threshold]
    accepted: list[MarkerPrediction] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.y, item.x)):
        if any(math.hypot(candidate.x - current.x, candidate.y - current.y) < max(5.0, 1.25 * max(candidate.radius, current.radius)) for current in accepted):
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: (item.y, item.x, -item.confidence)))


def _metrics(scenes, threshold: float) -> dict[str, object]:
    values = []
    prohibited_hits = 0
    proposal_count = 0
    covered_truth = 0
    truth_count = 0
    for scene in scenes:
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        proposal_count += len(proposals.coordinates)
        truth_count += len(scene.centers)
        covered_truth += sum(any(math.hypot(float(point[0]) - x, float(point[1]) - y) <= 5.0 for point in proposals.coordinates.tolist()) for x, y in scene.centers)
        predictions = _decode(scene, threshold)
        values.append(center_metrics(predictions, scene.centers, 5.0))
        prohibited_hits += sum(any(math.hypot(prediction.x - point.x, prediction.y - point.y) <= 5.0 for prediction in predictions) for point in scene.prohibited)
    aggregate = aggregate_scene_metrics(values, 5.0)
    return {"threshold": threshold, "scene_count": len(scenes), "proposal_count": proposal_count, "truth_coverage_recall": covered_truth / max(1, truth_count), "prediction_count": aggregate.true_positives + aggregate.false_positives, "true_positives": aggregate.true_positives, "false_positives": aggregate.false_positives, "false_negatives": aggregate.false_negatives, "duplicate_count": aggregate.duplicate_count, "precision": aggregate.precision, "recall": aggregate.recall, "f1": aggregate.f1, "prohibited_structure_hits": prohibited_hits}


def _separation(scenes) -> dict[str, float | int]:
    positive, negative = [], []
    for scene in scenes:
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        features = score_proposals(scene.tensor, proposals)
        for coordinate, feature in zip(proposals.coordinates.tolist(), features, strict=True):
            is_positive = any((float(coordinate[0]) - x) ** 2 + (float(coordinate[1]) - y) ** 2 <= 25.0 for x, y in scene.centers)
            (positive if is_positive else negative).append(feature.score)
    return {"positive_count": len(positive), "negative_count": len(negative), "positive_min": min(positive, default=0.0), "positive_median": float(np.median(positive)) if positive else 0.0, "negative_max": max(negative, default=0.0), "negative_median": float(np.median(negative)) if negative else 0.0}


def run(output_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_sha256 = seal_manifest(output_dir)
    scenes = {split: build_selection_scenes(split) for split in ("train", "dev")}
    rows = {split: [_metrics(values, threshold) for threshold in THRESHOLDS] for split, values in scenes.items()}
    policy_path = REPO_ROOT / EVIDENCE_POLICY_PATH
    source_paths = tuple(Path(path) for path in SOURCE_BUNDLE_PATHS)
    evaluator_paths = tuple(Path(path) for path in EVALUATOR_SOURCE_PATHS)
    passed_thresholds = [threshold for threshold in THRESHOLDS if float(rows["dev"][list(THRESHOLDS).index(threshold)]["precision"]) >= ACCEPTANCE_BAR["precision_minimum"] and float(rows["dev"][list(THRESHOLDS).index(threshold)]["recall"]) >= ACCEPTANCE_BAR["recall_minimum"]]
    report = {"schema": "graphreader.marker-center-geometry-classifier-diagnostic.v14", "revision": "marker-center-geometry-classifier-v14", "status": "pass" if passed_thresholds else "failed_dev", "synthetic_only": True, "private_or_article_images": False, "public_gate_archive_opened": False, "evidence_policy_path": EVIDENCE_POLICY_PATH, "evidence_policy_sha256": sha256_file(policy_path), "source_bundle_paths": list(SOURCE_BUNDLE_PATHS), "source_bundle_sha256": source_bundle_sha256(REPO_ROOT, source_paths), "canonical_evaluator_paths": list(EVALUATOR_SOURCE_PATHS), "canonical_evaluator_sha256": source_bundle_sha256(REPO_ROOT, evaluator_paths), "manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(), "manifest_sha256": manifest_sha256, "acceptance_bar": ACCEPTANCE_BAR, "thresholds": list(THRESHOLDS), "passed_thresholds": passed_thresholds, "feature_definition": "0.35 radial_support + 0.25 compactness + 0.20 isotropy + 0.20 mask_clear - 0.35 line_evidence, clipped to [0,1]", "separation": {split: _separation(values) for split, values in scenes.items()}, "metrics": rows, "sealed_runs": 0, "public_gate_evaluations": 0, "private_data": False, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    path = output_dir / "geometry-classifier-diagnostic.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "ml/markers/center/artifacts/geometry-classifier-v14")
    args = parser.parse_args()
    report = run(args.output_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
