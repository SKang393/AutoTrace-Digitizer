# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate-only diagnosis of retry4 accepted generic false positives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ml.markers.center.mask_preserving_v24.mask_preserving import extract_proposals, postprocess
from ml.markers.center.real_range_generator_v1.negative_sampler import CONNECTOR_ANCHOR_FRACTIONS, CONNECTOR_ANCHOR_MAX_DISTANCE_PX, TOPOLOGY_KINDS, TOPOLOGY_RADIUS_PX, _connector_anchor_indices, _topology_indices
from ml.markers.center.real_range_generator_v1.generator import build_split

from .diagnose_retry import (
    RETRY4_DEV_SPLIT_SHA256,
    RETRY4_GENERATOR_AUDIT_SHA256,
    RETRY4_ONNX_SHA256,
    RETRY5_DEV_SPLIT_SHA256,
    RETRY5_GENERATOR_AUDIT_SHA256,
    RETRY5_ONNX_SHA256,
    ROOT,
    THRESHOLD,
    _labels,
    _matched_truths,
    _patch_morphology,
    _quantiles,
    _sha,
    _strata,
)


TOLERANCE = 5.0
LOWER_BAND_Y_MIN = 130.0
HARD_DISTANCE_BINS = ((0.0, 8.0, "<=8"), (8.0, 16.0, "8-16"), (16.0, 32.0, "16-32"), (32.0, math.inf, ">32"))
TRUTH_DISTANCE_BINS = ((0.0, 5.0, "<=5"), (5.0, 15.0, "5-15"), (15.0, 30.0, "15-30"), (30.0, math.inf, ">30"))
LINE_DISTANCE_BINS = ((0.0, 2.0, "<=2"), (2.0, 4.0, "2-4"), (4.0, 8.0, "4-8"), (8.0, math.inf, ">8"))
HARD_KINDS = ("text", "line_intersection", "axis", "faint_line", "ocr_heavy", "topology_junction", "topology_fragment")


def _bin(value: float, bins: tuple[tuple[float, float, str], ...]) -> str:
    for lower, upper, label in bins:
        if lower <= value <= upper if math.isinf(upper) else lower <= value < upper:
            return label
    return bins[-1][2]


def _point_segment_distance(point: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
    px, py = point
    ax, ay = first
    bx, by = second
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.hypot(px - ax, py - ay)
    projection = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + projection * dx), py - (ay + projection * dy))


def _line_distance(point: tuple[float, float], centers: tuple[tuple[float, float], ...]) -> float:
    return min(
        (_point_segment_distance(point, first, second) for first, second in zip(centers, centers[1:])),
        default=math.inf,
    )


def _mask_bin(value: float) -> str:
    if value == 0.0:
        return "zero"
    if value <= 0.25:
        return "0-25pct"
    if value <= 0.50:
        return "25-50pct"
    if value <= 0.75:
        return "50-75pct"
    return ">75pct"


def summarize(model_path: Path, *, retry5: bool = False) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_hash = _sha(model_path)
    expected_model = RETRY5_ONNX_SHA256 if retry5 else RETRY4_ONNX_SHA256
    if model_hash != expected_model:
        mode = "retry5" if retry5 else "retry4"
        raise ValueError(f"{mode} ONNX hash mismatch: expected {expected_model}, got {model_hash}")
    audit_path = ROOT / "ml/markers/center/real_range_generator_v1/AUDIT.json"
    audit_hash = _sha(audit_path)
    expected_audit = RETRY5_GENERATOR_AUDIT_SHA256 if retry5 else RETRY4_GENERATOR_AUDIT_SHA256
    if audit_hash != expected_audit:
        mode = "retry5" if retry5 else "retry4"
        raise ValueError(f"{mode} generator audit hash mismatch: expected {expected_audit}, got {audit_hash}")
    audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
    dev_split_hash = audit_record["splits"]["dev"]["aggregate_sha256"]
    expected_dev = RETRY5_DEV_SPLIT_SHA256 if retry5 else RETRY4_DEV_SPLIT_SHA256
    if dev_split_hash != expected_dev:
        mode = "retry5" if retry5 else "retry4"
        raise ValueError(f"{mode} dev split hash mismatch: expected {expected_dev}, got {dev_split_hash}")

    sampler_path = ROOT / "ml/markers/center/real_range_generator_v1/negative_sampler.py"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("CPUExecutionProvider was not selected")
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    scenes = build_split("dev")
    hard_distances: dict[str, Counter[str]] = {kind: Counter() for kind in HARD_KINDS}
    truth_distances: Counter[str] = Counter()
    line_distances: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    mask_counts: dict[str, Counter[str]] = {"ocr_mean": Counter(), "artifact_mean": Counter()}
    morphology: dict[str, list[float]] = defaultdict(list)
    confidence: list[float] = []
    root_causes: Counter[str] = Counter()
    generic_count = 0
    accepted_count = 0
    all_fp_count = 0

    for scene in scenes:
        proposals = extract_proposals(scene.tensor)
        labels, hard = _labels(scene, proposals.coordinates)
        names = _strata(proposals.patches, hard, labels)
        topology = _topology_indices(scene, proposals.coordinates, labels)
        topology_by_index = {index: kind for kind in TOPOLOGY_KINDS for index in topology[kind]}
        connector_indices = _connector_anchor_indices(scene, proposals.coordinates, labels) if retry5 else set()
        connector_indices -= set(topology_by_index)
        output = session.run([output_name], {input_name: proposals.patches.numpy().astype(np.float32, copy=False)})[0]
        predictions = postprocess(scene, proposals, output)
        used_predictions, _ = _matched_truths(predictions, scene.centers)
        all_fp_count += len(predictions) - len(used_predictions)
        decoded = [
            (float(x + output[index, 1] * 4.0), float(y + output[index, 2] * 4.0), index)
            for index, (x, y) in enumerate(proposals.coordinates.tolist())
            if float(output[index, 0]) >= THRESHOLD
        ]
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in used_predictions or not decoded:
                continue
            _, _, source = min(decoded, key=lambda item: math.hypot(prediction.x - item[0], prediction.y - item[1]))
            source_name = topology_by_index.get(source, "connector_anchor" if retry5 and source in connector_indices else names[source])
            if source_name != "generic":
                continue
            generic_count += 1
            accepted_count += 1
            source_point = (float(prediction.x), float(prediction.y))
            nearest_truth = min((math.hypot(source_point[0] - x, source_point[1] - y) for x, y in scene.centers), default=math.inf)
            truth_distances[_bin(nearest_truth, TRUTH_DISTANCE_BINS)] += 1
            line_distance = _line_distance(source_point, scene.centers)
            line_distances[_bin(line_distance, LINE_DISTANCE_BINS)] += 1
            band_counts["lower_synthetic_negative_band" if source_point[1] >= LOWER_BAND_Y_MIN else "marker_field"] += 1
            hard_candidates: list[tuple[float, str]] = []
            for kind in HARD_KINDS:
                points = [(x, y) for item_kind, x, y in scene.hard_negatives if item_kind == kind]
                distance = min((math.hypot(source_point[0] - x, source_point[1] - y) for x, y in points), default=math.inf)
                hard_distances[kind][_bin(distance, HARD_DISTANCE_BINS)] += 1
                hard_candidates.append((distance, kind))
            patch = proposals.patches[source]
            features = _patch_morphology(patch)
            for key, value in features.items():
                morphology[key].append(float(value))
            confidence.append(float(output[source, 0]))
            ocr_mean = float(patch[1].mean())
            artifact_mean = float(patch[2].mean())
            mask_counts["ocr_mean"][_mask_bin(ocr_mean)] += 1
            mask_counts["artifact_mean"][_mask_bin(artifact_mean)] += 1
            nearest_hard_distance, nearest_hard_kind = min(hard_candidates)
            if nearest_hard_distance <= 8.0:
                root_causes[f"near_{nearest_hard_kind}"] += 1
            elif line_distance <= 8.0:
                root_causes["near_connecting_line"] += 1
            elif ocr_mean > 0.0 or artifact_mean > 0.0:
                root_causes["masked_context"] += 1
            elif source_point[1] >= LOWER_BAND_Y_MIN:
                root_causes["lower_band_generic"] += 1
            else:
                root_causes["marker_field_generic"] += 1

    expected_generic_count = 183 if retry5 else 136
    if generic_count != expected_generic_count:
        mode = "retry5" if retry5 else "retry4"
        raise RuntimeError(f"{mode} generic accepted false-positive count changed: {generic_count}")
    if sum(root_causes.values()) != generic_count:
        raise RuntimeError("generic false-positive root-cause partition is not exhaustive")

    result = {
        "schema": "graphreader.marker-center-mask-preserving-v24-retry5-generic-fp-diagnosis.v1" if retry5 else "graphreader.marker-center-mask-preserving-v24-retry4-generic-fp-diagnosis.v1",
        "revision": "marker-center-mask-preserving-v24",
        "scope": {"synthetic_only": True, "split": "real-range-generator-v1-dev", "scene_count": len(scenes), "truth_count": sum(len(scene.centers) for scene in scenes), "threshold": THRESHOLD, "private_data": False, "real_dev_reads": 0, "real_sealed_reads": 0, "optimizer_steps": 0, "case_ids_or_pixels_emitted": False, "retry_mode": "retry5" if retry5 else "retry4"},
        "binding": {"model_path": model_path.name, "model_sha256": model_hash, "provider": session.get_providers()[0], "generator_audit_sha256": audit_hash, "generator_dev_split_sha256": dev_split_hash, "negative_sampler_sha256": _sha(sampler_path), "topology_radius_px": TOPOLOGY_RADIUS_PX, "connector_anchor_fractions": list(CONNECTOR_ANCHOR_FRACTIONS) if retry5 else None, "connector_anchor_max_distance_px": CONNECTOR_ANCHOR_MAX_DISTANCE_PX if retry5 else None},
        "fixed_threshold_metrics": {"accepted_false_positive_count": all_fp_count, "generic_accepted_false_positive_count": generic_count, "expected_generic_count": expected_generic_count, "root_cause_partition_exhaustive": True},
        "root_cause_counts": dict(sorted(root_causes.items())),
        "source_geometry": {"nearest_truth_distance_bins": dict(sorted(truth_distances.items())), "declared_hard_negative_distance_bins": {kind: dict(sorted(values.items())) for kind, values in hard_distances.items()}, "connecting_line_distance_bins": dict(sorted(line_distances.items())), "source_band": dict(sorted(band_counts.items()))},
        "mask_occupancy_bins": {key: dict(sorted(values.items())) for key, values in mask_counts.items()},
        "morphology_quantiles": {key: _quantiles(values) for key, values in sorted(morphology.items())},
        "probability_quantiles": _quantiles(confidence),
        "diagnostic_conclusion": "retry4 generic accepted false positives are partitioned by fixed source geometry, masks, morphology, and confidence; no threshold or model change is proposed",
    }
    if retry5:
        result["comparison_to_retry4"] = {"root_cause_counts": {"near_connecting_line": 102, "masked_context": 13, "marker_field_generic": 21}}
        result["diagnostic_conclusion"] = "retry5 remaining generic accepted false positives are partitioned by fixed source geometry, masks, morphology, and confidence; retry4 root-cause counts are reported for comparison and no threshold or model change is proposed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry5", action="store_true")
    args = parser.parse_args()
    report = summarize(args.model.resolve(), retry5=args.retry5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
