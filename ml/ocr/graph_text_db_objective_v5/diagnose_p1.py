# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-time full-selection geometry diagnosis for failed candidate P1."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import median

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.ocr.official_bakeoff import structure_consensus_evaluate as detector_contract

from .dataset import FRAME_HEIGHT, FRAME_WIDTH, build_validation_split


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = Path("ml/ocr/graph_text_db_objective_v5/artifacts/P1-run/candidate-report.json")
ONNX_PATH = Path("ml/ocr/graph_text_db_objective_v5/artifacts/P1-run/graph-text-db-objective-v5-p1.onnx")
OUTPUT_PATH = Path("ml/ocr/graph_text_db_objective_v5/artifacts/P1-run/selection-diagnosis.json")
EXPECTED_REPORT_SHA256 = "21bcd008f62c3b5f223b57768a84bd53356f1c9317bbf0e12e09055a12651d10"
EXPECTED_ONNX_SHA256 = "9119e3031d2dbc16324b15d27acc4f0f8dc44ae658f9b3449cf63bd7aa3bd327"


def _iou(bounds: detector_contract.Box, truth: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = truth
    intersection_width = max(0.0, min(bounds.right, right) - max(bounds.left, left))
    intersection_height = max(0.0, min(bounds.bottom, bottom) - max(bounds.top, top))
    intersection = intersection_width * intersection_height
    union = bounds.width * bounds.height + (right - left) * (bottom - top) - intersection
    return 0.0 if union <= 0 else intersection / union


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def main() -> int:
    report_path = REPO_ROOT / REPORT_PATH
    onnx_path = REPO_ROOT / ONNX_PATH
    output_path = REPO_ROOT / OUTPUT_PATH
    if output_path.exists():
        raise RuntimeError(f"P1 diagnosis already exists and cannot be rerun: {output_path}")
    if sha256_file(report_path) != EXPECTED_REPORT_SHA256:
        raise RuntimeError("P1 report changed before diagnosis")
    if sha256_file(onnx_path) != EXPECTED_ONNX_SHA256:
        raise RuntimeError("P1 ONNX changed before diagnosis")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_by_id = {
        str(record["case_id"]): record
        for record in report["selection_metrics"]["records"]
    }
    frames = build_validation_split()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    group_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "fixture_count": 0,
            "exact_fixture_count": 0,
            "no_region_count": 0,
            "below_iou_count": 0,
            "multi_region_count": 0,
            "false_region_count": 0,
        }
    )
    no_region_maximums: list[float] = []
    no_region_pixels_above_threshold: list[float] = []
    exact_width_ratios: list[float] = []
    exact_height_ratios: list[float] = []
    below_iou_values: list[float] = []
    below_iou_width_ratios: list[float] = []
    below_iou_height_ratios: list[float] = []
    below_iou_center_offsets: list[float] = []
    multi_region_cases: list[dict[str, object]] = []
    exclusion_false_cases: list[dict[str, object]] = []

    for frame in frames:
        tensor = detector_contract.detector_tensor(frame.detector_bgr, FRAME_WIDTH, FRAME_HEIGHT)
        probabilities = np.asarray(
            session.run([output_name], {input_name: tensor})[0],
            dtype=np.float32,
        )
        regions = detector_contract.db_model_regions(probabilities, FRAME_WIDTH, FRAME_HEIGHT)
        expected = expected_by_id[frame.case_id]
        if len(regions) != int(expected["prediction_count"]):
            raise RuntimeError(f"P1 diagnosis prediction count drifted for {frame.case_id}")
        group_key = f"{frame.structure_family}|{frame.degradation_family}"
        group = group_counts[group_key]
        group["fixture_count"] += 1
        group["exact_fixture_count"] += int(bool(expected["exact"]))
        group["multi_region_count"] += int(len(regions) > 1)
        group["false_region_count"] += int(expected["false_region_count"])

        if frame.truth_bbox is None:
            if regions:
                exclusion_false_cases.append(
                    {
                        "case_id": frame.case_id,
                        "group": group_key,
                        "maximum_probability": float(probabilities.max()),
                        "pixels_above_0_3": int(np.count_nonzero(probabilities > np.float32(0.3))),
                        "regions": [
                            {"bounds": region.bounds.to_json(), "confidence": region.confidence}
                            for region in regions
                        ],
                    }
                )
            continue

        truth_width = frame.truth_bbox[2] - frame.truth_bbox[0]
        truth_height = frame.truth_bbox[3] - frame.truth_bbox[1]
        ranked = sorted(
            ((_iou(region.bounds, frame.truth_bbox), region) for region in regions),
            reverse=True,
            key=lambda item: item[0],
        )
        if not ranked:
            group["no_region_count"] += 1
            no_region_maximums.append(float(probabilities.max()))
            no_region_pixels_above_threshold.append(
                float(np.count_nonzero(probabilities > np.float32(0.3)))
            )
            continue

        best_iou, best = ranked[0]
        width_ratio = best.bounds.width / truth_width
        height_ratio = best.bounds.height / truth_height
        truth_center_x = (frame.truth_bbox[0] + frame.truth_bbox[2]) / 2.0
        truth_center_y = (frame.truth_bbox[1] + frame.truth_bbox[3]) / 2.0
        center_offset = float(
            np.hypot(
                ((best.bounds.left + best.bounds.right) / 2.0) - truth_center_x,
                ((best.bounds.top + best.bounds.bottom) / 2.0) - truth_center_y,
            )
        )
        if best_iou >= 0.5:
            exact_width_ratios.append(width_ratio)
            exact_height_ratios.append(height_ratio)
        else:
            group["below_iou_count"] += 1
            below_iou_values.append(best_iou)
            below_iou_width_ratios.append(width_ratio)
            below_iou_height_ratios.append(height_ratio)
            below_iou_center_offsets.append(center_offset)
        if len(regions) > 1:
            multi_region_cases.append(
                {
                    "case_id": frame.case_id,
                    "group": group_key,
                    "best_truth_iou": best_iou,
                    "regions": [
                        {"bounds": region.bounds.to_json(), "confidence": region.confidence}
                        for region in regions
                    ],
                }
            )

    diagnosis = {
        "schema": "graphreader.ocr-graph-text-db-objective-selection-diagnosis.v1",
        "task": "ocr-detection",
        "revision": "graph-text-db-objective-v5",
        "candidate_id": "P1",
        "diagnostic_runs": 1,
        "threshold_sweeps": 0,
        "selection_report_path": REPORT_PATH.as_posix(),
        "selection_report_sha256": EXPECTED_REPORT_SHA256,
        "onnx_path": ONNX_PATH.as_posix(),
        "onnx_sha256": EXPECTED_ONNX_SHA256,
        "public_gate_evaluations": 0,
        "sealed_public_archive_opened": False,
        "production_approval": False,
        "release_eligible": False,
        "group_counts": dict(sorted(group_counts.items())),
        "no_region_case_count": len(no_region_maximums),
        "no_region_maximum_probability_minimum": min(no_region_maximums, default=None),
        "no_region_pixels_above_0_3_median": _median(no_region_pixels_above_threshold),
        "exact_prediction_width_ratio_median": _median(exact_width_ratios),
        "exact_prediction_height_ratio_median": _median(exact_height_ratios),
        "below_iou_case_count": len(below_iou_values),
        "below_iou_median": _median(below_iou_values),
        "below_iou_width_ratio_median": _median(below_iou_width_ratios),
        "below_iou_height_ratio_median": _median(below_iou_height_ratios),
        "below_iou_center_offset_pixels_median": _median(below_iou_center_offsets),
        "multi_region_cases": multi_region_cases,
        "exclusion_false_cases": exclusion_false_cases,
    }
    output_path.write_bytes(canonical_json_bytes(diagnosis))
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
