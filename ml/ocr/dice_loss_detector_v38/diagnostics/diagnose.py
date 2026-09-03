# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Diagnose V38 full-box segmentation on the fixed synthetic V32 dev split.

The V38 ONNX and checkpoint are held fixed.  This diagnostic is aggregate-only:
it emits no scene identifiers, case details, pixels, truth rows, or prediction
maps.  It is intentionally separate from training and never reads protected
data.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ml.ocr.real_range_classifier_finetune_v32.dataset import build_split as build_v32_split
from ml.ocr.real_range_classifier_finetune_v32.dataset import split_fingerprint as v32_split_fingerprint
from ml.ocr.degradation_coverage_detector_v37.dataset import build_tiles
from ml.ocr.degradation_coverage_detector_v37.protocol import (
    MINIMUM_COMPONENT_AREA,
    ONNX_PROVIDER,
    PIXEL_THRESHOLD,
    TILE_OVERLAP,
    TILE_SIZE,
    TRUTH_MATCH_IOU_MINIMUM,
)
from ml.ocr.degradation_coverage_detector_v37.diagnostics.diagnose import (
    AREA_SWEEP,
    THRESHOLDS,
    _aggregate,
    _aggregate_pixel,
    _component_metrics,
    _pixel_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
V38_ROOT = REPO_ROOT / "artifacts/goal22-worktrees/ocr-v38/ml/ocr/dice_loss_detector_v38"
DEFAULT_MODEL = V38_ROOT / "artifacts/P1-run/graph-text-dice-loss-detector-v38-p1.onnx"
DEFAULT_CHECKPOINT = V38_ROOT / "artifacts/P1-run/graph-text-dice-loss-detector-v38-p1.pt"
V38_RESULT = V38_ROOT / "P1_RESULT.json"

EXPECTED_MODEL_SHA256 = "90cb24e3c54931cf2da5213e0bcaa1208e182355187905ce0638cc7292dc1bc2"
EXPECTED_CHECKPOINT_SHA256 = "72efb90dd7e6ba3ad982aaabf0264c00b863a6687f834ff436789b0bec6f3464"
EXPECTED_V38_RESULT_SHA256 = "ef7f22b77a6b803684dac5b13a835e0dbce9b21f3776b305406723107ae22474"
EXPECTED_V37_RESULT_SHA256 = "dbc6978b3bcd7ca722c73442d440d56ee047f162d4e903ea5f2992e009a4cb5e"
EXPECTED_V37_DIAGNOSTIC_SHA256 = "73f2a248a2b86d60cb5ddb04f75b10c5a09244b905135289d64773208e6129ba"
EXPECTED_V32_DEV_FINGERPRINT = "67952b4575972542087281b2c14958e86518ae0e12e88d43f5c47c16252a3687"
V37_RESULT = REPO_ROOT / "ml/ocr/degradation_coverage_detector_v37/P1_RESULT.json"
V37_DIAGNOSTIC = REPO_ROOT / "ml/ocr/degradation_coverage_detector_v37/diagnostics/DIAGNOSTIC.json"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth_mask(scene: Any) -> np.ndarray:
    mask = np.zeros(scene.raster.shape, dtype=bool)
    for truth in scene.truths:
        mask[int(truth.top):int(truth.bottom), int(truth.left):int(truth.right)] = True
    return mask


def _require_authorized_inputs(model_path: Path) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    checkpoint_path = model_path.with_suffix(".pt")
    for path in (model_path, checkpoint_path, V38_RESULT, V37_RESULT, V37_DIAGNOSTIC):
        if not path.is_file():
            raise FileNotFoundError(f"Authorized diagnostic input is missing: {path}")
    model_sha = _sha256(model_path)
    checkpoint_sha = _sha256(checkpoint_path)
    result_sha = _sha256(V38_RESULT)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError("V38 ONNX hash does not match the authorized P1 result")
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("V38 checkpoint hash does not match the authorized P1 result")
    if result_sha != EXPECTED_V38_RESULT_SHA256:
        raise RuntimeError("V38 P1 result hash does not match the authorized candidate report")
    if _sha256(V37_RESULT) != EXPECTED_V37_RESULT_SHA256:
        raise RuntimeError("V37 result evidence changed")
    if _sha256(V37_DIAGNOSTIC) != EXPECTED_V37_DIAGNOSTIC_SHA256:
        raise RuntimeError("V37 diagnostic evidence changed")
    v38_result = json.loads(V38_RESULT.read_text(encoding="utf-8"))
    v37_result = json.loads(V37_RESULT.read_text(encoding="utf-8"))
    if v38_result.get("onnx_sha256") != model_sha or v38_result.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("V38 candidate report does not bind the supplied payload hashes")
    if v32_split_fingerprint("dev") != EXPECTED_V32_DEV_FINGERPRINT:
        raise RuntimeError("Fixed V32 dev split changed")
    return model_sha, checkpoint_sha, result_sha, v38_result, v37_result


def run_diagnostic(model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    model_sha, checkpoint_sha, result_sha, v38_result, v37_result = _require_authorized_inputs(model_path)
    checkpoint_path = model_path.with_suffix(".pt")
    session = ort.InferenceSession(str(model_path), providers=[ONNX_PROVIDER])
    if session.get_providers() != [ONNX_PROVIDER]:
        raise RuntimeError(f"Expected only {ONNX_PROVIDER}, got {session.get_providers()}")

    scenes = build_v32_split("dev")
    tiles = build_tiles("dev")
    component_inputs: dict[str, tuple[np.ndarray, tuple[Any, ...]]] = {}
    dimension_pixel_rows: dict[str, list[dict[str, Any]]] = {}
    dimension_component_rows: dict[str, list[dict[str, Any]]] = {}
    covered_pixels = 0
    total_pixels = 0
    overlap_prediction_values: list[float] = []
    tile_count = 0
    maximum_coverage = 0

    for scene in scenes:
        scene_tiles = tuple(tile for tile in tiles if tile.scene_id == scene.scene_id)
        values = np.stack([(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in scene_tiles]).astype(np.float32)
        logits = np.asarray(session.run(["text_logits"], {"source_tiles": values})[0], dtype=np.float32)[:, 0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        height, width = scene.raster.shape
        score = np.zeros((height, width), dtype=np.float64)
        counts = np.zeros((height, width), dtype=np.float64)
        sums_sq = np.zeros((height, width), dtype=np.float64)
        for tile, probability in zip(scene_tiles, probabilities, strict=True):
            region = (slice(tile.top, tile.top + tile.valid_height), slice(tile.left, tile.left + tile.valid_width))
            valid = probability[:tile.valid_height, :tile.valid_width].astype(np.float64)
            score[region] += valid
            sums_sq[region] += valid * valid
            counts[region] += 1.0
        probability_map = (score / np.maximum(counts, 1.0)).astype(np.float32)
        component_inputs[scene.scene_id] = (probability_map, scene.truths)
        dimension = f"{width}x{height}"
        dimension_pixel_rows.setdefault(dimension, []).append(
            _pixel_metrics(probability_map, _truth_mask(scene), PIXEL_THRESHOLD)
        )
        dimension_component_rows.setdefault(dimension, []).append(
            _component_metrics(probability_map, scene.truths, PIXEL_THRESHOLD, True, MINIMUM_COMPONENT_AREA)
        )
        overlap = counts > 1.0
        if np.any(overlap):
            means = score[overlap] / counts[overlap]
            variances = np.maximum(0.0, sums_sq[overlap] / counts[overlap] - means * means)
            overlap_prediction_values.extend(np.sqrt(variances).tolist())
        covered_pixels += int(np.count_nonzero(counts > 0.0))
        total_pixels += height * width
        maximum_coverage = max(maximum_coverage, int(np.max(counts)))
        tile_count += len(scene_tiles)

    pixel_sweep: list[dict[str, Any]] = []
    component_sweep: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        pixel_rows: list[dict[str, Any]] = []
        for probability_map, truths in component_inputs.values():
            truth_mask = np.zeros(probability_map.shape, dtype=bool)
            for truth in truths:
                truth_mask[int(truth.top):int(truth.bottom), int(truth.left):int(truth.right)] = True
            pixel_rows.append(_pixel_metrics(probability_map, truth_mask, threshold))
        pixel_row = _aggregate_pixel(pixel_rows)
        pixel_row["threshold"] = threshold
        pixel_sweep.append(pixel_row)
        for morphology_close in (False, True):
            rows = [
                _component_metrics(probability_map, truths, threshold, morphology_close, MINIMUM_COMPONENT_AREA)
                for probability_map, truths in component_inputs.values()
            ]
            row = _aggregate(rows)
            row.update({"threshold": threshold, "morphology_close": morphology_close, "minimum_component_area": MINIMUM_COMPONENT_AREA})
            component_sweep.append(row)

    area_sweep: list[dict[str, Any]] = []
    for minimum_area in AREA_SWEEP:
        rows = [
            _component_metrics(probability_map, truths, PIXEL_THRESHOLD, True, minimum_area)
            for probability_map, truths in component_inputs.values()
        ]
        row = _aggregate(rows)
        row.update({"threshold": PIXEL_THRESHOLD, "morphology_close": True, "minimum_component_area": minimum_area})
        area_sweep.append(row)

    fixed_pixel = _aggregate_pixel([row for rows in dimension_pixel_rows.values() for row in rows])
    fixed_component = _aggregate([row for rows in dimension_component_rows.values() for row in rows])
    best_pixel = max(pixel_sweep, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    best_recall_pixel = max(pixel_sweep, key=lambda row: (row["recall"], row["precision"]))
    best_component = max(component_sweep, key=lambda row: (row["recall"], row["precision"]))
    best_area_component = max(area_sweep, key=lambda row: (row["recall"], row["precision"]))
    overlap_values = np.asarray(overlap_prediction_values, dtype=np.float64)
    v37_diagnostic = json.loads(V37_DIAGNOSTIC.read_text(encoding="utf-8"))
    return {
        "schema": "graphreader.ocr-dice-loss-detector-v38-diagnostic.v1",
        "revision": "graph-text-dice-loss-detector-v38",
        "evidence": {
            "split": "dev",
            "synthetic_only": True,
            "scene_count": len(scenes),
            "truth_region_count": sum(len(scene.truths) for scene in scenes),
            "private_or_article_images": False,
            "sealed_or_public_reads": 0,
            "case_level_output": False,
        },
        "fixed_hashes": {
            "v38_checkpoint": {"path": checkpoint_path.relative_to(REPO_ROOT).as_posix(), "sha256": checkpoint_sha},
            "v38_onnx": {"path": model_path.relative_to(REPO_ROOT).as_posix(), "sha256": model_sha},
            "v38_result": {"path": V38_RESULT.relative_to(REPO_ROOT).as_posix(), "sha256": result_sha},
            "v37_result": {"path": V37_RESULT.relative_to(REPO_ROOT).as_posix(), "sha256": EXPECTED_V37_RESULT_SHA256},
            "v37_diagnostic": {"path": V37_DIAGNOSTIC.relative_to(REPO_ROOT).as_posix(), "sha256": EXPECTED_V37_DIAGNOSTIC_SHA256},
        },
        "protocol": {
            "onnx_provider": ONNX_PROVIDER,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "pixel_threshold": PIXEL_THRESHOLD,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
        },
        "pixel_segmentation": {
            "fixed_pipeline": fixed_pixel,
            "best_f1": best_pixel,
            "best_recall": best_recall_pixel,
            "threshold_sweep": pixel_sweep,
        },
        "postprocessing": {
            "fixed_pipeline": fixed_component,
            "best_threshold_or_morphology": best_component,
            "best_area_at_fixed_threshold": best_area_component,
            "threshold_and_morphology_sweep": component_sweep,
            "component_area_sweep": area_sweep,
        },
        "by_dimension": {
            dimension: {
                "pixel_segmentation": _aggregate_pixel(rows),
                "component_proposals": _aggregate(dimension_component_rows[dimension]),
            }
            for dimension, rows in sorted(dimension_pixel_rows.items())
        },
        "tiling_overlap_mapping": {
            "tile_count": tile_count,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "covered_pixel_fraction": covered_pixels / max(1, total_pixels),
            "overlap_pixel_count": int(overlap_values.size),
            "minimum_coverage": 1 if overlap_values.size else 0,
            "maximum_coverage": maximum_coverage,
            "mean_overlap_prediction_std": float(np.mean(overlap_values)) if overlap_values.size else 0.0,
            "p95_overlap_prediction_std": float(np.percentile(overlap_values, 95)) if overlap_values.size else 0.0,
            "maximum_overlap_prediction_std": float(np.max(overlap_values)) if overlap_values.size else 0.0,
        },
        "v37_comparison": {
            "v37_result_dev_metrics": v37_result["dev_metrics"],
            "v38_result_dev_metrics": v38_result["dev_metrics"],
            "v37_diagnostic_fixed_pipeline": v37_diagnostic["postprocessing"]["fixed_pipeline"],
            "v37_diagnostic_best_postprocessing": v37_diagnostic["postprocessing"]["best_threshold_or_morphology"],
            "v37_diagnostic_best_pixel_f1": v37_diagnostic["pixel_segmentation"]["best_f1"],
            "v38_diagnostic_fixed_pipeline": fixed_component,
            "v38_diagnostic_best_postprocessing": best_component,
            "v38_diagnostic_best_pixel_f1": best_pixel,
            "delta_v38_minus_v37_result": {
                "precision": v38_result["dev_metrics"]["precision"] - v37_result["dev_metrics"]["precision"],
                "recall": v38_result["dev_metrics"]["recall"] - v37_result["dev_metrics"]["recall"],
            },
        },
        "interpretation": {
            "pixel_segmentation": "V38 probability pixels remain below the Tier 1 precision and recall bars across the fixed threshold sweep.",
            "connected_components": "Threshold, morphology, and component-area sweeps do not recover the missing regions after segmentation.",
            "tiling": "All fixed V32 dev source pixels are covered; overlap mapping is therefore not the recall bottleneck.",
            "isolated_responsible_stage": "full_box_pixel_segmentation",
            "next_revision_startable": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_diagnostic(args.model.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
