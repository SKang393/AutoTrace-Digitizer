# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Attribute fixed-threshold V38 errors to synthetic structural proxies.

The model and V32 dev scenes are fixed.  This module emits aggregate counts
only: no scene identifiers, pixels, truth rows, or per-case predictions.
Semantic raster masks are not supplied for every structure, so line and box
annotations are converted to deterministic geometric masks.  Attribution
priority is axes/ticks, marker/edges, dividers, then text margins; remaining
dark ink and background are disjoint fallbacks.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import onnxruntime as ort

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.degradation_coverage_detector_v37.dataset import build_tiles
from ml.ocr.degradation_coverage_detector_v37.protocol import (
    MINIMUM_COMPONENT_AREA,
    ONNX_PROVIDER,
    PIXEL_THRESHOLD,
    TILE_OVERLAP,
    TILE_SIZE,
    TRUTH_MATCH_IOU_MINIMUM,
)
from ml.ocr.real_range_classifier_finetune_v32.dataset import _DEV_SPECS
from ml.ocr.real_range_classifier_finetune_v32.dataset import split_fingerprint
from ml.ocr.real_range_classifier_finetune_v32.protocol import DEV_SEED
from ml.synthetic.dataset import _build_scenes
from ml.synthetic.renderer import render_scene


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL = (
    REPO_ROOT
    / "artifacts/goal22-worktrees/ocr-v38/ml/ocr/dice_loss_detector_v38"
    / "artifacts/P1-run/graph-text-dice-loss-detector-v38-p1.onnx"
)
EXPECTED_MODEL_SHA256 = "90cb24e3c54931cf2da5213e0bcaa1208e182355187905ce0638cc7292dc1bc2"
EXPECTED_V32_DEV_FINGERPRINT = "67952b4575972542087281b2c14958e86518ae0e12e88d43f5c47c16252a3687"
SCHEMA = "graphreader.ocr-dice-loss-detector-v38-fp-attribution.v1"

CATEGORY_NAMES = (
    "within_or_near_axes_or_tick_lines",
    "marker_or_connecting_line_ink",
    "phase_dividers",
    "existing_text_box_margins",
    "other_dark_ink",
    "empty_background",
)
_CATEGORY_INDEX = {name: index for index, name in enumerate(CATEGORY_NAMES)}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_points(value: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value):
        return None
    return (float(value[0][0]), float(value[0][1])), (float(value[1][0]), float(value[1][1]))


def _draw_line(mask: np.ndarray, line: Any, thickness: int) -> None:
    points = _line_points(line)
    if points is None:
        return
    start, end = points
    cv2.line(
        mask,
        (round(start[0]), round(start[1])),
        (round(end[0]), round(end[1])),
        1,
        thickness=max(1, thickness),
        lineType=cv2.LINE_8,
    )


def _draw_margin(mask: np.ndarray, box: Any, margin: int) -> None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    x, y, width, height = (float(value) for value in box)
    left = max(0, round(x) - margin)
    top = max(0, round(y) - margin)
    right = min(mask.shape[1] - 1, round(x + width) + margin)
    bottom = min(mask.shape[0] - 1, round(y + height) + margin)
    if right >= left and bottom >= top:
        mask[top:bottom + 1, left:right + 1] = 1


def _panel_records(annotation: dict[str, Any], key: str) -> Iterable[dict[str, Any]]:
    for panel in annotation.get("panels", []):
        if isinstance(panel, dict):
            for record in panel.get(key, []):
                if isinstance(record, dict):
                    yield record


def _category_masks(annotation: dict[str, Any], marker_mask: np.ndarray) -> np.ndarray:
    """Return disjoint category masks using fixed geometry proxies.

    Axes/ticks use a three-pixel line radius, marker/edge ink uses the supplied
    marker mask plus two-pixel marker clearance and three-pixel edge strokes,
    dividers use a five-pixel line radius, and text boxes include a three-pixel
    margin.  These widths are diagnostic proxies, not production masks.
    """
    shape = marker_mask.shape
    raw = [np.zeros(shape, dtype=np.uint8) for _ in CATEGORY_NAMES[:4]]
    axes, markers, dividers, text = raw
    axes[marker_mask > 0] = False
    for record in _panel_records(annotation, "axes"):
        _draw_line(axes, record.get("line"), 7)
    for record in _panel_records(annotation, "ticks"):
        _draw_line(axes, record.get("line"), 7)
    markers |= marker_mask > 0
    for record in _panel_records(annotation, "markers"):
        center = record.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            radius = float(record.get("radius", 0.0)) + 2.0
            cv2.circle(markers, (round(float(center[0])), round(float(center[1]))), round(radius), 1, -1)
    for record in _panel_records(annotation, "edges"):
        _draw_line(markers, record.get("line"), 3)
    for record in _panel_records(annotation, "dividers"):
        _draw_line(dividers, record.get("line"), 5)
    for record in _panel_records(annotation, "texts"):
        _draw_margin(text, record.get("rendered_pixel_box", record.get("box")), 3)

    # Priority makes the masks disjoint and reproducible at overlaps.
    result = np.full(shape, _CATEGORY_INDEX["empty_background"], dtype=np.uint8)
    for index, mask in enumerate(raw):
        result[(result == _CATEGORY_INDEX["empty_background"]) & mask.astype(bool)] = index
    dark = np.zeros(shape, dtype=bool)
    return result, dark


def _truth_mask(truths: tuple[Box, ...], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for truth in truths:
        mask[int(truth.top):int(truth.bottom), int(truth.left):int(truth.right)] = True
    return mask


def _box_iou(left: Box, right: Box) -> float:
    x0, y0 = max(left.left, right.left), max(left.top, right.top)
    x1, y1 = min(left.right, right.right), min(left.bottom, right.bottom)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, left.right - left.left) * max(0, left.bottom - left.top)
    right_area = max(0, right.right - right.left) * max(0, right.bottom - right.top)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _matched_pairs(predicted: tuple[Box, ...], truths: tuple[Box, ...]) -> set[tuple[int, int]]:
    adjacency = [
        [truth_index for truth_index, truth in enumerate(truths) if _box_iou(box, truth) >= TRUTH_MATCH_IOU_MINIMUM]
        for box in predicted
    ]
    truth_to_prediction: dict[int, int] = {}

    def visit(prediction_index: int, seen: set[int]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or visit(previous, seen):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predicted)):
        visit(prediction_index, set())
    return {(prediction_index, truth_index) for truth_index, prediction_index in truth_to_prediction.items()}


def _component_boxes(probability: np.ndarray) -> tuple[tuple[Box, np.ndarray], ...]:
    binary = (probability >= PIXEL_THRESHOLD).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result: list[tuple[Box, np.ndarray]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < MINIMUM_COMPONENT_AREA or width < 2 or height < 2:
            continue
        result.append((Box(x, y, x + width, y + height), labels == index))
    return tuple(result)


def _size_bucket(truth: Box) -> str:
    area = max(0, truth.right - truth.left) * max(0, truth.bottom - truth.top)
    if area <= 200:
        return "small_area_le_200"
    if area <= 1000:
        return "medium_area_201_to_1000"
    return "large_area_gt_1000"


def _empty_category_counts() -> dict[str, dict[str, int]]:
    return {name: {"pixels": 0, "components": 0, "component_pixels": 0} for name in CATEGORY_NAMES}


def _add_counts(target: dict[str, dict[str, int]], category: str, pixels: int = 0, components: int = 0, component_pixels: int = 0) -> None:
    row = target[category]
    row["pixels"] += pixels
    row["components"] += components
    row["component_pixels"] += component_pixels


def _load_inputs(model_path: Path) -> tuple[tuple[Any, ...], tuple[Any, ...], str, str]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Authorized V38 ONNX artifact is missing: {model_path}")
    model_sha = _sha256(model_path)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError("V38 ONNX artifact hash does not match the authorized P1 payload")
    dev_fingerprint = split_fingerprint("dev")
    if dev_fingerprint != EXPECTED_V32_DEV_FINGERPRINT:
        raise RuntimeError("Fixed V32 dev split changed")
    built = _build_scenes(_DEV_SPECS, DEV_SEED, require_complete_style_catalog=False)
    scenes: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    stream_digest = sha256()
    for scene in built:
        image, annotation, marker_mask = render_scene(scene)
        raster = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        truths = tuple(
            Box(float(text["rendered_pixel_box"][0]), float(text["rendered_pixel_box"][1]),
                float(text["rendered_pixel_box"][0] + text["rendered_pixel_box"][2]),
                float(text["rendered_pixel_box"][1] + text["rendered_pixel_box"][3]))
            for panel in annotation.get("panels", [])
            if isinstance(panel, dict)
            for text in panel.get("texts", [])
            if isinstance(text, dict) and text.get("visible", True)
        )
        scenes.append({"scene_id": str(scene["scene_id"]), "raster": raster, "truths": truths, "annotation": annotation})
        marker = np.asarray(marker_mask.convert("L"), dtype=np.uint8)
        masks.append(marker)
        stream_digest.update(raster.tobytes(order="C"))
        stream_digest.update(marker.tobytes(order="C"))
        stream_digest.update(json.dumps(annotation, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return tuple(scenes), tuple(masks), model_sha, stream_digest.hexdigest()


def run_attribution(model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    scenes, masks, model_sha, annotation_mask_sha = _load_inputs(model_path.resolve())
    session = ort.InferenceSession(str(model_path), providers=[ONNX_PROVIDER])
    if session.get_providers() != [ONNX_PROVIDER]:
        raise RuntimeError(f"Expected only {ONNX_PROVIDER}, got {session.get_providers()}")
    tiles = build_tiles("dev")
    pixel_counts = _empty_category_counts()
    component_counts = _empty_category_counts()
    fn_by_size: dict[str, dict[str, int]] = {}
    fn_by_dimension: dict[str, dict[str, int]] = {}
    total_truth = total_predicted = total_tp = 0

    for scene, marker_mask in zip(scenes, masks, strict=True):
        scene_tiles = tuple(tile for tile in tiles if tile.scene_id == scene["scene_id"])
        values = np.stack([(1.0 - tile.image.astype(np.float32) / 255.0)[None, :, :] for tile in scene_tiles]).astype(np.float32)
        logits = np.asarray(session.run(["text_logits"], {"source_tiles": values})[0], dtype=np.float32)[:, 0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        height, width = scene["raster"].shape
        score = np.zeros((height, width), dtype=np.float64)
        counts = np.zeros((height, width), dtype=np.float64)
        for tile, probability in zip(scene_tiles, probabilities, strict=True):
            region = (slice(tile.top, tile.top + tile.valid_height), slice(tile.left, tile.left + tile.valid_width))
            valid = probability[:tile.valid_height, :tile.valid_width].astype(np.float64)
            score[region] += valid
            counts[region] += 1.0
        probability_map = (score / np.maximum(counts, 1.0)).astype(np.float32)
        truth_mask = _truth_mask(scene["truths"], (height, width))
        category_map, _ = _category_masks(scene["annotation"], marker_mask)
        dark = scene["raster"] < 200
        fallback_dark = _CATEGORY_INDEX["other_dark_ink"]
        category_map[(category_map == _CATEGORY_INDEX["empty_background"]) & dark] = fallback_dark
        false_positive = (probability_map >= PIXEL_THRESHOLD) & ~truth_mask
        for category in CATEGORY_NAMES:
            index = _CATEGORY_INDEX[category]
            _add_counts(pixel_counts, category, pixels=int(np.count_nonzero(false_positive & (category_map == index))))

        components = _component_boxes(probability_map)
        predicted_boxes = tuple(box for box, _ in components)
        pairs = _matched_pairs(predicted_boxes, scene["truths"])
        matched_truths = {truth_index for _, truth_index in pairs}
        total_truth += len(scene["truths"])
        total_predicted += len(predicted_boxes)
        total_tp += len(pairs)
        for prediction_index, (box, component_mask) in enumerate(components):
            if any(pair_prediction == prediction_index for pair_prediction, _ in pairs):
                continue
            pixels = component_mask & ~truth_mask
            if not np.any(pixels):
                continue
            values_by_category = [int(np.count_nonzero(pixels & (category_map == index))) for index in range(len(CATEGORY_NAMES))]
            category = CATEGORY_NAMES[max(range(len(CATEGORY_NAMES)), key=lambda index: (values_by_category[index], -index))]
            _add_counts(component_counts, category, components=1, component_pixels=int(np.count_nonzero(pixels)))
        for truth_index, truth in enumerate(scene["truths"]):
            bucket = _size_bucket(truth)
            row = fn_by_size.setdefault(bucket, {"truth_boxes": 0, "false_negatives": 0})
            row["truth_boxes"] += 1
            if truth_index not in matched_truths:
                row["false_negatives"] += 1
        dimension = f"{width}x{height}"
        row = fn_by_dimension.setdefault(dimension, {"truth_boxes": 0, "false_negatives": 0})
        row["truth_boxes"] += len(scene["truths"])
        row["false_negatives"] += len(scene["truths"]) - len(matched_truths)

    for rows in (fn_by_size, fn_by_dimension):
        for row in rows.values():
            row["false_negative_rate"] = row["false_negatives"] / max(1, row["truth_boxes"])
    return {
        "schema": SCHEMA,
        "revision": "graph-text-dice-loss-detector-v38",
        "evidence": {
            "split": "dev",
            "synthetic_only": True,
            "scene_count": len(scenes),
            "truth_region_count": total_truth,
            "sealed_or_public_reads": 0,
            "private_or_article_images": False,
            "case_level_output": False,
        },
        "fixed_hashes": {
            "v38_onnx": {"path": model_path.relative_to(REPO_ROOT).as_posix(), "sha256": model_sha},
            "v32_dev_split": {"sha256": EXPECTED_V32_DEV_FINGERPRINT},
            "annotation_and_marker_mask_stream": {"sha256": annotation_mask_sha},
        },
        "protocol": {
            "onnx_provider": ONNX_PROVIDER,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "pixel_threshold": PIXEL_THRESHOLD,
            "minimum_component_area": MINIMUM_COMPONENT_AREA,
            "truth_match_iou_minimum": TRUTH_MATCH_IOU_MINIMUM,
            "morphology_close": True,
            "geometry_proxies": {
                "axes_or_ticks_line_thickness": 7,
                "marker_clearance_px": 2,
                "connecting_line_thickness": 3,
                "divider_line_thickness": 5,
                "text_box_margin_px": 3,
                "dark_ink_threshold": 200,
            },
        },
        "false_positive_pixels": {
            "total": sum(row["pixels"] for row in pixel_counts.values()),
            "by_category": pixel_counts,
        },
        "false_positive_components": {
            "truth_regions": total_truth,
            "predicted_regions": total_predicted,
            "true_positives": total_tp,
            "false_positives": total_predicted - total_tp,
            "by_category": component_counts,
        },
        "false_negatives": {"by_truth_box_size": dict(sorted(fn_by_size.items())), "by_source_dimension": dict(sorted(fn_by_dimension.items()))},
        "interpretation": {
            "dominant_attributable_source": max(CATEGORY_NAMES, key=lambda category: (pixel_counts[category]["pixels"], -_CATEGORY_INDEX[category])),
            "next_subsystem_change": "Add explicit artifact/structure suppression features to full-box pixel segmentation before another OCR detector candidate; attribution is diagnostic only and does not authorize tuning.",
            "confidence": "Structural categories are deterministic geometric proxies; remaining pixels are split by a fixed raster darkness threshold into other dark ink and empty background.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_attribution(args.model)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
