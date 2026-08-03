# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Run the fixed public-synthetic Session 07 GraphSR experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ml.synthetic.renderer import render_scene
from ml.synthetic.templates import build_scene

from .dataset import PairedTrainingSample, build_training_pairs
from .degradation import CoordinateTransform
from .export import export_onnx
from .model import ensure_artifact_outside_repository
from .train import DEFAULT_SEED, train


EXPERIMENT_ID = "session-07-graphsr-x2-public-synthetic-v1"
TRAINING_SCENES = (
    ("ab", 20260811, "vector_clean", 1, 24),
    ("abab", 20260812, "vector_clean", 1, 28),
    ("multiple_baseline", 20260813, "print_monochrome", 3, 24),
    ("alternating_treatments", 20260814, "print_monochrome", 1, 32),
)
HELDOUT_SCENES = (
    ("staggered_starts", 20260851, "hand_drawn", 3, 32),
    ("staggered_starts", 20260861, "hand_drawn", 3, 36),
)
OFFICIAL_BLOCKERS = {
    "RealESRGAN_x2plus": (
        "Official PyTorch weights are provenance-verified, but no approved local "
        "Windows runtime is configured for this fixed benchmark"
    ),
    "realesr-general-x4v3": (
        "Official PyTorch weights are provenance-verified, but no approved local "
        "Windows runtime is configured for outscale 2"
    ),
    "realesr-animevideov3": (
        "The NCNN x2 payload is provenance-verified, but the NCNN runtime and "
        "transitive redistribution notices are not approved for local execution"
    ),
}


def _records(annotation: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    values = [item for item in annotation.get(key, ()) if isinstance(item, Mapping)]
    for panel in annotation.get("panels", ()):
        if isinstance(panel, Mapping):
            values.extend(item for item in panel.get(key, ()) if isinstance(item, Mapping))
    return tuple(values)


def _transform_box(
    transform: CoordinateTransform,
    box: Sequence[float],
) -> list[float]:
    if len(box) != 4:
        raise ValueError("A text box must contain x, y, width, and height")
    x, y, width, height = (float(value) for value in box)
    corners = (
        transform.apply((x, y)),
        transform.apply((x + width, y)),
        transform.apply((x, y + height)),
        transform.apply((x + width, y + height)),
    )
    left = min(point[0] for point in corners)
    top = min(point[1] for point in corners)
    right = max(point[0] for point in corners)
    bottom = max(point[1] for point in corners)
    return [left, top, right - left, bottom - top]


def _inside_box(box: Sequence[float], width: int, height: int) -> bool:
    x, y, box_width, box_height = box
    return x + box_width > 0 and y + box_height > 0 and x < width and y < height


def _transformed_radius(
    transform: CoordinateTransform,
    center: Sequence[float],
    radius: float,
) -> float:
    mapped = transform.apply(center)
    horizontal = transform.apply((float(center[0]) + radius, float(center[1])))
    vertical = transform.apply((float(center[0]), float(center[1]) + radius))
    return max(
        0.75,
        (
            math.hypot(horizontal[0] - mapped[0], horizontal[1] - mapped[1])
            + math.hypot(vertical[0] - mapped[0], vertical[1] - mapped[1])
        )
        / 2.0,
    )


def _ocr_mask(
    annotation: Mapping[str, Any],
    sample: PairedTrainingSample,
) -> np.ndarray:
    height, width = sample.hr.shape[:2]
    image = Image.new("L", (width, height), 0)
    drawing = ImageDraw.Draw(image)
    for text in _records(annotation, "texts"):
        if text.get("visible") is not True or text.get("role") not in ("x_tick", "y_tick"):
            continue
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(text.get("text", ""))) is None:
            continue
        box = _transform_box(sample.source_to_hr, text["box"])
        if _inside_box(box, width, height):
            x, y, box_width, box_height = box
            drawing.rectangle((x, y, x + box_width, y + box_height), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def _training_samples() -> tuple[dict[str, object], ...]:
    samples: list[dict[str, object]] = []
    for design, seed, renderer, panels, sessions in TRAINING_SCENES:
        scene = build_scene(
            design,
            seed,
            renderer,
            panel_count=panels,
            session_count=sessions,
        )
        image, annotation, _ = render_scene(scene)
        pairs = build_training_pairs(
            image,
            annotation,
            seed=seed + 10_000,
            crop_size=(96, 96),
            count=4,
        )
        for pair in pairs:
            samples.append(
                {
                    "sample_id": pair.sample_id,
                    "lr": pair.lr,
                    "hr": pair.hr,
                    "marker_centers_hr": pair.marker_centers_hr,
                    "ocr_mask": _ocr_mask(annotation, pair),
                    "metadata": pair.metadata,
                }
            )
    return tuple(samples)


def _benchmark_annotations(
    annotation: Mapping[str, Any],
    sample: PairedTrainingSample,
) -> dict[str, list[object]]:
    height, width = sample.hr.shape[:2]
    markers: list[dict[str, object]] = []
    open_markers: list[dict[str, object]] = []
    for marker in _records(annotation, "markers"):
        center = sample.source_to_hr.apply(marker["center"])
        if not 0 <= center[0] < width or not 0 <= center[1] < height:
            continue
        radius = _transformed_radius(
            sample.source_to_hr,
            marker["center"],
            float(marker.get("radius", 4.0)),
        )
        value = {"center": [center[0], center[1]], "radius": radius}
        markers.append(value)
        if marker.get("fill") == "open" and marker.get("shape") not in ("cross", "asterisk"):
            open_markers.append(value)

    axis_lines: list[list[float]] = []
    for axis in _records(annotation, "axes"):
        if axis.get("visible") is not True:
            continue
        line = axis.get("line")
        if not isinstance(line, Sequence) or len(line) != 2:
            continue
        start = sample.source_to_hr.apply(line[0])
        end = sample.source_to_hr.apply(line[1])
        axis_lines.append([start[0], start[1], end[0], end[1]])

    ocr_regions: list[dict[str, object]] = []
    for text in _records(annotation, "texts"):
        if text.get("visible") is not True or text.get("role") not in ("x_tick", "y_tick"):
            continue
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(text.get("text", ""))) is None:
            continue
        box = _transform_box(sample.source_to_hr, text["box"])
        if _inside_box(box, width, height):
            ocr_regions.append({"box": box, "expected_text": str(text["text"])})

    if not markers or not open_markers or not axis_lines or not ocr_regions:
        raise RuntimeError("A held-out scene did not cover every structural benchmark annotation")
    return {
        "ocr_regions": ocr_regions,
        "marker_centers": markers,
        "axis_lines": axis_lines,
        "open_markers": open_markers,
    }


def _write_heldout_cases(output: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    dataset_root = output / "heldout"
    dataset_root.mkdir(parents=True, exist_ok=True)
    for index, (design, seed, renderer, panels, sessions) in enumerate(HELDOUT_SCENES, start=1):
        scene = build_scene(
            design,
            seed,
            renderer,
            panel_count=panels,
            session_count=sessions,
        )
        image, annotation, _ = render_scene(scene)
        width, height = image.size
        pair = build_training_pairs(
            image,
            annotation,
            seed=seed + 20_000,
            crop_size=(width, height),
            count=1,
        )[0]
        input_path = dataset_root / f"heldout-{index:02d}-lr.png"
        truth_path = dataset_root / f"heldout-{index:02d}-hr.png"
        Image.fromarray(pair.lr).save(input_path, format="PNG", optimize=False)
        Image.fromarray(pair.hr).save(truth_path, format="PNG", optimize=False)
        cases.append(
            {
                "case_id": f"heldout-{index:02d}",
                "input_path": input_path.relative_to(output).as_posix(),
                "ground_truth_path": truth_path.relative_to(output).as_posix(),
                "annotations": _benchmark_annotations(annotation, pair),
            }
        )
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(output: Path, *, steps: int = 100) -> dict[str, object]:
    output = ensure_artifact_outside_repository(output)
    output.mkdir(parents=True, exist_ok=True)
    training_samples = _training_samples()
    checkpoint, training_report = train(
        output / "model",
        seed=DEFAULT_SEED,
        epochs=100,
        max_steps=steps,
        batch_size=4,
        samples=training_samples,
    )
    onnx_path = output / "model" / "graphsr-x2.onnx"
    parity_path = output / "model" / "onnx-parity.json"
    parity_report = export_onnx(checkpoint, onnx_path, parity_path)
    cases = _write_heldout_cases(output)
    manifest = {
        "benchmark_manifest_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "dataset": {
            "dataset_id": "public-synthetic-hand-drawn-heldout-v1",
            "cases": cases,
        },
        "candidates": [
            {
                "model_id": model_id,
                "status": "blocked",
                "reason": reason,
            }
            for model_id, reason in OFFICIAL_BLOCKERS.items()
        ]
        + [
            {
                "model_id": "GraphSR-x2",
                "model_manifest_path": "models/manifest/graphsr/graphsr-x2-candidate-0.1.0.json",
                "runtime": {
                    "kind": "onnx",
                    "model_path": onnx_path.relative_to(output).as_posix(),
                    "sha256": _sha256(onnx_path),
                    "provider": "cpu",
                    "input_name": "image",
                    "output_name": "enhanced",
                },
            }
        ],
        "selection": {
            "thresholds": {
                "numeric_ocr_exact_match_min": 0.95,
                "marker_center_f1_min": 1.0,
                "shape_fill_classification_f1_min": 0.95,
                "axis_thin_line_recall_min": 0.98,
                "open_marker_preservation_rate_min": 1.0,
                "marker_center_mean_error_pixels_max": 0.25,
                "axis_line_localization_error_pixels_max": 0.25,
                "hallucinated_structure_rate_max": 0.01,
                "runtime_ms_mean_max": 5000.0,
                "peak_memory_bytes_max": 2147483648,
            }
        },
    }
    manifest_path = output / "benchmark-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "training_sample_count": len(training_samples),
        "training_dataset_identity": training_report["dataset_identity"],
        "steps_completed": training_report["steps_completed"],
        "checkpoint_sha256": training_report["checkpoint_sha256"],
        "onnx_sha256": parity_report["onnx_sha256"],
        "onnx_parity_maximum_absolute_error": parity_report["maximum_absolute_error"],
        "onnx_parity_tolerance": parity_report["tolerance"],
        "heldout_case_count": len(cases),
        "benchmark_manifest": str(manifest_path),
    }
    (output / "experiment-report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    arguments = parser.parse_args(tuple(argv) if argv is not None else None)
    report = run_experiment(arguments.output, steps=arguments.steps)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EXPERIMENT_ID", "HELDOUT_SCENES", "TRAINING_SCENES", "run_experiment"]
