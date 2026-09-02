# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate-only diagnosis of the real-dev Engauge corpus split."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import os
import struct
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from PIL import Image

# Make the documented direct-file CLI work from the repository root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.synthetic.renderer import production_resize_dimensions
from tools.private_acceptance.real_corpus import assign_splits


MAX_XML_BYTES = 64 * 1024 * 1024
MAX_EMBEDDED_IMAGE_BYTES = 64 * 1024 * 1024
RAY_COUNT = 72
RAY_STEP_PX = 0.5
MAX_MARKER_RADIUS_PX = 24.0
DARK_PIXEL_THRESHOLD = 160
MINIMUM_DARK_ANGULAR_FRACTION = 0.65


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _embedded_png_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if len(raw) > MAX_XML_BYTES:
        raise ValueError("DIG_XML_TOO_LARGE")
    declaration = raw[:8192].upper()
    if b"<!ENTITY" in declaration or b" SYSTEM " in declaration or b" PUBLIC " in declaration:
        raise ValueError("DIG_EXTERNAL_ENTITY_FORBIDDEN")
    root = ET.fromstring(raw)
    for element in root.iter():
        if "image" not in _local(element) and "raster" not in _local(element):
            continue
        text = "".join((element.text or "").split())
        if not text:
            continue
        try:
            payload = base64.b64decode(text, validate=True)
        except Exception as error:
            raise ValueError("DIG_EMBEDDED_IMAGE_INVALID") from error
        if not payload or len(payload) > MAX_EMBEDDED_IMAGE_BYTES:
            raise ValueError("DIG_EMBEDDED_IMAGE_INVALID")
        if len(payload) > 4 and payload[4:].startswith(b"\x89PNG\r\n\x1a\n"):
            payload = payload[4:]
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("DIG_EMBEDDED_IMAGE_NOT_PNG")
        return payload
    raise ValueError("DIG_EMBEDDED_IMAGE_MISSING")


def _load_embedded_png(path: Path) -> Image.Image:
    payload = _embedded_png_bytes(path)
    with Image.open(BytesIO(payload)) as image:
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        return image.copy()


def _assignment_hash(paths: Sequence[Path], assignments: dict[Path, str], root: Path) -> str:
    material = "\n".join(
        f"{hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()}={assignments[path]}"
        for path in sorted(paths)
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _effective_marker_diameters(image: Image.Image, points: Sequence[Any]) -> list[float]:
    gray = image.convert("L")
    width, height = gray.size
    diameters: list[float] = []
    for point in points:
        center_x = float(point.screen_x)
        center_y = float(point.screen_y)
        radius = RAY_STEP_PX
        outer_radius: float | None = None
        while radius <= MAX_MARKER_RADIUS_PX:
            dark_count = 0
            for angle_index in range(RAY_COUNT):
                angle = 6.283185307179586 * angle_index / RAY_COUNT
                x = min(width - 1, max(0, round(center_x + radius * math.cos(angle))))
                y = min(height - 1, max(0, round(center_y + radius * math.sin(angle))))
                if gray.getpixel((x, y)) < DARK_PIXEL_THRESHOLD:
                    dark_count += 1
            if dark_count / RAY_COUNT >= MINIMUM_DARK_ANGULAR_FRACTION:
                outer_radius = radius
            radius += RAY_STEP_PX
        if outer_radius is not None:
            diameters.append(2.0 * outer_radius)
    return diameters


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def diagnose(root: Path, *, sealed_target: int = 51, expected_real_dev: int | None = None) -> dict[str, Any]:
    """Read only the deterministic real-dev assignment and emit aggregates."""
    root = root.resolve(strict=True)
    paths = sorted(path.resolve() for path in root.rglob("*.dig"))
    assignments = assign_splits(paths, root, sealed_target=sealed_target)
    dev_paths = [path for path in paths if assignments[path] == "real-dev"]
    sealed_paths = [path for path in paths if assignments[path] == "real-sealed"]
    if expected_real_dev is not None and len(dev_paths) != expected_real_dev:
        raise ValueError(f"REAL_DEV_COUNT:{len(dev_paths)}")

    widths: list[int] = []
    heights: list[int] = []
    scales: list[float] = []
    modes: Counter[str] = Counter()
    diameters: list[float] = []
    truth_marker_count = 0
    for path in dev_paths:
        before = _sha256(path)
        from tools.private_acceptance.real_corpus import parse_dig

        truth = parse_dig(path)
        truth_marker_count += len(truth.points)
        image = _load_embedded_png(path)
        if before != _sha256(path):
            raise RuntimeError("REAL_DEV_SOURCE_MUTATED")
        widths.append(image.width)
        heights.append(image.height)
        modes[image.mode] += 1
        scales.append(float(production_resize_dimensions(image.width, image.height)["scale"]))
        diameters.extend(_effective_marker_diameters(image, truth.points))

    if not widths or not diameters:
        raise ValueError("REAL_DEV_NO_MEASURABLE_DATA")
    result = {
        "schema_version": 1,
        "report_scope": "real_dev_aggregate_only",
        "real_dev": len(dev_paths),
        "real_sealed_reads": 0,
        "optimizer_steps": 0,
        "model_inference_runs": 0,
        "candidate_selection": 0,
        "assignment_sha256": _assignment_hash(paths, assignments, root),
        "source_dimensions": {
            "width_px": [min(widths), int(median(widths)), max(widths)],
            "height_px": [min(heights), int(median(heights)), max(heights)],
        },
        "source_modes": dict(sorted(modes.items())),
        "production_resize_scale_before_padding": [min(scales), median(scales), max(scales)],
        "effective_marker_diameter_px": {
            "minimum": min(diameters),
            "p10": _percentile(diameters, 0.10),
            "median": median(diameters),
            "p90": _percentile(diameters, 0.90),
            "maximum": max(diameters),
        },
        "measured_marker_count": len(diameters),
        "truth_marker_count": truth_marker_count,
        "measurement_coverage": len(diameters) / truth_marker_count,
        "real_sealed_project_count": len(sealed_paths),
        "case_level_output": False,
        "study_identifiers_output": False,
        "truth_rows_output": False,
        "pixel_output": False,
        "privacy_status": "private",
        "training_use": False,
        "selection_use": False,
        "git_eligible": False,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate-only real-dev distribution diagnosis")
    parser.add_argument("root", type=Path)
    parser.add_argument("--explicit-opt-in", action="store_true")
    args = parser.parse_args(argv)
    if not args.explicit_opt_in:
        raise SystemExit("PRIVATE_CORPUS_EXPLICIT_OPT_IN_REQUIRED")
    ci_names = ("CI", "TF_BUILD", "GITHUB_ACTIONS", "GITLAB_CI", "BUILD_BUILDID", "JENKINS_URL", "TEAMCITY_VERSION")
    if any(os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no", "off"} for name in ci_names):
        raise SystemExit("PRIVATE_CORPUS_DISABLED_IN_CI")
    print(json.dumps(diagnose(args.root, expected_real_dev=120), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
