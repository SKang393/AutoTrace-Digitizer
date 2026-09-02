# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate-only reader for the local Engauge Digitizer corpus.

This module deliberately keeps the corpus on the caller's machine.  It parses
the small amount of geometry needed by validation, but its public inventory
contains counts and a split fingerprint only.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import struct
import sys
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET


MAX_XML_BYTES = 64 * 1024 * 1024
MAX_EMBEDDED_IMAGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ImageMetadata:
    width: int | None
    height: int | None
    format: str | None
    mode: str | None


@dataclass(frozen=True)
class AxisAnchor:
    screen_x: float
    screen_y: float
    graph_x: float
    graph_y: float


@dataclass(frozen=True)
class CurvePoint:
    screen_x: float
    screen_y: float


@dataclass(frozen=True)
class DigTruth:
    image: ImageMetadata
    anchors: tuple[AxisAnchor, ...]
    points: tuple[CurvePoint, ...]


@dataclass(frozen=True)
class CorpusInventory:
    schema_version: int
    study_directory_count: int
    project_count: int
    study_count_with_projects: int
    axis_anchor_count: int
    digitized_point_count: int
    image_width_minimum: int
    image_width_median: int
    image_width_maximum: int
    image_height_minimum: int
    image_height_median: int
    image_height_maximum: int
    image_modes: dict[str, int]
    image_formats: dict[str, int]
    real_dev_count: int
    real_sealed_count: int
    assignment_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_directory_count": self.study_directory_count,
            "project_count": self.project_count,
            "study_count_with_projects": self.study_count_with_projects,
            "axis_anchor_count": self.axis_anchor_count,
            "digitized_point_count": self.digitized_point_count,
            "image_width": {
                "minimum": self.image_width_minimum,
                "median": self.image_width_median,
                "maximum": self.image_width_maximum,
            },
            "image_height": {
                "minimum": self.image_height_minimum,
                "median": self.image_height_median,
                "maximum": self.image_height_maximum,
            },
            "image_modes": self.image_modes,
            "image_formats": self.image_formats,
            "real_dev_count": self.real_dev_count,
            "real_sealed_count": self.real_sealed_count,
            "assignment_sha256": self.assignment_sha256,
            "report_scope": "aggregate_only",
            "privacy_status": "private",
            "training_use": False,
            "git_eligible": False,
            "redistribution_authorized": False,
            "case_level_output": False,
            "study_identifiers_output": False,
            "truth_rows_output": False,
            "pixel_output": False,
        }


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _number(element: ET.Element, names: Sequence[str]) -> float | None:
    wanted = {name.lower() for name in names}
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1].lower() in wanted:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _pair(element: ET.Element) -> tuple[float, float] | None:
    x = _number(element, ("x", "positionx", "screenx", "graphx"))
    y = _number(element, ("y", "positiony", "screeny", "graphy"))
    return None if x is None or y is None else (x, y)


def _descendant(element: ET.Element, names: set[str]) -> ET.Element | None:
    for child in element.iter():
        if child is not element and _local(child) in names:
            return child
    return None


def _image_from_bytes(value: bytes) -> ImageMetadata:
    if len(value) > 4 and value[4:].startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8")):
        value = value[4:]
    if value.startswith(b"\x89PNG\r\n\x1a\n") and len(value) >= 26:
        width, height = struct.unpack(">II", value[16:24])
        color_type = value[25]
        mode = {0: "L", 2: "RGB", 3: "P", 4: "LA", 6: "RGBA"}.get(color_type)
        return ImageMetadata(width, height, "PNG", mode)
    if value.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(value):
            if value[index] != 0xFF:
                index += 1
                continue
            marker = value[index + 1]
            index += 2
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(value):
                break
            length = struct.unpack(">H", value[index:index + 2])[0]
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                if index + 7 <= len(value):
                    height, width = struct.unpack(">HH", value[index + 3:index + 7])
                    return ImageMetadata(width, height, "JPEG", "RGB")
            index += max(length, 2)
    return ImageMetadata(None, None, None, None)


def _embedded_image(root: ET.Element) -> ImageMetadata:
    for element in root.iter():
        name = _local(element)
        if "image" not in name and "raster" not in name:
            continue
        declared_width = _number(element, ("width",))
        declared_height = _number(element, ("height",))
        declared_format = next((str(v) for k, v in element.attrib.items() if k.lower() in {"format", "mime", "mimetype"}), None)
        text = "".join((element.text or "").split())
        if text:
            try:
                raw = base64.b64decode(text, validate=True)
            except Exception as error:
                raise ValueError("DIG_EMBEDDED_IMAGE_INVALID") from error
            if not raw or len(raw) > MAX_EMBEDDED_IMAGE_BYTES:
                raise ValueError("DIG_EMBEDDED_IMAGE_INVALID")
            detected = _image_from_bytes(raw)
            if detected.width is None or detected.height is None:
                raise ValueError("DIG_EMBEDDED_IMAGE_INVALID")
            if (
                declared_width is not None and int(declared_width) != detected.width
            ) or (
                declared_height is not None and int(declared_height) != detected.height
            ):
                raise ValueError("DIG_IMAGE_DIMENSION_MISMATCH")
            return detected
        raise ValueError("DIG_EMBEDDED_IMAGE_MISSING")
    raise ValueError("DIG_EMBEDDED_IMAGE_MISSING")


def parse_dig(path: Path) -> DigTruth:
    """Parse one Engauge XML project, retaining only geometry and image metadata."""
    raw = path.read_bytes()
    if len(raw) > MAX_XML_BYTES:
        raise ValueError("DIG_XML_TOO_LARGE")
    declaration = raw[:8192].upper()
    if b"<!ENTITY" in declaration or b" SYSTEM " in declaration or b" PUBLIC " in declaration:
        raise ValueError("DIG_EXTERNAL_ENTITY_FORBIDDEN")
    root = ET.fromstring(raw)
    anchors: list[AxisAnchor] = []
    point_candidates: list[CurvePoint] = []
    for element in root.iter():
        if _local(element) not in {"point", "datapoint", "curvepoint", "coordinate"}:
            continue
        screen = _descendant(element, {"positionscreen", "screenposition"})
        graph = _descendant(element, {"positiongraph", "graphposition"})
        if screen is not None and graph is not None:
            screen_pair = _pair(screen)
            graph_pair = _pair(graph)
            if screen_pair is not None and graph_pair is not None:
                anchors.append(AxisAnchor(*screen_pair, *graph_pair))
        elif screen is not None:
            pair = _pair(screen)
            if pair is not None:
                point_candidates.append(CurvePoint(*pair))
    if len(anchors) != 3:
        raise ValueError(f"DIG_AXIS_ANCHOR_COUNT:{len(anchors)}")
    return DigTruth(_embedded_image(root), tuple(anchors), tuple(point_candidates))


def _study_key(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    study = relative.parts[0] if len(relative.parts) > 1 else "__root__"
    return hashlib.sha256(study.encode("utf-8")).hexdigest()


def assign_splits(paths: Iterable[Path], root: Path, sealed_target: int = 51) -> dict[Path, str]:
    """Assign whole studies deterministically, approaching the sealed target."""
    projects = sorted(path.resolve() for path in paths)
    groups: dict[str, list[Path]] = {}
    for path in projects:
        groups.setdefault(_study_key(path, root), []).append(path)
    ordered = sorted(groups.items(), key=lambda item: item[0])
    sealed: set[Path] = set()
    count = 0
    for _, members in ordered:
        if count < sealed_target and (count + len(members) <= sealed_target or not sealed):
            sealed.update(members)
            count += len(members)
    return {path: ("real-sealed" if path in sealed else "real-dev") for path in projects}


def inventory(root: Path, sealed_target: int = 51) -> CorpusInventory:
    root = root.resolve(strict=True)
    paths = sorted(root.rglob("*.dig"))
    assignments = assign_splits(paths, root, sealed_target)
    study_keys = {_study_key(path, root) for path in paths}
    truths = [parse_dig(path) for path in paths]
    widths = [truth.image.width for truth in truths if truth.image.width is not None]
    heights = [truth.image.height for truth in truths if truth.image.height is not None]
    if len(widths) != len(paths) or len(heights) != len(paths):
        raise ValueError("DIG_IMAGE_DIMENSIONS_MISSING")
    modes = Counter(truth.image.mode or "unknown" for truth in truths)
    formats = Counter(truth.image.format or "unknown" for truth in truths)
    assignment_material = "\n".join(
        f"{hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()}={assignments[path]}"
        for path in paths
    ).encode("utf-8")
    return CorpusInventory(
        schema_version=1,
        study_directory_count=sum(path.is_dir() for path in root.iterdir()),
        project_count=len(paths),
        study_count_with_projects=len(study_keys),
        axis_anchor_count=sum(len(truth.anchors) for truth in truths),
        digitized_point_count=sum(len(truth.points) for truth in truths),
        image_width_minimum=min(widths),
        image_width_median=int(median(widths)),
        image_width_maximum=max(widths),
        image_height_minimum=min(heights),
        image_height_median=int(median(heights)),
        image_height_maximum=max(heights),
        image_modes=dict(sorted(modes.items())),
        image_formats=dict(sorted(formats.items())),
        real_dev_count=sum(value == "real-dev" for value in assignments.values()),
        real_sealed_count=sum(value == "real-sealed" for value in assignments.values()),
        assignment_sha256=hashlib.sha256(assignment_material).hexdigest(),
    )


def axis_pixel_error(predicted: Sequence[AxisAnchor], truth: Sequence[AxisAnchor]) -> dict[str, float | int]:
    if len(predicted) != len(truth) or not truth:
        raise ValueError("AXIS_ANCHOR_LENGTH_MISMATCH")
    predicted_by_graph = {(item.graph_x, item.graph_y): item for item in predicted}
    truth_by_graph = {(item.graph_x, item.graph_y): item for item in truth}
    if len(predicted_by_graph) != len(predicted) or predicted_by_graph.keys() != truth_by_graph.keys():
        raise ValueError("AXIS_GRAPH_ANCHOR_MISMATCH")
    errors = [
        ((predicted_by_graph[key].screen_x - item.screen_x) ** 2 +
         (predicted_by_graph[key].screen_y - item.screen_y) ** 2) ** 0.5
        for key, item in truth_by_graph.items()
    ]
    return {"anchor_count": len(errors), "mean_error_px": sum(errors) / len(errors), "maximum_error_px": max(errors)}


def _maximum_matches(edges: Sequence[Sequence[int]], truth_count: int) -> int:
    matched_prediction_for_truth = [-1] * truth_count

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for truth_index in edges[prediction_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            prior = matched_prediction_for_truth[truth_index]
            if prior < 0 or augment(prior, visited):
                matched_prediction_for_truth[truth_index] = prediction_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(edges)))


def marker_center_precision_recall(predicted: Sequence[tuple[float, float]], truth: Sequence[tuple[float, float]], tolerance_px: float) -> dict[str, float | int]:
    if tolerance_px < 0:
        raise ValueError("NEGATIVE_TOLERANCE")
    edges = [
        [
            index for index, (tx, ty) in sorted(
                enumerate(truth),
                key=lambda item: ((px - item[1][0]) ** 2 + (py - item[1][1]) ** 2),
            )
            if ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5 <= tolerance_px
        ]
        for px, py in predicted
    ]
    true_positives = _maximum_matches(edges, len(truth))
    false_positives = len(predicted) - true_positives
    false_negatives = len(truth) - true_positives
    return {"true_positives": true_positives, "false_positives": false_positives, "false_negatives": false_negatives, "precision": true_positives / len(predicted) if predicted else 0.0, "recall": true_positives / len(truth) if truth else 1.0}


def export_y_accuracy(predicted: Sequence[tuple[float, float]], truth: Sequence[tuple[float, float]], tolerance: float = 5.0) -> dict[str, float | int]:
    if tolerance < 0:
        raise ValueError("NEGATIVE_TOLERANCE")
    truth_by_x: dict[int, list[float]] = {}
    predicted_by_x: dict[int, list[float]] = {}
    for x, y in truth:
        truth_by_x.setdefault(round(x), []).append(y)
    for x, y in predicted:
        predicted_by_x.setdefault(round(x), []).append(y)
    matched = 0
    within = 0
    for x_value in predicted_by_x.keys() & truth_by_x.keys():
        predicted_values = predicted_by_x[x_value]
        truth_values = truth_by_x[x_value]
        matched += min(len(predicted_values), len(truth_values))
        edges = [
            [index for index, expected in enumerate(truth_values) if abs(value - expected) <= tolerance]
            for value in predicted_values
        ]
        within += _maximum_matches(edges, len(truth_values))
    return {"matched": matched, "within_tolerance": within, "accuracy": within / matched if matched else 0.0, "unmatched_predictions": len(predicted) - matched, "unmatched_truth": len(truth) - matched}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate-only real corpus inventory")
    parser.add_argument("root", type=Path)
    parser.add_argument("--explicit-opt-in", action="store_true")
    args = parser.parse_args(argv)
    if not args.explicit_opt_in:
        raise SystemExit("PRIVATE_CORPUS_EXPLICIT_OPT_IN_REQUIRED")
    ci_names = ("CI", "TF_BUILD", "GITHUB_ACTIONS", "GITLAB_CI", "BUILD_BUILDID", "JENKINS_URL", "TEAMCITY_VERSION")
    if any(os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no", "off"} for name in ci_names):
        raise SystemExit("PRIVATE_CORPUS_DISABLED_IN_CI")
    print(json.dumps(inventory(args.root).as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
