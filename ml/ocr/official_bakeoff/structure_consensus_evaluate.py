# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Freeze and execute the one-run PP-OCRv5 structure-consensus gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import random
import sys
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

from ml.ocr.official_bakeoff import production_evaluate as locked
from ml.ocr.production_gate import evaluate_partition


PROFILE = "graphreader-ocr-structure-consensus-public-gate-v1"
SPLIT_SCHEMA = "graphreader.ocr-structure-consensus-split.v1"
CORE_SCHEMA = "graphreader.ocr-structure-consensus-core-predictions.v1"
PREDICTIONS_SCHEMA = "graphreader.ocr-structure-consensus-predictions.v1"
RUNTIME_SCHEMA = "graphreader.ocr-structure-consensus-runtime-results.v1"
REPORT_SCHEMA = "graphreader.ocr-structure-consensus-production-gate.v1"
COMPOSITION_ID = "graph-structure-consensus-v1"
DETECTION_MODEL_ID = locked.DETECTION_MODEL_ID
RECOGNITION_MODEL_ID = locked.RECOGNITION_MODEL_ID
TEXT_FAMILIES = ("integer", "decimal", "negative", "percentage", "ambiguity")
TEXT_ROLES = ("x_tick", "y_tick", "phase_header", "annotation", "participant")
GRAPH_CONTEXT_FAMILIES = (
    "axes_and_ticks",
    "phase_divider",
    "open_markers_and_connectors",
    "filled_markers_and_connectors",
    "legend_frame",
    "bracket",
    "arrow_shaft_and_head",
    "line_intersection",
    "mixed_graph_structures",
)
DEGRADATIONS = ("clean", "soft_blur", "downsample_restore", "jpeg_roundtrip")
EXPECTED_TEXT_PER_PARTITION = 200
EXPECTED_EXCLUSIONS_PER_PARTITION = 50
EXPECTED_CASES = 500
MAXIMUM_RESOURCE_BYTES = 8 * 1024 * 1024


ProductionGateError = locked.ProductionGateError
DuplicateJsonKeyError = locked.DuplicateJsonKeyError
load_strict_json = locked.load_strict_json
canonical_json_bytes = locked.canonical_json_bytes
hash_bytes = locked.hash_bytes
hash_file = locked.hash_file


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def to_json(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True)
class ModelRegion:
    region_id: str
    bounds: Box
    confidence: float


@dataclass(frozen=True)
class StructureCandidate:
    region_id: str
    bounds: Box
    text_likelihood: float
    structure_likelihood: float
    likely_graph_structure: bool
    component_count: int


@dataclass(frozen=True)
class DetectionEvidence:
    model_regions: tuple[ModelRegion, ...]
    structure_candidates: tuple[StructureCandidate, ...]
    matches: tuple[tuple[str, str, float], ...]
    final_regions: tuple[ModelRegion, ...]
    input_sha256: str
    input_shape: tuple[int, ...]
    output_sha256: str
    output_shape: tuple[int, ...]
    duration_ms: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionGateError(message)


def _default_protocol() -> Path:
    return Path(__file__).resolve().with_name("STRUCTURE_CONSENSUS_GATE_PROTOCOL.json")


def _default_metrics_evaluator() -> Path:
    return Path(__file__).resolve().parents[1] / "production_gate.py"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_hex(value: str, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower()),
        f"{label} must be SHA-256.",
    )
    return value.lower()


def validate_protocol(
    protocol_path: Path,
    metrics_evaluator_path: Path,
    workflow_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_strict_json(protocol_path)
    workflow_path = workflow_path or Path(__file__).resolve()
    _require(
        protocol.get("schema") == "graphreader.ocr-structure-consensus-gate-protocol.v1",
        "Structure-consensus protocol schema is invalid.",
    )
    _require(protocol.get("profile") == PROFILE, "Structure-consensus profile changed.")
    _require(
        protocol.get("status") == "frozen_before_fixture_generation_and_inference",
        "Structure-consensus protocol is not frozen before inference.",
    )
    _require(protocol.get("scope") == "public_synthetic", "OCR scope must be public synthetic.")
    _require(protocol.get("private_data") is False, "Private data is prohibited.")
    _require(protocol.get("chandler_used") is False, "Chandler is prohibited before private validation.")
    _require(
        protocol.get("selection_locked_before_inference") is True,
        "Candidate selection must be locked before inference.",
    )
    _require(
        protocol.get("metrics_evaluator_sha256") == hash_file(metrics_evaluator_path),
        "Frozen metrics evaluator changed.",
    )
    _require(
        protocol.get("execution_workflow_sha256") == hash_file(workflow_path),
        "Frozen structure-consensus workflow changed.",
    )
    sources = protocol.get("reviewed_source_sha256")
    _require(isinstance(sources, dict) and sources, "Reviewed source inventory is missing.")
    root = _repo_root()
    for relative, expected in sources.items():
        _require(isinstance(relative, str), "Reviewed source path is invalid.")
        source = (root / relative).resolve()
        _require(source.is_relative_to(root) and source.is_file(), f"Reviewed source is missing: {relative}")
        _require(hash_file(source) == expected, f"Reviewed source changed: {relative}")
    prior = protocol.get("prior_exposed_split_forbidden")
    _require(isinstance(prior, dict), "Prior exposed split denial is missing.")
    _sha256_hex(str(prior.get("split_sha256")), "Prior split")
    _sha256_hex(str(prior.get("fixture_archive_sha256")), "Prior fixture archive")
    budget = protocol.get("experiment_budget")
    _require(
        budget
        == {
            "fixture_freezes": 1,
            "official_composition_evaluations": 1,
            "split_regeneration_after_inference": 0,
            "threshold_changes_after_inference": 0,
            "workflow_changes_after_inference": 0,
        },
        "Structure-consensus experiment budget changed.",
    )
    return protocol


def _text_for(family: str, index: int) -> tuple[str, str]:
    if family == "integer":
        value = str((index * 17) % 101)
        return value, value
    if family == "decimal":
        value = f"{(index * 7) % 31}.{(index * 3) % 10}"
        return value, value
    if family == "negative":
        value = f"-{((index * 11) % 99) + 1}"
        return value, value
    if family == "percentage":
        value = f"{(index * 13) % 101}%"
        return value, value
    ambiguity = ("O", "o", "l", "I", "O.l", "-O", "lO%", "I.O")
    display = ambiguity[index % len(ambiguity)]
    truth = display.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"}))
    return display, truth


def _role_position(role: str, width: int, height: int, text_width: int, text_height: int) -> tuple[int, int]:
    if role == "phase_header":
        return ((width - text_width) // 2, 10)
    if role == "y_tick":
        return (8, max(46, (height - text_height) // 2))
    if role == "x_tick":
        return (max(82, width // 3), height - text_height - 10)
    if role == "participant":
        return (width - text_width - 12, height - text_height - 10)
    return (width - text_width - 20, max(48, height // 2 - text_height // 2))


def _mask_rectangle(rectangles: list[dict[str, Any]], kind: str, left: int, top: int, right: int, bottom: int) -> None:
    rectangles.append(
        {
            "kind": kind,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }
    )


def _draw_structures(draw: Any, width: int, height: int, context: str, seed: int) -> list[dict[str, Any]]:
    rectangles: list[dict[str, Any]] = []
    axis_x = 54
    axis_y = height - 34
    draw.line((axis_x, 30, axis_x, axis_y), fill=(24, 24, 24), width=2)
    draw.line((axis_x, axis_y, width - 18, axis_y), fill=(24, 24, 24), width=2)
    _mask_rectangle(rectangles, "y_axis", axis_x - 2, 28, axis_x + 3, axis_y + 3)
    _mask_rectangle(rectangles, "x_axis", axis_x - 2, axis_y - 2, width - 16, axis_y + 3)
    for offset in range(0, 5):
        x = axis_x + 22 + (offset * 40)
        y = axis_y - 18 - ((offset * 13 + seed) % 58)
        draw.line((x, axis_y - 4, x, axis_y + 4), fill=(24, 24, 24), width=1)
        draw.line((axis_x - 4, y, axis_x + 4, y), fill=(24, 24, 24), width=1)
        _mask_rectangle(rectangles, "x_tick", x - 2, axis_y - 6, x + 3, axis_y + 7)
        _mask_rectangle(rectangles, "y_tick", axis_x - 6, y - 2, axis_x + 7, y + 3)
    if context in {"phase_divider", "mixed_graph_structures"}:
        divider_x = width // 2
        draw.line((divider_x, 24, divider_x, axis_y), fill=(30, 30, 30), width=2)
        _mask_rectangle(rectangles, "phase_divider", divider_x - 2, 22, divider_x + 3, axis_y + 2)
    if context in {"open_markers_and_connectors", "filled_markers_and_connectors", "mixed_graph_structures"}:
        points = [(100, 88), (134, 64), (168, 82), (202, 52)]
        draw.line(points, fill=(28, 28, 28), width=2)
        filled = context != "open_markers_and_connectors"
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(20, 20, 20), fill=(20, 20, 20) if filled else (255, 255, 255), width=2)
    if context in {"legend_frame", "mixed_graph_structures"}:
        draw.rectangle((225, 34, 306, 70), outline=(30, 30, 30), width=2)
        draw.ellipse((235, 48, 243, 56), fill=(20, 20, 20))
        draw.line((248, 52, 292, 52), fill=(30, 30, 30), width=2)
    if context == "bracket":
        draw.line((228, 48, 228, 92), fill=(25, 25, 25), width=2)
        draw.line((228, 48, 252, 48), fill=(25, 25, 25), width=2)
        draw.line((228, 92, 252, 92), fill=(25, 25, 25), width=2)
    if context in {"arrow_shaft_and_head", "mixed_graph_structures"}:
        draw.line((214, 104, 284, 76), fill=(25, 25, 25), width=2)
        draw.polygon(((284, 76), (272, 75), (279, 87)), fill=(25, 25, 25))
    if context == "line_intersection":
        draw.line((190, 48, 272, 108), fill=(25, 25, 25), width=2)
        draw.line((190, 108, 272, 48), fill=(25, 25, 25), width=2)
    return rectangles


def _apply_degradation(image: Any, family: str) -> Any:
    from io import BytesIO
    from PIL import Image, ImageFilter

    if family == "clean":
        return image
    if family == "soft_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=0.55))
    if family == "downsample_restore":
        reduced = image.resize((240, 120), Image.Resampling.BILINEAR)
        return reduced.resize(image.size, Image.Resampling.BILINEAR)
    if family == "jpeg_roundtrip":
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=86, optimize=False, progressive=False)
        stream.seek(0)
        with Image.open(stream) as loaded:
            return loaded.convert("RGB")
    raise ProductionGateError(f"Unknown degradation family: {family}")


def _masked_bgr(image: Any, rectangles: Sequence[dict[str, Any]]) -> bytes:
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    for rectangle in rectangles:
        left = int(rectangle["left"])
        top = int(rectangle["top"])
        right = int(rectangle["right"])
        bottom = int(rectangle["bottom"])
        _require(0 <= left < right <= image.width, "Mask rectangle x bounds are invalid.")
        _require(0 <= top < bottom <= image.height, "Mask rectangle y bounds are invalid.")
        bgr[top:bottom, left:right, :] = 255
    return bgr.tobytes(order="C")


def _source_bgr(image: Any) -> bytes:
    import numpy as np

    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1]).tobytes(order="C")


def _render_case(
    case_id: str,
    kind: str,
    family: str,
    role: str,
    context: str,
    degradation: str,
    index: int,
    font_path: Path,
) -> tuple[Any, str, str, Box | None, list[dict[str, Any]]]:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 320, 160
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    rectangles = _draw_structures(draw, width, height, context, index)
    display = truth = ""
    truth_box: Box | None = None
    if kind == "text":
        display, truth = _text_for(family, index)
        font = ImageFont.truetype(str(font_path), 22 + (index % 3))
        raw = draw.textbbox((0, 0), display, font=font, stroke_width=0)
        text_width = raw[2] - raw[0]
        text_height = raw[3] - raw[1]
        x, y = _role_position(role, width, height, text_width, text_height)
        draw.text((x, y), display, font=font, fill=(18, 18, 18))
        rendered = draw.textbbox((x, y), display, font=font, stroke_width=0)
        truth_box = Box(float(rendered[0]), float(rendered[1]), float(rendered[2]), float(rendered[3]))
    image = _apply_degradation(image, degradation)
    return image, display, truth, truth_box, rectangles


def _write_new(path: Path, value: bytes) -> None:
    if path.exists():
        raise ProductionGateError(f"Refusing to overwrite frozen evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def freeze_split(
    output_root: Path,
    protocol_path: Path,
    metrics_evaluator_path: Path,
    font_path: Path,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path, metrics_evaluator_path)
    _require(font_path.is_file(), f"Frozen renderer font is missing: {font_path}")
    _require(not output_root.exists(), "Frozen output root already exists.")
    cases: list[dict[str, Any]] = []
    for partition_index, partition in enumerate(("validation", "sealed_test")):
        for index in range(EXPECTED_TEXT_PER_PARTITION):
            family = TEXT_FAMILIES[index % len(TEXT_FAMILIES)]
            role = TEXT_ROLES[(index // len(TEXT_FAMILIES)) % len(TEXT_ROLES)]
            context = GRAPH_CONTEXT_FAMILIES[(index * 5 + partition_index) % len(GRAPH_CONTEXT_FAMILIES)]
            degradation = DEGRADATIONS[(index * 3 + partition_index) % len(DEGRADATIONS)]
            case_id = f"{partition}-text-{index:03d}"
            image, display, truth, truth_box, rectangles = _render_case(
                case_id, "text", family, role, context, degradation,
                index + (partition_index * EXPECTED_TEXT_PER_PARTITION), font_path,
            )
            path = output_root / "assets" / f"{case_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG", optimize=False, compress_level=9)
            with __import__("PIL.Image", fromlist=["Image"]).open(path) as loaded:
                exact = loaded.convert("RGB")
            source_bgr = _source_bgr(exact)
            detector_bgr = _masked_bgr(exact, rectangles)
            cases.append(
                {
                    "case_id": case_id,
                    "partition": partition,
                    "kind": "text",
                    "family": family,
                    "graph_context_family": context,
                    "degradation_family": degradation,
                    "display_text": display,
                    "truth_text": truth,
                    "truth_role": role,
                    "truth_bbox": truth_box.to_json() if truth_box is not None else None,
                    "expected_region_count": 1,
                    "source_path": f"assets/{case_id}.png",
                    "source_sha256": hash_file(path),
                    "source_width": exact.width,
                    "source_height": exact.height,
                    "source_bgr_sha256": hash_bytes(source_bgr),
                    "detector_image_bgr_sha256": hash_bytes(detector_bgr),
                    "mask_rectangles": rectangles,
                }
            )
        for index in range(EXPECTED_EXCLUSIONS_PER_PARTITION):
            context = GRAPH_CONTEXT_FAMILIES[(index * 7 + partition_index) % len(GRAPH_CONTEXT_FAMILIES)]
            degradation = DEGRADATIONS[(index + partition_index) % len(DEGRADATIONS)]
            case_id = f"{partition}-exclusion-{index:03d}"
            image, _, _, _, rectangles = _render_case(
                case_id, "exclusion", "exclusion", "other", context, degradation,
                index + 1000 + (partition_index * EXPECTED_EXCLUSIONS_PER_PARTITION), font_path,
            )
            path = output_root / "assets" / f"{case_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG", optimize=False, compress_level=9)
            with __import__("PIL.Image", fromlist=["Image"]).open(path) as loaded:
                exact = loaded.convert("RGB")
            source_bgr = _source_bgr(exact)
            detector_bgr = _masked_bgr(exact, rectangles)
            cases.append(
                {
                    "case_id": case_id,
                    "partition": partition,
                    "kind": "exclusion",
                    "family": "exclusion",
                    "graph_context_family": context,
                    "degradation_family": degradation,
                    "display_text": "",
                    "truth_text": "",
                    "truth_role": "other",
                    "truth_bbox": None,
                    "expected_region_count": 0,
                    "source_path": f"assets/{case_id}.png",
                    "source_sha256": hash_file(path),
                    "source_width": exact.width,
                    "source_height": exact.height,
                    "source_bgr_sha256": hash_bytes(source_bgr),
                    "detector_image_bgr_sha256": hash_bytes(detector_bgr),
                    "mask_rectangles": rectangles,
                }
            )
    _require(len(cases) == EXPECTED_CASES, "Frozen split case count is invalid.")
    fixture_archive = locked.build_fixture_archive(output_root, cases)
    split = {
        "schema": SPLIT_SCHEMA,
        "profile": PROFILE,
        "scope": "public_synthetic",
        "sealed": True,
        "selection_locked_before_inference": True,
        "private_data": False,
        "chandler_used": False,
        "protocol_sha256": hash_file(protocol_path),
        "evaluator_source_sha256": hash_file(metrics_evaluator_path),
        "workflow_source_sha256": hash_file(Path(__file__)),
        "fixture_archive_sha256": hash_bytes(fixture_archive),
        "renderer": {
            "family": "graphreader-structure-context-renderer-v1",
            "seed": protocol["new_split"]["renderer_seed"],
            "font_sha256": hash_file(font_path),
            "width": 320,
            "height": 160,
        },
        "cases": cases,
    }
    split_bytes = canonical_json_bytes(split)
    forbidden = protocol["prior_exposed_split_forbidden"]
    _require(hash_bytes(split_bytes) != forbidden["split_sha256"], "New split reused the exposed split.")
    _require(hash_bytes(fixture_archive) != forbidden["fixture_archive_sha256"], "New fixtures reused the exposed archive.")
    _write_new(output_root / "fixtures.zip", fixture_archive)
    _write_new(output_root / "split.json", split_bytes)
    return {
        "case_count": len(cases),
        "sealed_split_sha256": hash_bytes(split_bytes),
        "fixture_archive_sha256": hash_bytes(fixture_archive),
        "split": split,
    }


def verify_frozen_split(
    frozen_root: Path,
    protocol_path: Path,
    metrics_evaluator_path: Path,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path, metrics_evaluator_path)
    split_path = frozen_root / "split.json"
    archive_path = frozen_root / "fixtures.zip"
    split = load_strict_json(split_path)
    _require(split.get("schema") == SPLIT_SCHEMA, "Frozen split schema changed.")
    _require(split.get("profile") == PROFILE, "Frozen split profile changed.")
    _require(split.get("sealed") is True, "Frozen split is not sealed.")
    _require(split.get("private_data") is False, "Frozen split contains private data.")
    _require(split.get("chandler_used") is False, "Frozen split used Chandler.")
    _require(split.get("protocol_sha256") == hash_file(protocol_path), "Frozen protocol hash changed.")
    _require(split.get("evaluator_source_sha256") == hash_file(metrics_evaluator_path), "Frozen evaluator hash changed.")
    _require(split.get("workflow_source_sha256") == hash_file(Path(__file__)), "Frozen workflow hash changed.")
    fixture_archive = archive_path.read_bytes()
    _require(len(fixture_archive) <= MAXIMUM_RESOURCE_BYTES, "Fixture archive exceeds gate resource limit.")
    _require(split.get("fixture_archive_sha256") == hash_bytes(fixture_archive), "Fixture archive hash changed.")
    cases = split.get("cases")
    _require(isinstance(cases, list) and len(cases) == EXPECTED_CASES, "Frozen case count changed.")
    locked.verify_fixture_archive(fixture_archive, cases)
    text_counts = {"validation": 0, "sealed_test": 0}
    exclusion_counts = {"validation": 0, "sealed_test": 0}
    seen: set[str] = set()
    for case in cases:
        _require(isinstance(case, dict), "Frozen case is invalid.")
        case_id = case.get("case_id")
        partition = case.get("partition")
        kind = case.get("kind")
        _require(isinstance(case_id, str) and case_id not in seen, "Frozen case ID is invalid.")
        _require(partition in text_counts, "Frozen partition is invalid.")
        _require(kind in {"text", "exclusion"}, "Frozen case kind is invalid.")
        seen.add(case_id)
        source = frozen_root / str(case.get("source_path"))
        _require(source.is_file() and hash_file(source) == case.get("source_sha256"), f"Frozen source changed: {case_id}")
        from PIL import Image

        with Image.open(source) as loaded:
            image = loaded.convert("RGB")
        _require(image.width == case.get("source_width") and image.height == case.get("source_height"), f"Frozen dimensions changed: {case_id}")
        _require(hash_bytes(_source_bgr(image)) == case.get("source_bgr_sha256"), f"Frozen source pixels changed: {case_id}")
        rectangles = case.get("mask_rectangles")
        _require(isinstance(rectangles, list) and rectangles, f"Frozen mask geometry is missing: {case_id}")
        _require(hash_bytes(_masked_bgr(image, rectangles)) == case.get("detector_image_bgr_sha256"), f"Frozen detector pixels changed: {case_id}")
        if kind == "text":
            text_counts[partition] += 1
            _require(case.get("family") in TEXT_FAMILIES, "Frozen text family is invalid.")
            _require(case.get("truth_role") in TEXT_ROLES, "Frozen text role is invalid.")
            _require(isinstance(case.get("truth_bbox"), list) and len(case["truth_bbox"]) == 4, "Frozen text bbox is invalid.")
            _require(case.get("expected_region_count") == 1, "Frozen text region count changed.")
        else:
            exclusion_counts[partition] += 1
            _require(case.get("truth_bbox") is None, "Exclusion bbox must be null.")
            _require(case.get("expected_region_count") == 0, "Exclusion region count changed.")
    _require(
        text_counts == {"validation": EXPECTED_TEXT_PER_PARTITION, "sealed_test": EXPECTED_TEXT_PER_PARTITION},
        "Frozen text partition counts changed.",
    )
    _require(
        exclusion_counts == {"validation": EXPECTED_EXCLUSIONS_PER_PARTITION, "sealed_test": EXPECTED_EXCLUSIONS_PER_PARTITION},
        "Frozen exclusion partition counts changed.",
    )
    return {
        "protocol": protocol,
        "split": split,
        "fixture_archive_bytes": fixture_archive,
        "fixture_archive_sha256": hash_bytes(fixture_archive),
    }


def detector_tensor(masked_bgr: bytes, width: int, height: int) -> Any:
    import cv2
    import numpy as np

    source = np.frombuffer(masked_bgr, dtype=np.uint8).reshape((height, width, 3))
    ratio = 960.0 / max(width, height)
    resized_width = int(width * ratio)
    resized_height = int(height * ratio)
    target_width = ((resized_width + 127) // 128) * 128
    target_height = ((resized_height + 127) // 128) * 128
    resized = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    values = resized.astype(np.float32) / np.float32(255.0)
    means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    scales = np.asarray([1 / 0.229, 1 / 0.224, 1 / 0.225], dtype=np.float32)
    values = (values - means) * scales
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32)


def _box_from_points(points: Any, width: int, height: int, tensor_width: int, tensor_height: int) -> Box:
    import numpy as np

    values = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    x = np.clip(np.rint(values[:, 0] * width / tensor_width), 0, width)
    y = np.clip(np.rint(values[:, 1] * height / tensor_height), 0, height)
    return Box(float(x.min()), float(y.min()), float(x.max()), float(y.max()))


def _order_polygon(points: Any) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    center = values.mean(axis=0)
    angles = np.arctan2(values[:, 1] - center[1], values[:, 0] - center[0])
    order = np.lexsort((values[:, 0], values[:, 1], angles))
    ordered = values[order]
    signed = 0.5 * np.sum(
        ordered[:, 0] * np.roll(ordered[:, 1], -1)
        - np.roll(ordered[:, 0], -1) * ordered[:, 1]
    )
    if signed < 0:
        ordered = ordered[::-1]
    start = min(range(len(ordered)), key=lambda index: (ordered[index, 1], ordered[index, 0]))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def _unclip_round(points: Any, ratio: float) -> Any | None:
    import numpy as np

    ordered = _order_polygon(points).astype(np.float64)
    area = abs(0.5 * np.sum(
        ordered[:, 0] * np.roll(ordered[:, 1], -1)
        - np.roll(ordered[:, 0], -1) * ordered[:, 1]
    ))
    lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    perimeter = float(lengths.sum())
    if area <= 0 or perimeter <= 0:
        return None
    distance = area * ratio / perimeter
    maximum_step = math.pi / 2 if distance <= 0.25 else 2 * math.acos(max(-1.0, min(1.0, 1 - (0.25 / distance))))
    expanded: list[tuple[float, float]] = []
    for index, vertex in enumerate(ordered):
        previous = ordered[(index - 1) % len(ordered)]
        following = ordered[(index + 1) % len(ordered)]
        previous_delta = vertex - previous
        next_delta = following - vertex
        if np.linalg.norm(previous_delta) <= 1e-9 or np.linalg.norm(next_delta) <= 1e-9:
            return None
        previous_normal = np.asarray([previous_delta[1], -previous_delta[0]]) / np.linalg.norm(previous_delta)
        next_normal = np.asarray([next_delta[1], -next_delta[0]]) / np.linalg.norm(next_delta)
        start = math.atan2(float(previous_normal[1]), float(previous_normal[0]))
        end = math.atan2(float(next_normal[1]), float(next_normal[0]))
        while end <= start:
            end += 2 * math.pi
        segments = max(1, math.ceil((end - start) / maximum_step))
        for segment in range(segments + 1):
            angle = start + ((end - start) * segment / segments)
            expanded.append((vertex[0] + math.cos(angle) * distance, vertex[1] + math.sin(angle) * distance))
        if len(expanded) > 4096:
            return None
    return _order_polygon(np.asarray(expanded, dtype=np.float32))


def db_model_regions(probabilities: Any, width: int, height: int) -> tuple[ModelRegion, ...]:
    import cv2
    import numpy as np

    values = np.asarray(probabilities, dtype=np.float32).reshape((probabilities.shape[-2], probabilities.shape[-1]))
    binary = (values > np.float32(0.3)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result: list[ModelRegion] = []
    for contour in contours[:1000]:
        if len(contour) < 3:
            continue
        rectangle = cv2.minAreaRect(contour.astype(np.float32))
        points = cv2.boxPoints(rectangle)
        if min(rectangle[1]) < 3:
            continue
        left = max(0, math.floor(float(points[:, 0].min())))
        top = max(0, math.floor(float(points[:, 1].min())))
        right = min(values.shape[1] - 1, math.ceil(float(points[:, 0].max())))
        bottom = min(values.shape[0] - 1, math.ceil(float(points[:, 1].max())))
        local = points - np.asarray([left, top], dtype=np.float32)
        mask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [local.astype(np.int32)], 255)
        selected = values[top : bottom + 1, left : right + 1][mask != 0]
        if selected.size == 0:
            continue
        confidence = float(selected.mean())
        if confidence < 0.6:
            continue
        expanded = _unclip_round(points, 1.5)
        if expanded is None:
            continue
        expanded_rectangle = cv2.minAreaRect(expanded.astype(np.float32))
        if min(expanded_rectangle[1]) < 5:
            continue
        expanded_points = cv2.boxPoints(expanded_rectangle)
        bounds = _box_from_points(expanded_points, width, height, values.shape[1], values.shape[0])
        if bounds.width <= 0 or bounds.height <= 0:
            continue
        region_id = hash_bytes(
            canonical_json_bytes(
                {
                    "model": "d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb",
                    "bounds": bounds.to_json(),
                    "confidence": confidence,
                }
            )
        )[:32]
        result.append(ModelRegion(region_id, bounds, confidence))
    return tuple(result)


def connected_component_candidates(masked_bgr: bytes, width: int, height: int) -> tuple[StructureCandidate, ...]:
    import cv2
    import numpy as np

    image = np.frombuffer(masked_bgr, dtype=np.uint8).reshape((height, width, 3))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = int(min(224, max(32, round(float(gray.mean()) * 0.80))))
    foreground = (gray <= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=4)
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        left, top, component_width, component_height, area = [int(value) for value in stats[label]]
        if area < 2 or component_width > max(2, width * 0.15) or component_height > max(2, height * 0.20):
            continue
        crop = foreground[top : top + component_height, left : left + component_width]
        maximum_row = float(crop.sum(axis=1).max()) / max(1, component_width)
        maximum_column = float(crop.sum(axis=0).max()) / max(1, component_height)
        density = area / max(1, component_width * component_height)
        aspect = component_width / max(1, component_height)
        marker = (
            0.65 <= aspect <= 1.55
            and 3 <= component_width <= 20
            and 3 <= component_height <= 20
            and density >= 0.25
            and maximum_row >= 0.60
            and maximum_column >= 0.60
        )
        components.append(
            {
                "left": left,
                "top": top,
                "right": left + component_width - 1,
                "bottom": top + component_height - 1,
                "area": area,
                "count": 1,
                "maximum_row": maximum_row,
                "maximum_column": maximum_column,
                "marker_count": 1 if marker else 0,
            }
        )
    remaining = sorted(components, key=lambda item: (item["top"], item["left"]))
    lines: list[dict[str, Any]] = []
    while remaining:
        line = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for index in range(len(remaining) - 1, -1, -1):
                candidate = remaining[index]
                overlap = max(0, min(line["bottom"], candidate["bottom"]) - max(line["top"], candidate["top"]) + 1)
                fraction = overlap / max(1, min(line["bottom"] - line["top"] + 1, candidate["bottom"] - candidate["top"] + 1))
                gap = max(0, candidate["left"] - line["right"] - 1, line["left"] - candidate["right"] - 1)
                maximum_gap = max(line["bottom"] - line["top"] + 1, candidate["bottom"] - candidate["top"] + 1) * 2.5
                if fraction >= 0.35 and gap <= maximum_gap:
                    other = remaining.pop(index)
                    line = {
                        "left": min(line["left"], other["left"]),
                        "top": min(line["top"], other["top"]),
                        "right": max(line["right"], other["right"]),
                        "bottom": max(line["bottom"], other["bottom"]),
                        "area": line["area"] + other["area"],
                        "count": line["count"] + other["count"],
                        "maximum_row": max(line["maximum_row"], other["maximum_row"]),
                        "maximum_column": max(line["maximum_column"], other["maximum_column"]),
                        "marker_count": line["marker_count"] + other["marker_count"],
                    }
                    changed = True
        lines.append(line)

    remaining = sorted(lines, key=lambda item: (item["left"], item["top"]))
    vertical_lines: list[dict[str, Any]] = []

    def is_rotated_glyph_candidate(item: dict[str, Any]) -> bool:
        item_width = item["right"] - item["left"] + 1
        item_height = item["bottom"] - item["top"] + 1
        return item["count"] == 1 and item_width >= item_height * 1.2 and item["marker_count"] == 0

    while remaining:
        line = remaining.pop(0)
        if not is_rotated_glyph_candidate(line):
            vertical_lines.append(line)
            continue
        changed = True
        while changed:
            changed = False
            for index in range(len(remaining) - 1, -1, -1):
                candidate = remaining[index]
                if not is_rotated_glyph_candidate(candidate):
                    continue
                overlap = max(0, min(line["right"], candidate["right"]) - max(line["left"], candidate["left"]) + 1)
                line_width = line["right"] - line["left"] + 1
                candidate_width = candidate["right"] - candidate["left"] + 1
                fraction = overlap / max(1, min(line_width, candidate_width))
                gap = max(0, candidate["top"] - line["bottom"] - 1, line["top"] - candidate["bottom"] - 1)
                maximum_gap = max(line_width, candidate_width) * 2.5
                if fraction >= 0.35 and gap <= maximum_gap:
                    other = remaining.pop(index)
                    line = {
                        "left": min(line["left"], other["left"]),
                        "top": min(line["top"], other["top"]),
                        "right": max(line["right"], other["right"]),
                        "bottom": max(line["bottom"], other["bottom"]),
                        "area": line["area"] + other["area"],
                        "count": line["count"] + other["count"],
                        "maximum_row": max(line["maximum_row"], other["maximum_row"]),
                        "maximum_column": max(line["maximum_column"], other["maximum_column"]),
                        "marker_count": line["marker_count"] + other["marker_count"],
                    }
                    changed = True
        vertical_lines.append(line)
    lines = vertical_lines
    result: list[StructureCandidate] = []
    for line in lines:
        component_width = line["right"] - line["left"] + 1
        component_height = line["bottom"] - line["top"] + 1
        density = line["area"] / max(1, component_width * component_height)
        aspect = component_width / max(1, component_height)
        long_thin = (aspect >= 2.5 or aspect <= 0.4) and density >= 0.60
        compact = 0.65 <= aspect <= 1.55
        compact_stroke = compact and line["maximum_row"] >= 0.80 and line["maximum_column"] >= 0.80 and component_width <= 20 and component_height <= 20
        intersection = compact and density <= 0.60 and line["maximum_row"] >= 0.80 and line["maximum_column"] >= 0.80
        likely_structure = (line["count"] > 1 and line["marker_count"] == line["count"]) or long_thin or compact_stroke or intersection
        structure_likelihood = 0.98 if line["count"] > 1 and line["marker_count"] == line["count"] else 0.95 if likely_structure else 0.08 if line["count"] > 1 else 0.35
        text_likelihood = 0.12 if line["marker_count"] == line["count"] else min(0.94, 0.68 + line["count"] * 0.06) if line["count"] > 1 else max(0.15, min(0.65, 0.58 - structure_likelihood * 0.30))
        bounds = Box(float(line["left"]), float(line["top"]), float(line["right"] + 1), float(line["bottom"] + 1))
        region_id = hash_bytes(canonical_json_bytes({"bounds": bounds.to_json()}))[:32]
        result.append(StructureCandidate(region_id, bounds, text_likelihood, structure_likelihood, likely_structure, line["count"]))
    return tuple(sorted(result, key=lambda item: (item.bounds.top, item.bounds.left, item.region_id)))


def overlap_coefficient(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left.right, right.right) - max(left.left, right.left))
    intersection_height = max(0.0, min(left.bottom, right.bottom) - max(left.top, right.top))
    denominator = min(left.width * left.height, right.width * right.height)
    return 0.0 if denominator <= 0 else intersection_width * intersection_height / denominator


def compose_consensus(
    model_regions: Sequence[ModelRegion],
    candidates: Sequence[StructureCandidate],
) -> tuple[tuple[tuple[str, str, float], ...], tuple[ModelRegion, ...]]:
    eligible = [candidate for candidate in candidates if not candidate.likely_graph_structure and candidate.text_likelihood >= 0.45]
    matches = [
        (model, candidate, overlap_coefficient(model.bounds, candidate.bounds))
        for model in model_regions
        for candidate in eligible
    ]
    matches = [item for item in matches if item[2] >= 0.50]
    matches.sort(key=lambda item: (-item[0].confidence, -item[2], -item[1].text_likelihood, item[0].region_id, item[1].region_id))
    used_models: set[str] = set()
    used_candidates: set[str] = set()
    selected: list[ModelRegion] = []
    evidence: list[tuple[str, str, float]] = []
    for model, candidate, overlap in matches:
        if model.region_id in used_models or candidate.region_id in used_candidates:
            continue
        used_models.add(model.region_id)
        used_candidates.add(candidate.region_id)
        selected.append(model)
        evidence.append((model.region_id, candidate.region_id, overlap))
    selected.sort(key=lambda item: (item.bounds.top, item.bounds.left, item.region_id))
    return tuple(evidence), tuple(selected)


def detect_regions(session: Any, masked_bgr: bytes, width: int, height: int) -> DetectionEvidence:
    import numpy as np

    tensor = detector_tensor(masked_bgr, width, height)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    _require(len(inputs) == 1 and len(outputs) == 1, "Detector ONNX contract changed.")
    started = perf_counter()
    output = np.asarray(session.run([outputs[0].name], {inputs[0].name: tensor})[0], dtype=np.float32)
    duration_ms = (perf_counter() - started) * 1000.0
    _require(output.shape == (1, 1, tensor.shape[2], tensor.shape[3]), "Detector output shape changed.")
    _require(np.isfinite(output).all(), "Detector output contains non-finite values.")
    _require(float(output.min()) >= 0.0 and float(output.max()) <= 1.0, "Detector output is not a probability tensor.")
    model_regions = db_model_regions(output, width, height)
    candidates = connected_component_candidates(masked_bgr, width, height)
    matches, final_regions = compose_consensus(model_regions, candidates)
    return DetectionEvidence(
        model_regions,
        candidates,
        matches,
        final_regions,
        hash_bytes(tensor.tobytes(order="C")),
        tuple(int(value) for value in tensor.shape),
        hash_bytes(output.tobytes(order="C")),
        tuple(int(value) for value in output.shape),
        duration_ms,
    )


def _region_json(region: ModelRegion) -> dict[str, Any]:
    return {"region_id": region.region_id, "bounds": region.bounds.to_json(), "confidence": region.confidence}


def _candidate_json(candidate: StructureCandidate) -> dict[str, Any]:
    return {
        "region_id": candidate.region_id,
        "bounds": candidate.bounds.to_json(),
        "text_likelihood": candidate.text_likelihood,
        "structure_likelihood": candidate.structure_likelihood,
        "likely_graph_structure": candidate.likely_graph_structure,
        "component_count": candidate.component_count,
    }


def _tensor_json(sha256_value: str, shape: Sequence[int]) -> dict[str, Any]:
    return {"sha256": sha256_value, "dtype": "float32", "shape": list(shape)}


def _iou(left: Box, right: Sequence[float]) -> float:
    truth = Box(*[float(value) for value in right])
    intersection_width = max(0.0, min(left.right, truth.right) - max(left.left, truth.left))
    intersection_height = max(0.0, min(left.bottom, truth.bottom) - max(left.top, truth.top))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + truth.width * truth.height - intersection
    return 0.0 if union <= 0 else intersection / union


def _duplicate_count(regions: Sequence[ModelRegion]) -> int:
    duplicates = 0
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            if _iou(left.bounds, right.bounds.to_json()) >= 0.70:
                duplicates += 1
    return duplicates


def _missing_marker_evidence(
    cases: Sequence[dict[str, Any]],
) -> tuple[bool, dict[str, int], None, list[str]]:
    return False, {str(case["case_id"]): 0 for case in cases}, None, [
        "No checksum-bound independent marker-stage run was available; external precomputed marker-result injection is prohibited."
    ]


def _embedded_resource(media_type: str, content: bytes, label: str) -> dict[str, Any]:
    _require(len(content) <= MAXIMUM_RESOURCE_BYTES, f"{label} exceeds the gate resource limit.")
    return locked._embedded(media_type, content)


def _threshold_blockers(metrics: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    minimums = {
        "validation_exact_match": 0.90,
        "validation_role_accuracy": 0.90,
        "sealed_test_exact_match": 0.90,
        "sealed_test_role_accuracy": 0.90,
        "detection_exact_rate": 1.0,
    }
    maximums = {
        "validation_cer": 0.05,
        "sealed_test_cer": 0.05,
        "onnx_max_abs_error": 1e-4,
        "duplicate_region_count": 0,
        "exclusion_false_region_count": 0,
        "marker_creation_count": 0,
    }
    for name, expected in minimums.items():
        if float(metrics[name]) < expected:
            blockers.append(f"{name}={metrics[name]} is below {expected}.")
    for name, expected in maximums.items():
        if float(metrics[name]) > expected:
            blockers.append(f"{name}={metrics[name]} exceeds {expected}.")
    return blockers


def evaluate_official_candidate(
    frozen_root: Path,
    protocol_path: Path,
    metrics_evaluator_path: Path,
    conversion_report_path: Path,
    source_root: Path,
    output_root: Path,
    *,
    session_factory: Callable[[Path], Any] = locked._cpu_session,
    parity_runner: Callable[[dict[str, Any], Path], list[dict[str, Any]]] = locked._direct_parity_pairs,
) -> dict[str, Any]:
    _require(not output_root.exists(), "One-run OCR evaluation output already exists.")
    verification = verify_frozen_split(frozen_root, protocol_path, metrics_evaluator_path)
    split = verification["split"]
    split_bytes = (frozen_root / "split.json").read_bytes()
    split_sha = hash_bytes(split_bytes)
    conversion_report = load_strict_json(conversion_report_path)
    models = locked._conversion_models(conversion_report, conversion_report_path)
    detector = models[DETECTION_MODEL_ID]
    recognizer = models[RECOGNITION_MODEL_ID]
    detection_sha = str(detector["onnx"]["sha256"])
    recognition_sha = str(recognizer["onnx"]["sha256"])
    detector_session = session_factory(Path(detector["resolved_onnx_path"]))
    recognizer_session = session_factory(Path(recognizer["resolved_onnx_path"]))
    alphabet = locked.read_character_alphabet(source_root / f"{RECOGNITION_MODEL_ID}_infer" / "inference.yml")
    records: list[dict[str, Any]] = []
    detection_total_ms = 0.0
    recognition_total_ms = 0.0
    for case in split["cases"]:
        from PIL import Image

        source = frozen_root / case["source_path"]
        with Image.open(source) as loaded:
            image = loaded.convert("RGB")
        masked_bgr = _masked_bgr(image, case["mask_rectangles"])
        _require(hash_bytes(masked_bgr) == case["detector_image_bgr_sha256"], "Detector pixels changed before inference.")
        detection = detect_regions(detector_session, masked_bgr, image.width, image.height)
        detection_total_ms += detection.duration_ms
        ranked = [] if case["truth_bbox"] is None else sorted(
            ((_iou(region.bounds, case["truth_bbox"]), region) for region in detection.final_regions),
            key=lambda item: (-item[0], item[1].region_id),
        )
        matched = ranked[0][1] if ranked and ranked[0][0] >= 0.50 else None
        recognition = None
        if matched is not None:
            legacy_region = locked.Region(
                int(matched.bounds.left), int(matched.bounds.top),
                int(matched.bounds.right), int(matched.bounds.bottom), matched.confidence,
            )
            recognition = locked.recognize_region(recognizer_session, image, legacy_region, alphabet)
            recognition_total_ms += recognition.duration_ms
            predicted_text = recognition.text
            predicted_role = locked.classify_role(legacy_region, image.width, image.height)
        else:
            predicted_text = ""
            predicted_role = "other"
        detected_count = 1 if matched is not None else 0
        false_count = len(detection.final_regions) - detected_count
        records.append(
            {
                "case_id": case["case_id"],
                "source_sha256": case["source_sha256"],
                "source_bgr_sha256": case["source_bgr_sha256"],
                "detector_image_bgr_sha256": case["detector_image_bgr_sha256"],
                "composition_id": COMPOSITION_ID,
                "predicted_text": predicted_text,
                "predicted_role": predicted_role,
                "detected_region_count": detected_count,
                "false_region_count": false_count,
                "duplicate_region_count": _duplicate_count(detection.final_regions),
                "model_regions": [_region_json(region) for region in detection.model_regions],
                "structure_candidates": [_candidate_json(candidate) for candidate in detection.structure_candidates],
                "consensus_matches": [
                    {"model_region_id": model, "candidate_region_id": candidate, "overlap_coefficient": overlap}
                    for model, candidate, overlap in detection.matches
                ],
                "final_regions": [_region_json(region) for region in detection.final_regions],
                "detector_input_tensor": _tensor_json(detection.input_sha256, detection.input_shape),
                "detector_output_tensor": _tensor_json(detection.output_sha256, detection.output_shape),
                "recognizer_executed": recognition is not None,
                "recognizer_input_tensor": recognition.input_tensor.to_json() if recognition is not None else None,
                "recognizer_output_tensor": recognition.output_tensor.to_json() if recognition is not None else None,
            }
        )
    core = {
        "schema": CORE_SCHEMA,
        "profile": PROFILE,
        "provider": "cpu",
        "composition_id": COMPOSITION_ID,
        "sealed_split_sha256": split_sha,
        "fixture_archive_sha256": verification["fixture_archive_sha256"],
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "records": records,
    }
    core_bytes = canonical_json_bytes(core)
    core_sha = hash_bytes(core_bytes)
    marker_evaluated, marker_counts, marker_bytes, marker_blockers = _missing_marker_evidence(split["cases"])
    final_records = [
        {**record, "marker_creation_count": marker_counts[str(record["case_id"])]}
        for record in records
    ]
    predictions = {
        "schema": PREDICTIONS_SCHEMA,
        "profile": PROFILE,
        "provider": "cpu",
        "composition_id": COMPOSITION_ID,
        "sealed_split_sha256": split_sha,
        "fixture_archive_sha256": verification["fixture_archive_sha256"],
        "core_predictions_sha256": core_sha,
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "records": final_records,
    }
    prediction_bytes = canonical_json_bytes(predictions)
    prediction_sha = hash_bytes(prediction_bytes)
    detection_parity = parity_runner(detector, source_root)
    recognition_parity = parity_runner(recognizer, source_root)
    _require(len(detection_parity) >= 16 and len(recognition_parity) >= 16, "Direct parity requires 16 pairs per model.")
    import numpy as np
    import onnxruntime as ort
    import PIL

    runtime = {
        "schema": RUNTIME_SCHEMA,
        "profile": PROFILE,
        "provider": "cpu",
        "composition_id": COMPOSITION_ID,
        "detection_executed": True,
        "recognition_executed": True,
        "evaluator_source_sha256": hash_file(metrics_evaluator_path),
        "workflow_source_sha256": hash_file(Path(__file__)),
        "sealed_split_sha256": split_sha,
        "core_predictions_sha256": core_sha,
        "predictions_sha256": prediction_sha,
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "execution_provenance": {
            "conversion_report_sha256": hash_file(conversion_report_path),
            "fixture_archive_sha256": verification["fixture_archive_sha256"],
            "python_executable_sha256": hash_file(Path(sys.executable)),
            "python_implementation": platform.python_implementation(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "onnxruntime_version": ort.__version__,
            "pillow_version": PIL.__version__,
            "opencv_version": __import__("cv2").__version__,
            "onnxruntime_providers": ["CPUExecutionProvider"],
        },
        "detection_parity": detection_parity,
        "recognition_parity": recognition_parity,
        "timing_ms": {
            "detection_total": detection_total_ms,
            "recognition_total": recognition_total_ms,
            "total": detection_total_ms + recognition_total_ms,
        },
    }
    runtime_bytes = canonical_json_bytes(runtime)
    truth = {str(case["case_id"]): case for case in split["cases"]}
    predicted = {str(record["case_id"]): record for record in final_records}

    def metrics_for(partition: str) -> Any:
        return evaluate_partition(
            (
                str(case["truth_text"]),
                str(predicted[case_id]["predicted_text"]),
                str(case["truth_role"]),
                str(predicted[case_id]["predicted_role"]),
            )
            for case_id, case in truth.items()
            if case["partition"] == partition and case["kind"] == "text"
        )

    validation = metrics_for("validation")
    sealed = metrics_for("sealed_test")
    detection_exact = sum(
        record["detected_region_count"] == truth[str(record["case_id"])]["expected_region_count"]
        and record["false_region_count"] == 0
        for record in final_records
    ) / len(final_records)
    metrics = {
        "validation_exact_match": validation.exact_match,
        "validation_cer": validation.character_error_rate,
        "validation_role_accuracy": validation.role_accuracy,
        "sealed_test_exact_match": sealed.exact_match,
        "sealed_test_cer": sealed.character_error_rate,
        "sealed_test_role_accuracy": sealed.role_accuracy,
        "onnx_max_abs_error": max(
            abs(float(item["reference"]) - float(item["onnx"]))
            for item in detection_parity + recognition_parity
        ),
        "detection_exact_rate": detection_exact,
        "duplicate_region_count": sum(int(record["duplicate_region_count"]) for record in final_records),
        "exclusion_false_region_count": sum(
            int(record["false_region_count"])
            for record in final_records
            if truth[str(record["case_id"])]["kind"] == "exclusion"
        ),
        "marker_creation_count": sum(marker_counts.values()),
    }
    blockers = [*marker_blockers, *_threshold_blockers(metrics)]
    approved = marker_evaluated and not blockers
    evaluator_bytes = metrics_evaluator_path.read_bytes()
    workflow_bytes = Path(__file__).read_bytes()
    resources: dict[str, Any] = {
        "gate_protocol": _embedded_resource("application/json", protocol_path.read_bytes(), "Gate protocol"),
        "evaluator_source": _embedded_resource("text/x-python", evaluator_bytes, "Metrics evaluator"),
        "workflow_source": _embedded_resource("text/x-python", workflow_bytes, "Execution workflow"),
        "sealed_split": _embedded_resource("application/json", split_bytes, "Sealed split"),
        "fixture_archive": _embedded_resource("application/zip", verification["fixture_archive_bytes"], "Fixture archive"),
        "core_predictions": _embedded_resource("application/json", core_bytes, "Core predictions"),
        "predictions": _embedded_resource("application/json", prediction_bytes, "Predictions"),
        "runtime_results": _embedded_resource("application/json", runtime_bytes, "Runtime results"),
    }
    if marker_bytes is not None:
        resources["marker_creation_results"] = _embedded_resource(
            "application/json", marker_bytes, "Marker creation results"
        )
    report = {
        "schema": REPORT_SCHEMA,
        "profile": PROFILE,
        "status": "pass" if approved else "fail",
        "scope": "public_synthetic_sealed",
        "release_eligible": approved,
        "production_approval": approved,
        "private_data": False,
        "chandler_used": False,
        "marker_creation_evaluated": marker_evaluated,
        "provider": "cpu",
        "coordinate_space": "original_pixels",
        "composition_id": COMPOSITION_ID,
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "protocol_sha256": hash_file(protocol_path),
        "evaluator_source_sha256": hash_bytes(evaluator_bytes),
        "workflow_source_sha256": hash_bytes(workflow_bytes),
        "sealed_split_sha256": split_sha,
        "fixture_archive_sha256": verification["fixture_archive_sha256"],
        "core_predictions_sha256": core_sha,
        "predictions_sha256": prediction_sha,
        "runtime_results_sha256": hash_bytes(runtime_bytes),
        **metrics,
        "blockers": blockers,
        "reviewed_resources": resources,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "core-predictions.json").write_bytes(core_bytes)
    (output_root / "predictions.json").write_bytes(prediction_bytes)
    (output_root / "runtime-results.json").write_bytes(runtime_bytes)
    (output_root / "report.json").write_bytes(canonical_json_bytes(report))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-root", required=True, type=Path)
    freeze.add_argument("--protocol", type=Path, default=_default_protocol())
    freeze.add_argument("--metrics-evaluator", type=Path, default=_default_metrics_evaluator())
    freeze.add_argument(
        "--font",
        type=Path,
        default=_repo_root() / "src" / "GraphReader.App" / "Assets" / "Fonts" / "NotoSans-Regular.ttf",
    )
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--frozen-root", required=True, type=Path)
    evaluate.add_argument("--protocol", type=Path, default=_default_protocol())
    evaluate.add_argument("--metrics-evaluator", type=Path, default=_default_metrics_evaluator())
    evaluate.add_argument("--conversion-report", required=True, type=Path)
    evaluate.add_argument("--source-root", required=True, type=Path)
    evaluate.add_argument("--output-root", required=True, type=Path)
    verify = subparsers.add_parser("verify-freeze")
    verify.add_argument("--frozen-root", required=True, type=Path)
    verify.add_argument("--protocol", type=Path, default=_default_protocol())
    verify.add_argument("--metrics-evaluator", type=Path, default=_default_metrics_evaluator())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_split(args.output_root, args.protocol, args.metrics_evaluator, args.font)
            print(json.dumps({key: value for key, value in result.items() if key != "split"}, sort_keys=True))
        elif args.command == "verify-freeze":
            result = verify_frozen_split(args.frozen_root, args.protocol, args.metrics_evaluator)
            print(json.dumps({"status": "pass", "case_count": len(result["split"]["cases"])}, sort_keys=True))
        else:
            report = evaluate_official_candidate(
                args.frozen_root,
                args.protocol,
                args.metrics_evaluator,
                args.conversion_report,
                args.source_root,
                args.output_root,
            )
            print(json.dumps({"status": report["status"], "production_approval": report["production_approval"]}, sort_keys=True))
            return 0 if report["production_approval"] else 2
    except (ProductionGateError, OSError, ValueError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
