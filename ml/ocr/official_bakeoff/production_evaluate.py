# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze and evaluate the official PP-OCRv5 production candidate fail closed.

The generated images and model outputs are intentionally written beneath the
ignored ``runs`` tree. The tracked protocol freezes the renderer, degradations,
case counts, roles, thresholds, and one-evaluation budget before inference.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence

from ml.ocr.production_gate import evaluate_partition


PROFILE = "graphreader-ocr-public-gate-v1"
SPLIT_SCHEMA = "graphreader.ocr-sealed-split.v1"
PREDICTIONS_SCHEMA = "graphreader.ocr-predictions.v1"
RUNTIME_SCHEMA = "graphreader.ocr-runtime-results.v1"
REPORT_SCHEMA = "graphreader.ocr-production-gate.v1"
MARKER_SCHEMA = "graphreader.ocr-marker-creation-results.v1"
EXPECTED_EVALUATOR_SHA256 = (
    "cc354ec53e4d0ecc5eab7dcf6243e5538e39f043057a949fd3d6ce84a83d50ee"
)
DETECTION_MODEL_ID = "PP-OCRv5_mobile_det"
RECOGNITION_MODEL_ID = "en_PP-OCRv5_mobile_rec"
ALLOWED_ROLES = {
    "x_tick",
    "y_tick",
    "phase_header",
    "annotation",
    "participant",
    "other",
}
TEXT_ROLES = ("x_tick", "y_tick", "phase_header", "annotation", "participant")
FAMILIES = ("integer", "decimal", "negative", "percentage", "ambiguity")
THRESHOLDS = {
    "validation_exact_match": 0.90,
    "validation_cer": 0.05,
    "validation_role_accuracy": 0.90,
    "sealed_test_exact_match": 0.90,
    "sealed_test_cer": 0.05,
    "sealed_test_role_accuracy": 0.90,
    "onnx_max_abs_error": 1e-4,
    "detection_exact_rate": 1.0,
    "marker_creation_count": 0,
}


class ProductionGateError(ValueError):
    """Raised when production-gate evidence is incomplete or inconsistent."""


class DuplicateJsonKeyError(ProductionGateError):
    """Raised when a checksum-reviewed JSON object repeats a key."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionGateError(f"Cannot read reviewed JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProductionGateError(f"Reviewed JSON root must be an object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionGateError(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_protocol() -> Path:
    return Path(__file__).with_name("PRODUCTION_GATE_PROTOCOL.json")


def _default_evaluator() -> Path:
    return _repo_root() / "ml" / "ocr" / "production_gate.py"


def _resolve_fonts(font_root: Path, names: Sequence[str]) -> list[Path]:
    fonts = [(font_root / name).resolve() for name in names]
    missing = [str(path) for path in fonts if not path.is_file()]
    if missing:
        raise ProductionGateError(f"Frozen renderer fonts are missing: {missing}")
    return fonts


def _ambiguity_pairs(protocol: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = protocol.get("ambiguity_policy", {}).get("pairs")
    _require(isinstance(pairs, list) and pairs, "Protocol ambiguity pairs are missing.")
    result: list[tuple[str, str]] = []
    for pair in pairs:
        _require(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(item, str) and item for item in pair),
            "Every ambiguity pair must contain display text and normalized truth.",
        )
        result.append((pair[0], pair[1]))
    return result


def _case_text(family: str, family_index: int, protocol: dict[str, Any]) -> tuple[str, str]:
    if family == "integer":
        value = str((family_index * 7 + 3) % 101)
        return value, value
    if family == "decimal":
        value = f"{(family_index * 3 + 1) % 10}.{(family_index * 7 + 2) % 10}"
        return value, value
    if family == "negative":
        value = f"-{((family_index * 11) % 99) + 1}"
        return value, value
    if family == "percentage":
        value = f"{(family_index * 13 + 5) % 101}%"
        return value, value
    if family == "ambiguity":
        pairs = _ambiguity_pairs(protocol)
        return pairs[family_index % len(pairs)]
    raise ProductionGateError(f"Unsupported OCR family: {family}")


ROLE_ANCHORS: dict[str, tuple[str, int, int]] = {
    "phase_header": ("center", 192, 12),
    "y_tick": ("right", 68, 78),
    "annotation": ("center", 278, 78),
    "x_tick": ("center", 192, 148),
    "participant": ("right", 360, 148),
}


def _apply_degradation(image: Any, family: str) -> Any:
    from PIL import Image, ImageFilter

    if family == "clean":
        return image
    if family == "soft_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=0.35))
    if family == "downsample_restore":
        width, height = image.size
        reduced = image.resize((width * 3 // 4, height * 3 // 4), Image.Resampling.LANCZOS)
        return reduced.resize((width, height), Image.Resampling.LANCZOS)
    if family == "jpeg_roundtrip":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92, subsampling=0, optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    raise ProductionGateError(f"Unsupported degradation family: {family}")


def _draw_graph_context(draw: Any) -> None:
    draw.line((78, 35, 78, 142), fill=(215, 215, 215), width=1)
    draw.line((78, 142, 360, 142), fill=(215, 215, 215), width=1)


def _draw_text_case(
    display_text: str,
    role: str,
    font_path: Path,
    font_size: int,
    degradation: str,
    width: int,
    height: int,
) -> tuple[bytes, list[int]]:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    _draw_graph_context(draw)
    font = ImageFont.truetype(str(font_path), font_size)
    alignment, anchor_x, anchor_y = ROLE_ANCHORS[role]
    raw = draw.textbbox((0, 0), display_text, font=font, stroke_width=0)
    text_width = raw[2] - raw[0]
    text_height = raw[3] - raw[1]
    if alignment == "center":
        left = anchor_x - text_width // 2
    elif alignment == "right":
        left = anchor_x - text_width
    else:
        left = anchor_x
    top = anchor_y
    draw.text((left, top), display_text, fill=(20, 20, 20), font=font, stroke_width=0)
    bbox = [left + raw[0], top + raw[1], text_width, text_height]
    image = _apply_degradation(image, degradation)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue(), bbox


def _draw_exclusion_case(
    exclusion_index: int,
    degradation: str,
    width: int,
    height: int,
) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    kind = exclusion_index % 5
    if kind == 0:
        draw.line((58, 28, 58, 158), fill=(20, 20, 20), width=2)
        draw.line((58, 158, 348, 158), fill=(20, 20, 20), width=2)
        for x in range(82, 330, 32):
            draw.line((x, 154, x, 162), fill=(20, 20, 20), width=2)
    elif kind == 1:
        draw.line((192, 28, 192, 158), fill=(30, 30, 30), width=2)
        draw.line((106, 35, 192, 35), fill=(30, 30, 30), width=2)
        draw.line((106, 35, 106, 24), fill=(30, 30, 30), width=2)
    elif kind == 2:
        points = [(92, 120), (152, 84), (220, 104), (292, 62)]
        draw.line(points, fill=(25, 25, 25), width=3)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(20, 20, 20), width=2)
    elif kind == 3:
        for x, y in ((96, 118), (154, 82), (224, 102), (300, 58)):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(20, 20, 20))
    else:
        draw.line((84, 142, 310, 54), fill=(25, 25, 25), width=2)
        draw.line((194, 34, 194, 162), fill=(25, 25, 25), width=2)
        draw.polygon(((310, 54), (297, 54), (305, 67)), fill=(25, 25, 25))
    image = _apply_degradation(image, degradation)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _write_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(value)


def build_fixture_archive(root: Path, cases: Sequence[dict[str, Any]]) -> bytes:
    """Build a deterministic stored ZIP containing every checksum-bound fixture."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for case in sorted(cases, key=lambda value: str(value["source_path"])):
            source_path = str(case["source_path"])
            source = _case_source_path(root.resolve(), source_path)
            source_bytes = source.read_bytes()
            _require(hash_bytes(source_bytes) == case["source_sha256"], f"Fixture changed: {source}")
            info = zipfile.ZipInfo(source_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source_bytes)
    return output.getvalue()


def verify_fixture_archive(
    archive_bytes: bytes,
    cases: Sequence[dict[str, Any]],
) -> None:
    expected = {str(case["source_path"]): str(case["source_sha256"]) for case in cases}
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Fixture archive contains duplicate paths.")
        _require(set(names) == set(expected), "Fixture archive inventory does not match the split.")
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            _require(
                not pure.is_absolute()
                and ".." not in pure.parts
                and "." not in pure.parts
                and not info.is_dir(),
                f"Fixture archive path is unsafe: {info.filename}",
            )
            value = archive.read(info)
            _require(hash_bytes(value) == expected[info.filename], f"Archived fixture changed: {info.filename}")


def freeze_split(
    output_root: Path,
    protocol_path: Path,
    evaluator_path: Path,
    font_root: Path,
) -> dict[str, Any]:
    """Create a new immutable split. Existing output is never overwritten."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise ProductionGateError(f"Frozen split output already exists: {output_root}")
    protocol = load_strict_json(protocol_path)
    _require(protocol.get("profile") == PROFILE, "Frozen protocol profile is invalid.")
    _require(protocol.get("status") == "frozen_before_inference", "Protocol is not frozen.")
    _require(protocol.get("scope") == "public_synthetic", "Protocol scope must be public synthetic.")
    _require(protocol.get("private_data") is False, "Private data is forbidden in the OCR split.")
    _require(protocol.get("chandler_used") is False, "Chandler is forbidden in the OCR split.")
    _require(
        protocol.get("selection_locked_before_inference") is True,
        "Selection must be locked before inference.",
    )
    evaluator_sha = hash_file(evaluator_path)
    _require(
        evaluator_sha == EXPECTED_EVALUATOR_SHA256,
        "The C#-frozen OCR metric evaluator changed without a coordinated gate review.",
    )
    image_config = protocol.get("image")
    _require(isinstance(image_config, dict), "Protocol image configuration is missing.")
    width = int(image_config.get("width", 0))
    height = int(image_config.get("height", 0))
    font_size = int(image_config.get("font_size", 0))
    font_names = image_config.get("font_files")
    _require(
        width > 0 and height > 0 and font_size > 0 and isinstance(font_names, list),
        "Protocol image configuration is invalid.",
    )
    fonts = _resolve_fonts(font_root, [str(name) for name in font_names])
    degradations = protocol.get("degradation_families")
    _require(isinstance(degradations, list) and degradations, "No degradation families are frozen.")
    partitions = protocol.get("partitions")
    _require(isinstance(partitions, dict), "Protocol partitions are missing.")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=str(output_root.parent)))
    try:
        assets = temporary / "assets"
        assets.mkdir(parents=True)
        cases: list[dict[str, Any]] = []
        for partition_index, partition in enumerate(("validation", "sealed_test")):
            settings = partitions.get(partition)
            _require(isinstance(settings, dict), f"Protocol partition is missing: {partition}")
            text_count = int(settings.get("text_cases", 0))
            exclusion_count = int(settings.get("exclusion_cases", 0))
            _require(
                text_count == 100 and exclusion_count == 10,
                f"Partition {partition} must freeze 100 text and 10 exclusion cases.",
            )
            family_positions = {family: 0 for family in FAMILIES}
            for index in range(text_count):
                family = FAMILIES[index % len(FAMILIES)]
                family_index = family_positions[family]
                family_positions[family] += 1
                display_text, truth_text = _case_text(family, family_index + partition_index * 20, protocol)
                role = TEXT_ROLES[(index // len(FAMILIES) + index) % len(TEXT_ROLES)]
                font_index = (index + partition_index) % len(fonts)
                degradation_index = (index // len(fonts) + partition_index) % len(degradations)
                degradation = str(degradations[degradation_index])
                case_id = f"{partition}-text-{index:03d}"
                source_path = PurePosixPath("assets") / f"{case_id}.png"
                source_bytes, truth_bbox = _draw_text_case(
                    display_text,
                    role,
                    fonts[font_index],
                    font_size,
                    degradation,
                    width,
                    height,
                )
                _write_new(temporary / Path(*source_path.parts), source_bytes)
                cases.append(
                    {
                        "case_id": case_id,
                        "partition": partition,
                        "kind": "text",
                        "family": family,
                        "display_text": display_text,
                        "truth_text": truth_text,
                        "truth_role": role,
                        "expected_region_count": 1,
                        "truth_bbox": truth_bbox,
                        "source_path": source_path.as_posix(),
                        "source_sha256": hash_bytes(source_bytes),
                        "renderer_family": "windows_sans" if font_index == 0 else "windows_serif",
                        "degradation_family": degradation,
                    }
                )
            for index in range(exclusion_count):
                degradation = str(degradations[(index + partition_index) % len(degradations)])
                case_id = f"{partition}-exclusion-{index:03d}"
                source_path = PurePosixPath("assets") / f"{case_id}.png"
                source_bytes = _draw_exclusion_case(index, degradation, width, height)
                _write_new(temporary / Path(*source_path.parts), source_bytes)
                cases.append(
                    {
                        "case_id": case_id,
                        "partition": partition,
                        "kind": "exclusion",
                        "family": "exclusion",
                        "display_text": "",
                        "truth_text": "",
                        "truth_role": "other",
                        "expected_region_count": 0,
                        "truth_bbox": [],
                        "source_path": source_path.as_posix(),
                        "source_sha256": hash_bytes(source_bytes),
                        "renderer_family": "procedural_graph_structure",
                        "degradation_family": degradation,
                    }
                )
        fixture_archive_bytes = build_fixture_archive(temporary, cases)
        fixture_archive_sha256 = hash_bytes(fixture_archive_bytes)
        _write_new(temporary / "fixtures.zip", fixture_archive_bytes)
        split = {
            "schema": SPLIT_SCHEMA,
            "profile": PROFILE,
            "scope": "public_synthetic",
            "sealed": True,
            "selection_locked_before_inference": True,
            "private_data": False,
            "chandler_used": False,
            "evaluator_source_sha256": evaluator_sha,
            "fixture_archive_sha256": fixture_archive_sha256,
            "cases": cases,
        }
        split_bytes = canonical_json_bytes(split)
        _write_new(temporary / "split.json", split_bytes)
        record = {
            "schema": "graphreader.ocr-split-freeze-record.v1",
            "profile": PROFILE,
            "status": "frozen_before_inference",
            "scope": "public_synthetic",
            "selection_locked_before_inference": True,
            "private_data": False,
            "chandler_used": False,
            "protocol_path": str(protocol_path.resolve()),
            "protocol_sha256": hash_file(protocol_path),
            "workflow_source_sha256": hash_file(Path(__file__)),
            "evaluator_source_sha256": evaluator_sha,
            "sealed_split_sha256": hash_bytes(split_bytes),
            "fixture_archive_sha256": fixture_archive_sha256,
            "fixture_archive_bytes": len(fixture_archive_bytes),
            "asset_count": len(cases),
            "font_files": [
                {"path": str(path), "sha256": hash_file(path)} for path in fonts
            ],
            "experiment_budget": protocol.get("experiment_budget"),
        }
        _write_new(temporary / "freeze-record.json", canonical_json_bytes(record))
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_frozen_split(output_root, protocol_path, evaluator_path)


def _case_source_path(root: Path, raw: Any) -> Path:
    _require(isinstance(raw, str) and raw, "OCR case source_path must be nonempty.")
    pure = PurePosixPath(raw)
    _require(
        not pure.is_absolute() and ".." not in pure.parts and "." not in pure.parts,
        f"OCR source_path is unsafe: {raw}",
    )
    path = (root / Path(*pure.parts)).resolve()
    _require(path.is_relative_to(root.resolve()), f"OCR source_path escapes the frozen root: {raw}")
    return path


def _csharp_contract_blockers(split: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for case in split.get("cases", []):
        if case.get("kind") == "text" and case.get("family") == "ambiguity":
            truth = case.get("truth_text")
            display = case.get("display_text")
            if isinstance(truth, str) and isinstance(display, str) and truth != display:
                blockers.append(
                    "ProductionOcrApprovalGate.TruthMatchesFamily requires ambiguity truth_text "
                    "to contain O/o/l/I and cannot score separately rendered confusions against "
                    "normalized numeric truth_text."
                )
                break
    return blockers


def verify_frozen_split(
    frozen_root: Path,
    protocol_path: Path,
    evaluator_path: Path,
) -> dict[str, Any]:
    root = frozen_root.resolve()
    split_path = root / "split.json"
    record_path = root / "freeze-record.json"
    fixture_archive_path = root / "fixtures.zip"
    _require(
        split_path.is_file() and record_path.is_file() and fixture_archive_path.is_file(),
        "Frozen split metadata is incomplete.",
    )
    split = load_strict_json(split_path)
    record = load_strict_json(record_path)
    protocol = load_strict_json(protocol_path)
    _require(split.get("schema") == SPLIT_SCHEMA, "Frozen split schema is invalid.")
    _require(split.get("profile") == PROFILE, "Frozen split profile is invalid.")
    _require(split.get("scope") == "public_synthetic", "Frozen split scope is invalid.")
    _require(split.get("sealed") is True, "Frozen split is not sealed.")
    _require(split.get("selection_locked_before_inference") is True, "Split selection is not locked.")
    _require(split.get("private_data") is False, "Frozen split includes private data.")
    _require(split.get("chandler_used") is False, "Frozen split includes Chandler.")
    evaluator_sha = hash_file(evaluator_path)
    _require(evaluator_sha == EXPECTED_EVALUATOR_SHA256, "Frozen metric evaluator changed.")
    _require(split.get("evaluator_source_sha256") == evaluator_sha, "Split evaluator hash changed.")
    _require(record.get("sealed_split_sha256") == hash_file(split_path), "Split hash changed.")
    _require(
        split.get("fixture_archive_sha256") == hash_file(fixture_archive_path)
        and record.get("fixture_archive_sha256") == hash_file(fixture_archive_path),
        "Fixture archive hash changed.",
    )
    _require(record.get("protocol_sha256") == hash_file(protocol_path), "Protocol hash changed.")
    _require(
        record.get("workflow_source_sha256") == hash_file(Path(__file__)),
        "Freeze/evaluation workflow source changed after the split was sealed.",
    )
    cases = split.get("cases")
    _require(isinstance(cases, list), "Frozen split cases are missing.")
    ids: set[str] = set()
    expected_assets: set[Path] = set()
    text_counts = {"validation": 0, "sealed_test": 0}
    exclusion_count = 0
    families: set[str] = set()
    for case in cases:
        _require(isinstance(case, dict), "Every OCR case must be an object.")
        case_id = case.get("case_id")
        _require(isinstance(case_id, str) and case_id not in ids, "OCR case IDs must be unique.")
        ids.add(case_id)
        partition = case.get("partition")
        kind = case.get("kind")
        role = case.get("truth_role")
        _require(partition in text_counts, f"Invalid OCR partition: {partition}")
        _require(kind in {"text", "exclusion"}, f"Invalid OCR case kind: {kind}")
        _require(role in ALLOWED_ROLES, f"Invalid OCR role: {role}")
        source = _case_source_path(root, case.get("source_path"))
        _require(source.is_file(), f"Frozen OCR source is missing: {source}")
        _require(case.get("source_sha256") == hash_file(source), f"Frozen OCR source changed: {source}")
        expected_assets.add(source)
        if kind == "text":
            display = case.get("display_text")
            truth = case.get("truth_text")
            _require(isinstance(display, str) and display, "OCR display_text must be nonempty.")
            _require(isinstance(truth, str) and truth, "OCR truth_text must be nonempty.")
            _require(case.get("expected_region_count") == 1, "Text cases require one region.")
            bbox = case.get("truth_bbox")
            _require(
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, int) and value >= 0 for value in bbox),
                "Text cases require an integer truth_bbox.",
            )
            family = case.get("family")
            _require(family in FAMILIES, f"Invalid OCR text family: {family}")
            families.add(str(family))
            text_counts[str(partition)] += 1
        else:
            _require(case.get("display_text") == "" and case.get("truth_text") == "", "Exclusion text must be empty.")
            _require(role == "other" and case.get("expected_region_count") == 0, "Exclusion metadata is invalid.")
            exclusion_count += 1
    _require(text_counts == {"validation": 100, "sealed_test": 100}, "Frozen text counts changed.")
    _require(exclusion_count == 20, "Frozen exclusion count changed.")
    _require(families == set(FAMILIES), "Frozen OCR families are incomplete.")
    actual_assets = {path.resolve() for path in (root / "assets").glob("*.png")}
    _require(actual_assets == expected_assets, "Frozen OCR asset inventory contains missing or extra files.")
    _require(record.get("asset_count") == len(cases), "Freeze record asset count changed.")
    fixture_archive_bytes = fixture_archive_path.read_bytes()
    _require(record.get("fixture_archive_bytes") == len(fixture_archive_bytes), "Fixture archive size changed.")
    verify_fixture_archive(fixture_archive_bytes, cases)
    return {
        "status": "frozen_split_verified",
        "sealed_split_sha256": hash_file(split_path),
        "case_count": len(cases),
        "text_counts": text_counts,
        "exclusion_count": exclusion_count,
        "fixture_archive_sha256": hash_bytes(fixture_archive_bytes),
        "fixture_archive_bytes": fixture_archive_bytes,
        "csharp_contract_blockers": _csharp_contract_blockers(split),
        "protocol": protocol,
        "split": split,
    }


def _yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return json.loads(value)
    return value


def read_character_alphabet(inference_yaml: Path) -> str:
    lines = inference_yaml.read_text(encoding="utf-8").splitlines()
    start = next((index for index, line in enumerate(lines) if line.strip() == "character_dict:"), None)
    if start is None:
        raise ProductionGateError("Recognition inference.yml has no character_dict.")
    characters: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("  - "):
            character = _yaml_scalar(line[4:])
            _require(len(character) == 1, f"Recognition alphabet item is not one character: {line}")
            characters.append(character)
        elif line.startswith("  ") and not line.strip():
            continue
        else:
            break
    # PaddleOCR's BaseRecLabelDecode enables the space class by default. The
    # inference YAML lists the explicit dictionary only, while the model emits
    # one additional space class and one CTC blank class.
    if " " not in characters:
        characters.append(" ")
    _require(characters and len(set(characters)) == len(characters), "Recognition alphabet is empty or duplicated.")
    return "".join(characters)


def _cpu_session(path: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ProductionGateError(f"OCR runtime is not CPU-only: {session.get_providers()}")
    _require(len(session.get_inputs()) == 1 and len(session.get_outputs()) == 1, "OCR ONNX model I/O is invalid.")
    return session


def _round_up_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, int(math.ceil(value / multiple) * multiple))


def detector_tensor(image: Any, maximum_side: int = 960) -> tuple[Any, tuple[int, int]]:
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    ratio = min(1.0, maximum_side / max(width, height))
    target_width = _round_up_multiple(max(1, int(round(width * ratio))))
    target_height = _round_up_multiple(max(1, int(round(height * ratio))))
    resized = rgb.resize((target_width, target_height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)[:, :, ::-1] / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    standard_deviation = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((array - mean) / standard_deviation).transpose(2, 0, 1)[None, :, :, :]
    return np.ascontiguousarray(tensor), (width, height)


@dataclass(frozen=True)
class Region:
    left: float
    top: float
    width: float
    height: float
    confidence: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


@dataclass(frozen=True)
class TensorEvidence:
    sha256: str
    dtype: str
    shape: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class DetectionResult:
    regions: tuple[Region, ...]
    duration_ms: float
    input_tensor: TensorEvidence
    output_tensor: TensorEvidence


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    duration_ms: float
    input_tensor: TensorEvidence
    output_tensor: TensorEvidence


def _tensor_evidence(value: Any) -> TensorEvidence:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return TensorEvidence(
        sha256=hash_bytes(array.tobytes(order="C")),
        dtype=str(array.dtype),
        shape=tuple(int(dimension) for dimension in array.shape),
    )


def _components(probabilities: Any, threshold: float = 0.30) -> list[tuple[int, int, int, int, int, float]]:
    import numpy as np

    mask = np.asarray(probabilities) >= threshold
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    result: list[tuple[int, int, int, int, int, float]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            visited[y, x] = True
            if not mask[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            left = right = x
            top = bottom = y
            count = 0
            confidence_sum = 0.0
            while queue:
                current_x, current_y = queue.popleft()
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)
                count += 1
                confidence_sum += float(probabilities[current_y, current_x])
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        neighbor_x = current_x + offset_x
                        neighbor_y = current_y + offset_y
                        if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                            continue
                        if visited[neighbor_y, neighbor_x]:
                            continue
                        visited[neighbor_y, neighbor_x] = True
                        if mask[neighbor_y, neighbor_x]:
                            queue.append((neighbor_x, neighbor_y))
            component_width = right - left + 1
            component_height = bottom - top + 1
            if count >= 3 and component_width >= 2 and component_height >= 2:
                result.append((left, top, right, bottom, count, confidence_sum / count))
    return result


def detect_regions(session: Any, image: Any) -> DetectionResult:
    import numpy as np

    tensor, original_size = detector_tensor(image)
    started = time.perf_counter()
    output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: tensor},
    )[0]
    duration_ms = (time.perf_counter() - started) * 1000.0
    probabilities = np.asarray(output, dtype=np.float32).squeeze()
    _require(probabilities.ndim == 2 and np.isfinite(probabilities).all(), "Detector output is invalid.")
    if float(probabilities.min()) < 0 or float(probabilities.max()) > 1:
        probabilities = 1.0 / (1.0 + np.exp(-probabilities))
    map_height, map_width = probabilities.shape
    original_width, original_height = original_size
    regions: list[Region] = []
    for left, top, right, bottom, _, confidence in _components(probabilities):
        if confidence < 0.60:
            continue
        component_width = right - left + 1
        component_height = bottom - top + 1
        area = component_width * component_height
        perimeter = 2.0 * (component_width + component_height)
        expansion = 0.0 if perimeter <= 0 else area * 1.5 / perimeter
        mapped_left = max(0.0, (left - expansion) * original_width / map_width)
        mapped_top = max(0.0, (top - expansion) * original_height / map_height)
        mapped_right = min(original_width, (right + 1 + expansion) * original_width / map_width)
        mapped_bottom = min(original_height, (bottom + 1 + expansion) * original_height / map_height)
        if mapped_right > mapped_left and mapped_bottom > mapped_top:
            regions.append(
                Region(
                    mapped_left,
                    mapped_top,
                    mapped_right - mapped_left,
                    mapped_bottom - mapped_top,
                    confidence,
                )
            )
    return DetectionResult(
        tuple(sorted(regions, key=lambda region: (region.top, region.left))),
        duration_ms,
        _tensor_evidence(tensor),
        _tensor_evidence(output),
    )


def recognition_tensor(image: Any, maximum_width: int = 320) -> Any:
    import numpy as np
    from PIL import Image

    bgr = np.asarray(image.convert("RGB"), dtype=np.float32)[:, :, ::-1]
    height, width = bgr.shape[:2]
    resized_width = min(maximum_width, max(1, int(math.ceil(48.0 * width / max(1, height)))))
    resized = Image.fromarray(bgr[:, :, ::-1].astype(np.uint8), "RGB").resize(
        (resized_width, 48), Image.Resampling.BILINEAR
    )
    resized_bgr = np.asarray(resized, dtype=np.float32)[:, :, ::-1] / 255.0
    normalized = (resized_bgr - 0.5) / 0.5
    tensor = np.zeros((1, 3, 48, maximum_width), dtype=np.float32)
    tensor[0, :, :, :resized_width] = normalized.transpose(2, 0, 1)
    return np.ascontiguousarray(tensor)


def decode_ctc(output: Any, alphabet: str) -> str:
    import numpy as np

    logits = np.asarray(output)
    _require(logits.ndim == 3 and logits.shape[0] == 1, "Recognizer output must be [1,time,class].")
    _require(logits.shape[2] == len(alphabet) + 1, "Recognizer alphabet does not match output classes.")
    classes = logits[0].argmax(axis=1).tolist()
    result: list[str] = []
    prior = -1
    for value in classes:
        if value != 0 and value != prior:
            result.append(alphabet[value - 1])
        prior = value
    return "".join(result)


def recognize_region(session: Any, image: Any, region: Region, alphabet: str) -> RecognitionResult:
    left = max(0, int(math.floor(region.left)))
    top = max(0, int(math.floor(region.top)))
    right = min(image.width, int(math.ceil(region.right)))
    bottom = min(image.height, int(math.ceil(region.bottom)))
    if right <= left or bottom <= top:
        raise ProductionGateError("Detected OCR crop is empty.")
    tensor = recognition_tensor(image.crop((left, top, right, bottom)))
    started = time.perf_counter()
    output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: tensor},
    )[0]
    duration_ms = (time.perf_counter() - started) * 1000.0
    return RecognitionResult(
        decode_ctc(output, alphabet),
        duration_ms,
        _tensor_evidence(tensor),
        _tensor_evidence(output),
    )


def intersection_over_union(region: Region, truth_bbox: Sequence[int]) -> float:
    left, top, width, height = truth_bbox
    right = left + width
    bottom = top + height
    intersection_width = max(0.0, min(region.right, right) - max(region.left, left))
    intersection_height = max(0.0, min(region.bottom, bottom) - max(region.top, top))
    intersection = intersection_width * intersection_height
    union = region.width * region.height + width * height - intersection
    return 0.0 if union <= 0 else intersection / union


def classify_role(region: Region, image_width: int, image_height: int) -> str:
    center_x = (region.left + region.right) / 2.0 / image_width
    center_y = (region.top + region.bottom) / 2.0 / image_height
    if center_y < 0.33 and 0.20 <= center_x <= 0.80:
        return "phase_header"
    if center_x < 0.26:
        return "y_tick"
    if center_y > 0.68 and center_x < 0.72:
        return "x_tick"
    if center_y > 0.68:
        return "participant"
    return "annotation"


def _conversion_models(report: dict[str, Any], conversion_report_path: Path) -> dict[str, dict[str, Any]]:
    _require(report.get("production_approved") is False, "Conversion report improperly claims production approval.")
    _require(report.get("release_ready") is False, "Conversion report improperly claims release readiness.")
    conversion = report.get("conversion")
    _require(isinstance(conversion, dict), "Conversion report is incomplete.")
    models = conversion.get("models")
    _require(isinstance(models, list), "Conversion model evidence is missing.")
    selected: dict[str, dict[str, Any]] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("model_id")
        if model_id not in {DETECTION_MODEL_ID, RECOGNITION_MODEL_ID}:
            continue
        _require(model.get("status") == "reproducible_conversion_and_raw_tensor_parity_passed", f"Conversion failed: {model_id}")
        onnx = model.get("onnx")
        parity = model.get("cpu_parity")
        _require(isinstance(onnx, dict) and isinstance(parity, dict), f"Conversion evidence is incomplete: {model_id}")
        path = Path(str(onnx.get("path")))
        if not path.is_absolute():
            path = (conversion_report_path.parent / path).resolve()
        _require(path.is_file(), f"Converted ONNX is missing: {path}")
        _require(onnx.get("sha256") == hash_file(path), f"Converted ONNX changed: {path}")
        _require(parity.get("passed") is True and parity.get("provider") == "CPUExecutionProvider", f"CPU parity is missing: {model_id}")
        _require(model.get("reproducibility", {}).get("byte_identical") is True, f"Conversion is not reproducible: {model_id}")
        selected[str(model_id)] = {**model, "resolved_onnx_path": path}
    _require(set(selected) == {DETECTION_MODEL_ID, RECOGNITION_MODEL_ID}, "Both official OCR models are required.")
    return selected


def _direct_parity_pairs(model: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    import numpy as np

    from ml.ocr.official_bakeoff import convert_models

    model_id = str(model["model_id"])
    shapes = convert_models.MODEL_INPUT_SHAPES[model_id]
    model_root = source_root / f"{model_id}_infer"
    onnx_path = Path(model["resolved_onnx_path"])
    _, _, paddle_run = convert_models._paddle_runner(model_root)
    _, _, onnx_run = convert_models._onnx_runner(onnx_path)
    rng = np.random.default_rng(convert_models.RNG_SEED)
    records: list[dict[str, Any]] = []
    for index in range(convert_models.PARITY_CASES):
        shape = shapes[index % len(shapes)]
        sample = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        reference = np.asarray(paddle_run(sample)[0])
        candidate = np.asarray(onnx_run(sample)[0])
        _require(reference.shape == candidate.shape, "Direct OCR parity output shapes differ.")
        errors = np.abs(reference - candidate).reshape(-1)
        maximum_index = int(errors.argmax())
        records.append(
            {
                "case": index,
                "input_sha256": hash_bytes(sample.tobytes(order="C")),
                "reference": float(reference.reshape(-1)[maximum_index]),
                "onnx": float(candidate.reshape(-1)[maximum_index]),
            }
        )
    return records


def _load_marker_evidence(
    path: Path | None,
    case_sources: dict[str, str],
    split_sha: str,
    detection_sha: str,
    recognition_sha: str,
    core_predictions_sha: str,
) -> tuple[bool, dict[str, int], bytes | None, list[str]]:
    case_ids = set(case_sources)
    if path is None:
        return False, {case_id: 0 for case_id in case_ids}, None, [
            "No checksum-bound downstream marker-creation evidence was supplied."
        ]
    evidence = load_strict_json(path)
    _require(evidence.get("schema") == MARKER_SCHEMA, "Marker evidence schema is invalid.")
    _require(evidence.get("profile") == PROFILE, "Marker evidence profile is invalid.")
    _require(evidence.get("provider") == "cpu", "Marker evidence must use CPU.")
    run_id = evidence.get("run_id")
    try:
        parsed_run_id = uuid.UUID(str(run_id))
    except (ValueError, TypeError, AttributeError) as error:
        raise ProductionGateError("Marker evidence run_id is not a UUID.") from error
    _require(str(parsed_run_id) == str(run_id).lower(), "Marker evidence run_id is not canonical.")
    _require(evidence.get("stage") == "markers", "Marker evidence must identify the marker stage.")
    _require(
        evidence.get("composition_id") == "production-ocr-to-marker-composed-v1",
        "Marker evidence composition is not the reviewed OCR-to-marker workflow.",
    )
    marker_model_id = evidence.get("marker_model_id")
    marker_model_sha = evidence.get("marker_model_sha256")
    _require(isinstance(marker_model_id, str) and marker_model_id, "Marker evidence model ID is missing.")
    _require(
        isinstance(marker_model_sha, str)
        and len(marker_model_sha) == 64
        and all(character in "0123456789abcdef" for character in marker_model_sha.lower()),
        "Marker evidence model SHA-256 is invalid.",
    )
    _require(evidence.get("sealed_split_sha256") == split_sha, "Marker evidence split hash changed.")
    _require(evidence.get("detection_model_sha256") == detection_sha, "Marker evidence detector hash changed.")
    _require(evidence.get("recognition_model_sha256") == recognition_sha, "Marker evidence recognizer hash changed.")
    _require(
        evidence.get("ocr_core_predictions_sha256") == core_predictions_sha,
        "Marker evidence is not bound to the exact OCR core predictions.",
    )
    records = evidence.get("records")
    _require(isinstance(records, list), "Marker evidence records are missing.")
    counts: dict[str, int] = {}
    for record in records:
        _require(isinstance(record, dict), "Marker evidence record is invalid.")
        case_id = record.get("case_id")
        source_sha = record.get("source_sha256")
        count = record.get("marker_creation_count")
        _require(
            isinstance(case_id, str)
            and case_id not in counts
            and source_sha == case_sources.get(case_id)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0,
            "Marker evidence contains invalid or duplicate values.",
        )
        counts[case_id] = count
    _require(set(counts) == case_ids, "Marker evidence case IDs do not match the frozen split.")
    return True, counts, path.read_bytes(), []


def _embedded(media_type: str, value: bytes) -> dict[str, Any]:
    return {
        "media_type": media_type,
        "encoding": "base64",
        "sha256": hash_bytes(value),
        "content_base64": base64.b64encode(value).decode("ascii"),
    }


def _threshold_blockers(metrics: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if metrics["validation_exact_match"] < THRESHOLDS["validation_exact_match"]:
        blockers.append("Validation exact match is below 0.90.")
    if metrics["validation_cer"] > THRESHOLDS["validation_cer"]:
        blockers.append("Validation CER exceeds 0.05.")
    if metrics["validation_role_accuracy"] < THRESHOLDS["validation_role_accuracy"]:
        blockers.append("Validation role accuracy is below 0.90.")
    if metrics["sealed_test_exact_match"] < THRESHOLDS["sealed_test_exact_match"]:
        blockers.append("Sealed-test exact match is below 0.90.")
    if metrics["sealed_test_cer"] > THRESHOLDS["sealed_test_cer"]:
        blockers.append("Sealed-test CER exceeds 0.05.")
    if metrics["sealed_test_role_accuracy"] < THRESHOLDS["sealed_test_role_accuracy"]:
        blockers.append("Sealed-test role accuracy is below 0.90.")
    if metrics["onnx_max_abs_error"] > THRESHOLDS["onnx_max_abs_error"]:
        blockers.append("Direct CPU ONNX parity exceeds 1e-4.")
    if metrics["detection_exact_rate"] != THRESHOLDS["detection_exact_rate"]:
        blockers.append("Detection exact rate is not 1.0.")
    if metrics["marker_creation_count"] != THRESHOLDS["marker_creation_count"]:
        blockers.append("Recognized text created marker candidates.")
    return blockers


def evaluate_official_candidate(
    frozen_root: Path,
    protocol_path: Path,
    evaluator_path: Path,
    conversion_report_path: Path,
    source_root: Path,
    output_root: Path,
    marker_evidence_path: Path | None = None,
    *,
    session_factory: Callable[[Path], Any] = _cpu_session,
    parity_runner: Callable[[dict[str, Any], Path], list[dict[str, Any]]] = _direct_parity_pairs,
) -> dict[str, Any]:
    if output_root.exists():
        raise ProductionGateError(
            "Official candidate evaluation output already exists; the one-evaluation budget forbids rerun."
        )
    verification = verify_frozen_split(frozen_root, protocol_path, evaluator_path)
    split = verification["split"]
    split_path = frozen_root.resolve() / "split.json"
    split_bytes = split_path.read_bytes()
    split_sha = hash_bytes(split_bytes)
    conversion_report = load_strict_json(conversion_report_path)
    models = _conversion_models(conversion_report, conversion_report_path)
    detector = models[DETECTION_MODEL_ID]
    recognizer = models[RECOGNITION_MODEL_ID]
    detection_path = Path(detector["resolved_onnx_path"])
    recognition_path = Path(recognizer["resolved_onnx_path"])
    detection_sha = str(detector["onnx"]["sha256"])
    recognition_sha = str(recognizer["onnx"]["sha256"])
    detector_session = session_factory(detection_path)
    recognition_session = session_factory(recognition_path)
    alphabet = read_character_alphabet(
        source_root / f"{RECOGNITION_MODEL_ID}_infer" / "inference.yml"
    )
    case_sources = {
        str(case["case_id"]): str(case["source_sha256"]) for case in split["cases"]
    }
    core_records: list[dict[str, Any]] = []
    detection_total_ms = 0.0
    recognition_total_ms = 0.0
    for case in split["cases"]:
        from PIL import Image

        source = _case_source_path(frozen_root.resolve(), case["source_path"])
        _require(hash_file(source) == case["source_sha256"], f"OCR source changed during inference: {source}")
        with Image.open(source) as loaded:
            image = loaded.convert("RGB")
        detection = detect_regions(detector_session, image)
        regions = detection.regions
        detection_total_ms += detection.duration_ms
        recognition: RecognitionResult | None = None
        if case["kind"] == "text":
            ranked = sorted(
                ((intersection_over_union(region, case["truth_bbox"]), region) for region in regions),
                key=lambda pair: pair[0],
                reverse=True,
            )
            matched = ranked[0][1] if ranked and ranked[0][0] >= 0.50 else None
            detected_count = 1 if matched is not None else 0
            false_count = len(regions) - detected_count
            if matched is None:
                predicted_text = ""
                predicted_role = "other"
            else:
                recognition = recognize_region(
                    recognition_session, image, matched, alphabet
                )
                predicted_text = recognition.text
                recognition_total_ms += recognition.duration_ms
                predicted_role = classify_role(matched, image.width, image.height)
        else:
            detected_count = 0
            false_count = len(regions)
            predicted_text = ""
            predicted_role = "other"
        core_records.append(
            {
                "case_id": case["case_id"],
                "source_sha256": case["source_sha256"],
                "predicted_text": predicted_text,
                "predicted_role": predicted_role,
                "detected_region_count": detected_count,
                "false_region_count": false_count,
                "detector_input_tensor": detection.input_tensor.to_json(),
                "detector_output_tensor": detection.output_tensor.to_json(),
                "recognizer_executed": recognition is not None,
                "recognizer_input_tensor": (
                    recognition.input_tensor.to_json() if recognition is not None else None
                ),
                "recognizer_output_tensor": (
                    recognition.output_tensor.to_json() if recognition is not None else None
                ),
            }
        )
    core_predictions = {
        "schema": "graphreader.ocr-core-predictions.v1",
        "profile": PROFILE,
        "provider": "cpu",
        "sealed_split_sha256": split_sha,
        "fixture_archive_sha256": verification["fixture_archive_sha256"],
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "records": core_records,
    }
    core_prediction_bytes = canonical_json_bytes(core_predictions)
    core_prediction_sha = hash_bytes(core_prediction_bytes)
    marker_evaluated, marker_counts, marker_bytes, marker_blockers = _load_marker_evidence(
        marker_evidence_path,
        case_sources,
        split_sha,
        detection_sha,
        recognition_sha,
        core_prediction_sha,
    )
    records = [
        {
            **record,
            "marker_creation_count": marker_counts[str(record["case_id"])],
        }
        for record in core_records
    ]
    predictions = {
        "schema": PREDICTIONS_SCHEMA,
        "profile": PROFILE,
        "provider": "cpu",
        "sealed_split_sha256": split_sha,
        "fixture_archive_sha256": verification["fixture_archive_sha256"],
        "core_predictions_sha256": core_prediction_sha,
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "records": records,
    }
    prediction_bytes = canonical_json_bytes(predictions)
    prediction_sha = hash_bytes(prediction_bytes)
    detection_parity = parity_runner(detector, source_root)
    recognition_parity = parity_runner(recognizer, source_root)
    _require(len(detection_parity) >= 16 and len(recognition_parity) >= 16, "Direct parity requires 16 cases per model.")
    import numpy as np
    import onnxruntime as ort
    import PIL

    runtime_results = {
        "schema": RUNTIME_SCHEMA,
        "profile": PROFILE,
        "provider": "cpu",
        "execution_provenance": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": hash_file(Path(sys.executable)),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "onnxruntime_version": ort.__version__,
            "pillow_version": PIL.__version__,
            "onnxruntime_providers": ["CPUExecutionProvider"],
            "conversion_report_sha256": hash_file(conversion_report_path),
            "workflow_source_sha256": hash_file(Path(__file__)),
            "fixture_archive_sha256": verification["fixture_archive_sha256"],
        },
        "detection_executed": True,
        "recognition_executed": True,
        "evaluator_source_sha256": hash_file(evaluator_path),
        "sealed_split_sha256": split_sha,
        "core_predictions_sha256": core_prediction_sha,
        "predictions_sha256": prediction_sha,
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "detection_parity": detection_parity,
        "recognition_parity": recognition_parity,
        "timing_ms": {
            "detection_total": detection_total_ms,
            "recognition_total": recognition_total_ms,
            "total": detection_total_ms + recognition_total_ms,
        },
    }
    runtime_bytes = canonical_json_bytes(runtime_results)
    truth_by_id = {str(case["case_id"]): case for case in split["cases"]}
    predictions_by_id = {str(record["case_id"]): record for record in records}

    def partition_metrics(partition: str) -> Any:
        return evaluate_partition(
            (
                str(case["truth_text"]),
                str(predictions_by_id[case_id]["predicted_text"]),
                str(case["truth_role"]),
                str(predictions_by_id[case_id]["predicted_role"]),
            )
            for case_id, case in truth_by_id.items()
            if case["partition"] == partition and case["kind"] == "text"
        )

    validation = partition_metrics("validation")
    sealed_test = partition_metrics("sealed_test")
    detection_exact = sum(
        record["detected_region_count"] == truth_by_id[str(record["case_id"])]["expected_region_count"]
        and record["false_region_count"] == 0
        for record in records
    ) / len(records)
    parity_maximum = max(
        abs(float(record["reference"]) - float(record["onnx"]))
        for record in detection_parity + recognition_parity
    )
    metrics = {
        "validation_exact_match": validation.exact_match,
        "validation_cer": validation.character_error_rate,
        "validation_role_accuracy": validation.role_accuracy,
        "sealed_test_exact_match": sealed_test.exact_match,
        "sealed_test_cer": sealed_test.character_error_rate,
        "sealed_test_role_accuracy": sealed_test.role_accuracy,
        "onnx_max_abs_error": parity_maximum,
        "detection_exact_rate": detection_exact,
        "marker_creation_count": sum(marker_counts.values()),
    }
    blockers = [
        *verification["csharp_contract_blockers"],
        *marker_blockers,
        *_threshold_blockers(metrics),
    ]
    approved = not blockers and marker_evaluated
    evaluator_bytes = evaluator_path.read_bytes()
    workflow_bytes = Path(__file__).read_bytes()
    reviewed_resources: dict[str, Any] = {
        "evaluator_source": _embedded("text/x-python", evaluator_bytes),
        "workflow_source": _embedded("text/x-python", workflow_bytes),
        "sealed_split": _embedded("application/json", split_bytes),
        "fixture_archive": _embedded(
            "application/zip", verification["fixture_archive_bytes"]
        ),
        "core_predictions": _embedded("application/json", core_prediction_bytes),
        "predictions": _embedded("application/json", prediction_bytes),
        "runtime_results": _embedded("application/json", runtime_bytes),
    }
    if marker_bytes is not None:
        reviewed_resources["marker_creation_results"] = _embedded("application/json", marker_bytes)
    report = {
        "schema": REPORT_SCHEMA,
        "profile": PROFILE,
        "status": "pass" if approved else "fail",
        "scope": "public_synthetic_sealed",
        "release_eligible": approved,
        "production_approval": approved,
        "private_data": False,
        "chandler_used": False,
        "provider": "cpu",
        "coordinate_space": "original_pixels",
        "detection_model_sha256": detection_sha,
        "recognition_model_sha256": recognition_sha,
        "evaluator_source_sha256": hash_bytes(evaluator_bytes),
        "workflow_source_sha256": hash_bytes(workflow_bytes),
        "sealed_split_sha256": split_sha,
        "fixture_archive_sha256": verification["fixture_archive_sha256"],
        "core_predictions_sha256": core_prediction_sha,
        "predictions_sha256": prediction_sha,
        "runtime_results_sha256": hash_bytes(runtime_bytes),
        **metrics,
        "marker_creation_evaluated": marker_evaluated,
        "blockers": blockers,
        "reviewed_resources": reviewed_resources,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "core-predictions.json").write_bytes(core_prediction_bytes)
    (output_root / "predictions.json").write_bytes(prediction_bytes)
    (output_root / "runtime-results.json").write_bytes(runtime_bytes)
    (output_root / "report.json").write_bytes(canonical_json_bytes(report))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="Freeze a new public synthetic split.")
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, default=_default_protocol())
    freeze.add_argument("--evaluator", type=Path, default=_default_evaluator())
    freeze.add_argument("--font-root", type=Path, default=Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts")
    verify = subparsers.add_parser("verify-freeze", help="Verify exact frozen split bytes.")
    verify.add_argument("--frozen-root", type=Path, required=True)
    verify.add_argument("--protocol", type=Path, default=_default_protocol())
    verify.add_argument("--evaluator", type=Path, default=_default_evaluator())
    evaluate = subparsers.add_parser("evaluate", help="Run one official-model evaluation.")
    evaluate.add_argument("--frozen-root", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, default=_default_protocol())
    evaluate.add_argument("--evaluator", type=Path, default=_default_evaluator())
    evaluate.add_argument("--conversion-report", type=Path, required=True)
    evaluate.add_argument("--source-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--marker-evidence", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_split(args.output_root, args.protocol, args.evaluator, args.font_root)
            print(f"PASS: frozen {result['case_count']} public synthetic OCR cases")
            print(f"split_sha256={result['sealed_split_sha256']}")
            if result["csharp_contract_blockers"]:
                print(f"BLOCKED: {result['csharp_contract_blockers'][0]}")
            return 0
        if args.command == "verify-freeze":
            result = verify_frozen_split(args.frozen_root, args.protocol, args.evaluator)
            print(f"PASS: {result['status']} {result['sealed_split_sha256']}")
            if result["csharp_contract_blockers"]:
                print(f"BLOCKED: {result['csharp_contract_blockers'][0]}")
            return 0
        report = evaluate_official_candidate(
            args.frozen_root,
            args.protocol,
            args.evaluator,
            args.conversion_report,
            args.source_root,
            args.output_root,
            args.marker_evidence,
        )
        print(json.dumps({key: report[key] for key in (*THRESHOLDS, "status", "blockers")}, indent=2))
        return 0 if report["status"] == "pass" else 2
    except (ProductionGateError, OSError, RuntimeError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
