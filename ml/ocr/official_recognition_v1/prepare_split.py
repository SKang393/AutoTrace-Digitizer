# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze fresh recognition-only selection and truth-hidden public crops."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from typing import Any
import zipfile

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "graphreader.ocr-official-recognition-split.v1"
SEED = 20260911
COUNTS = {"selection": 192, "sealed_public": 256}
FONT_PATHS = (
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)
NUMERIC_TEXTS = (
    "0", "1", "9", "10", "24", "33", "42", "75", "99", "100",
    "-1", "-8", "-0.7", "0.25", "3.5", "9.25", "42%", "75%", "100%",
)
WORD_CASES = (
    ("Chandler", "participant"),
    ("Smith", "participant"),
    ("Jordan", "participant"),
    ("Rivera", "participant"),
    ("Baseline", "phase_header"),
    ("Intervention", "phase_header"),
    ("Maintenance", "phase_header"),
    ("Generalization", "phase_header"),
    ("Session", "axis_title"),
    ("Percent", "axis_title"),
    ("Frequency", "axis_title"),
    ("Follow-up", "annotation"),
    ("Treatment", "annotation"),
    ("Probe", "annotation"),
    ("O o l I", "annotation"),
    ("Phase A", "phase_header"),
    ("Phase B", "phase_header"),
)
SELECTION_DEGRADATIONS = (
    "warm-paper-subpixel-v1",
    "horizontal-resample-v1",
    "soft-scan-v1",
    "jpeg-gray-v1",
)
PUBLIC_DEGRADATIONS = (
    "cool-paper-subpixel-v1",
    "vertical-resample-v1",
    "low-contrast-sharpen-v1",
    "jpeg-chroma-v1",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def hash_file(path: Path) -> str:
    return hash_bytes(path.read_bytes())


def _case_text(index: int) -> tuple[str, str, str]:
    if index % 4 != 3:
        text = NUMERIC_TEXTS[(index * 7 + 5) % len(NUMERIC_TEXTS)]
        return text, "numeric_text", "numeric"
    text, role = WORD_CASES[(index * 5 + 2) % len(WORD_CASES)]
    return text, role, "word"


def _degrade(image: Image.Image, family: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    if "horizontal-resample" in family:
        width, height = image.size
        reduced = image.resize((max(8, int(width * 0.73)), height), Image.Resampling.BICUBIC)
        return reduced.resize((width, height), Image.Resampling.LANCZOS)
    if "vertical-resample" in family:
        width, height = image.size
        reduced = image.resize((width, max(8, int(height * 0.79))), Image.Resampling.BICUBIC)
        return reduced.resize((width, height), Image.Resampling.LANCZOS)
    if family == "soft-scan-v1":
        return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.35))).enhance(0.88)
    if family == "low-contrast-sharpen-v1":
        return ImageEnhance.Sharpness(ImageEnhance.Contrast(image).enhance(0.82)).enhance(1.25)
    if family in {"jpeg-gray-v1", "jpeg-chroma-v1"}:
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=82 if family == "jpeg-gray-v1" else 76, subsampling=2)
        stream.seek(0)
        with Image.open(stream) as loaded:
            return loaded.convert("RGB")
    array = np.asarray(image, dtype=np.int16)
    noise = rng.integers(-3, 4, size=array.shape[:2], dtype=np.int16)[:, :, None]
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")


def render_case(partition: str, index: int) -> tuple[bytes, dict[str, Any]]:
    if partition not in COUNTS or not 0 <= index < COUNTS[partition]:
        raise ValueError("Recognition fixture identity is out of range")
    partition_offset = 0 if partition == "selection" else 100_000
    seed = SEED + partition_offset + index
    rng = random.Random(seed)
    text, role, family = _case_text(index + (0 if partition == "selection" else 37))
    font_relative = FONT_PATHS[(index * 5 + partition_offset) % len(FONT_PATHS)]
    font_path = REPO_ROOT / font_relative
    font_size = 17 + ((index * 11 + partition_offset) % 17)
    font = ImageFont.truetype(str(font_path), font_size)
    probe = Image.new("RGB", (640, 120), (252, 252, 250))
    probe_draw = ImageDraw.Draw(probe)
    bounds = probe_draw.textbbox((0, 0), text, font=font, stroke_width=0)
    text_width = max(1, bounds[2] - bounds[0])
    text_height = max(1, bounds[3] - bounds[1])
    left_pad = 3 + (index % 5)
    right_pad = 4 + ((index * 3) % 7)
    top_pad = 3 + ((index * 7) % 5)
    bottom_pad = 3 + ((index * 13) % 5)
    width = min(420, text_width + left_pad + right_pad)
    height = text_height + top_pad + bottom_pad
    background = (250 - (index % 3), 249 - (index % 2), 247 + (index % 4))
    foreground_value = 20 + ((index * 9) % 31)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.text(
        (left_pad - bounds[0], top_pad - bounds[1]),
        text,
        font=font,
        fill=(foreground_value, foreground_value, foreground_value),
    )
    if index % 9 in {4, 8}:
        angle = -0.8 if index % 2 else 0.8
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=background)
    degradations = SELECTION_DEGRADATIONS if partition == "selection" else PUBLIC_DEGRADATIONS
    degradation = degradations[(index * 3 + 1) % len(degradations)]
    image = _degrade(image, degradation, seed)
    if rng.random() < 0.25:
        image = ImageEnhance.Contrast(image).enhance(0.92 + rng.random() * 0.16)
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    source = stream.getvalue()
    case_id = f"{partition}-recognition-{index:04d}"
    record = {
        "case_id": case_id,
        "source_path": f"fixtures/{case_id}.png",
        "source_sha256": hash_bytes(source),
        "truth_text": text,
        "truth_role": role,
        "text_family": family,
        "renderer_family": f"noto-crop-recognition-{partition}-v1",
        "degradation_family": degradation,
        "font_path": font_relative.as_posix(),
        "font_sha256": hash_file(font_path),
        "font_size": font_size,
        "private_or_article_image": False,
        "chandler_image": False,
    }
    return source, record


def build_partition(partition: str) -> tuple[bytes, bytes]:
    cases: list[dict[str, Any]] = []
    archive_stream = BytesIO()
    with zipfile.ZipFile(archive_stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for index in range(COUNTS[partition]):
            source, record = render_case(partition, index)
            info = zipfile.ZipInfo(str(record["source_path"]), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source)
            cases.append(record)
    manifest = {
        "schema": SCHEMA,
        "partition": partition,
        "seed": SEED,
        "case_count": len(cases),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "cases": cases,
    }
    return canonical_json_bytes(manifest), archive_stream.getvalue()


def write_freeze(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"Recognition split output already exists: {output_root}")
    summary: dict[str, Any] = {"schema": "graphreader.ocr-official-recognition-freeze.v1", "partitions": {}}
    for partition in COUNTS:
        manifest_bytes, archive_bytes = build_partition(partition)
        partition_root = output_root / partition
        partition_root.mkdir(parents=True)
        manifest_path = partition_root / "private-manifest.json"
        archive_path = partition_root / "fixtures.zip"
        manifest_path.write_bytes(manifest_bytes)
        archive_path.write_bytes(archive_bytes)
        summary["partitions"][partition] = {
            "case_count": COUNTS[partition],
            "private_manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
            "private_manifest_sha256": hash_bytes(manifest_bytes),
            "fixture_archive_path": archive_path.relative_to(REPO_ROOT).as_posix(),
            "fixture_archive_sha256": hash_bytes(archive_bytes),
            "fixture_archive_bytes": len(archive_bytes),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ml/ocr/official_recognition_v1/artifacts/split-freeze"),
    )
    arguments = parser.parse_args()
    summary = write_freeze(REPO_ROOT / arguments.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
