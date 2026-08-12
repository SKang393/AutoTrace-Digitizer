# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Generate fresh spacing-repair selection and truth-hidden public crops."""

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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .protocol import COUNTS, SEED


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "graphreader.ocr-official-recognition-spacing-split.v1"
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
    ("Chandler", "participant"), ("Smith", "participant"), ("Jordan", "participant"),
    ("Rivera", "participant"), ("Baseline", "phase_header"),
    ("Intervention", "phase_header"), ("Maintenance", "phase_header"),
    ("Generalization", "phase_header"), ("Session", "axis_title"),
    ("Percent", "axis_title"), ("Frequency", "axis_title"),
    ("Follow-up", "annotation"), ("Treatment", "annotation"), ("Probe", "annotation"),
    ("Phase A", "phase_header"), ("Phase B", "phase_header"),
)
SELECTION_DEGRADATIONS = ("mild-shear-v2", "box-resample-v2", "uneven-paper-v2", "contrast-dither-v2")
PUBLIC_DEGRADATIONS = ("subpixel-offset-v2", "area-resample-v2", "gamma-raster-v2", "row-fade-v2")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def hash_file(path: Path) -> str:
    return hash_bytes(path.read_bytes())


def _case_text(index: int) -> tuple[str, str, str]:
    if index % 8 == 7:
        return "O o l I", "annotation", "ambiguity"
    if index % 4 != 3:
        text = NUMERIC_TEXTS[(index * 11 + 7) % len(NUMERIC_TEXTS)]
        return text, "numeric_text", "numeric"
    text, role = WORD_CASES[(index * 7 + 3) % len(WORD_CASES)]
    return text, role, "word"


def _degrade(image: Image.Image, family: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    width, height = image.size
    if family == "mild-shear-v2":
        return image.transform(
            image.size, Image.Transform.AFFINE, (1.0, 0.025, -0.3, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BICUBIC, fillcolor=image.getpixel((0, 0)),
        )
    if family == "box-resample-v2":
        reduced = image.resize((max(8, width - 3), max(8, height - 1)), Image.Resampling.BOX)
        return reduced.resize((width, height), Image.Resampling.BICUBIC)
    if family == "uneven-paper-v2":
        array = np.asarray(image, dtype=np.int16)
        ramp = np.linspace(-3, 4, width, dtype=np.int16)[None, :, None]
        return Image.fromarray(np.clip(array + ramp, 0, 255).astype(np.uint8), "RGB")
    if family == "contrast-dither-v2":
        array = np.asarray(ImageEnhance.Contrast(image).enhance(0.91), dtype=np.int16)
        dither = rng.integers(-2, 3, size=array.shape[:2], dtype=np.int16)[:, :, None]
        return Image.fromarray(np.clip(array + dither, 0, 255).astype(np.uint8), "RGB")
    if family == "subpixel-offset-v2":
        return image.transform(
            image.size, Image.Transform.AFFINE, (1.0, 0.0, 0.45, 0.0, 1.0, -0.25),
            resample=Image.Resampling.BICUBIC, fillcolor=image.getpixel((0, 0)),
        )
    if family == "area-resample-v2":
        reduced = image.resize((max(8, width - 5), max(8, height - 2)), Image.Resampling.BOX)
        return reduced.resize((width, height), Image.Resampling.LANCZOS)
    if family == "gamma-raster-v2":
        array = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(np.rint(255.0 * np.power(array, 1.035)).clip(0, 255).astype(np.uint8), "RGB")
    if family == "row-fade-v2":
        array = np.asarray(image, dtype=np.uint8).copy()
        row = (seed * 13) % max(1, height)
        array[max(0, row - 1):min(height, row + 1)] = np.maximum(
            array[max(0, row - 1):min(height, row + 1)], 176
        )
        return Image.fromarray(array, "RGB")
    raise ValueError(f"Unknown spacing degradation family: {family}")


def render_case(partition: str, index: int) -> tuple[bytes, dict[str, Any]]:
    if partition not in COUNTS or not 0 <= index < COUNTS[partition]:
        raise ValueError("Spacing fixture identity is out of range")
    offset = 0 if partition == "selection" else 200_000
    seed = SEED + offset + index
    rng = random.Random(seed)
    text, role, family = _case_text(index + (0 if partition == "selection" else 53))
    font_relative = FONT_PATHS[(index * 7 + offset) % len(FONT_PATHS)]
    font_path = REPO_ROOT / font_relative
    font_size = 17 + ((index * 13 + offset) % 16)
    font = ImageFont.truetype(str(font_path), font_size)
    probe = Image.new("RGB", (640, 120), (252, 251, 249))
    bounds = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    text_width = max(1, bounds[2] - bounds[0])
    text_height = max(1, bounds[3] - bounds[1])
    left_pad = 4 + index % 6
    right_pad = 5 + (index * 5) % 8
    top_pad = 3 + (index * 3) % 5
    bottom_pad = 4 + (index * 11) % 5
    background = (249 - index % 3, 248 - index % 2, 246 + index % 5)
    foreground = 18 + (index * 7) % 34
    image = Image.new("RGB", (min(430, text_width + left_pad + right_pad), text_height + top_pad + bottom_pad), background)
    ImageDraw.Draw(image).text(
        (left_pad - bounds[0], top_pad - bounds[1]), text, font=font,
        fill=(foreground, foreground, foreground),
    )
    if index % 10 in {3, 8}:
        image = image.rotate(
            -0.55 if index % 2 else 0.55, resample=Image.Resampling.BICUBIC,
            expand=True, fillcolor=background,
        )
    degradations = SELECTION_DEGRADATIONS if partition == "selection" else PUBLIC_DEGRADATIONS
    degradation = degradations[(index * 5 + 2) % len(degradations)]
    image = _degrade(image, degradation, seed)
    if rng.random() < 0.2:
        image = ImageEnhance.Sharpness(image.filter(ImageFilter.GaussianBlur(0.18))).enhance(1.12)
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    source = stream.getvalue()
    case_id = f"{partition}-spacing-{index:04d}"
    record = {
        "case_id": case_id,
        "source_path": f"fixtures/{case_id}.png",
        "source_sha256": hash_bytes(source),
        "truth_text": text,
        "truth_role": role,
        "text_family": family,
        "renderer_family": f"noto-variable-pad-spacing-v2-{partition}",
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
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
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
        "seed": SEED if partition == "selection" else SEED + 200_000,
        "case_count": len(cases),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "cases": cases,
    }
    return canonical_json_bytes(manifest), stream.getvalue()


def write_freeze(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"Spacing split output already exists: {output_root}")
    summary: dict[str, Any] = {"schema": "graphreader.ocr-official-recognition-spacing-freeze.v1", "partitions": {}}
    for partition in COUNTS:
        manifest, archive = build_partition(partition)
        root = output_root / partition
        root.mkdir(parents=True)
        manifest_path = root / "private-manifest.json"
        archive_path = root / "fixtures.zip"
        manifest_path.write_bytes(manifest)
        archive_path.write_bytes(archive)
        summary["partitions"][partition] = {
            "case_count": COUNTS[partition],
            "private_manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
            "private_manifest_sha256": hash_bytes(manifest),
            "fixture_archive_path": archive_path.relative_to(REPO_ROOT).as_posix(),
            "fixture_archive_sha256": hash_bytes(archive),
            "fixture_archive_bytes": len(archive),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("ml/ocr/official_recognition_spacing_v2/artifacts/split-freeze"))
    args = parser.parse_args()
    print(json.dumps(write_freeze(REPO_ROOT / args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

