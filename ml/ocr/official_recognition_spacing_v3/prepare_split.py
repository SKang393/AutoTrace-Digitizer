# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Generate fresh conservative-spacing selection and truth-hidden public crops."""

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
SCHEMA = "graphreader.ocr-conservative-spacing-split.v1"
FONT_PATHS = (
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)
NUMERIC_TEXTS = (
    "0", "1", "8", "10", "24", "33", "40", "75", "99", "100",
    "-1", "-8", "-0.7", "0.25", "3.5", "9.25", "42%", "75%", "100%",
)
COMPACT_WORDS = (
    ("Baseline", "phase_header"), ("Intervention", "phase_header"),
    ("Maintenance", "phase_header"), ("Treatment", "annotation"),
    ("Target", "legend_text"), ("Level", "legend_text"), ("Series", "legend_text"),
    ("Session", "axis_title"), ("Percent", "axis_title"), ("Frequency", "axis_title"),
    ("Smith", "participant"), ("Jordan", "participant"), ("Rivera", "participant"),
    ("Followup", "annotation"), ("Probe", "annotation"), ("Condition", "annotation"),
)
SPACED_WORDS = (
    ("Phase A", "phase_header"), ("Phase B", "phase_header"),
    ("Rule One", "annotation"), ("Probe Set", "annotation"),
    ("Low High", "annotation"), ("Phase Note", "annotation"),
    ("Follow Up", "annotation"), ("Data Series", "legend_text"),
)
AMBIGUITY_TEXTS = ("O o l I", "l I O o", "I l o O", "o O I l")
PARTIAL_SPACING_TEXTS = ("A B C", "A BC", "AB C", "X Y Z", "X YZ", "XY Z")
SELECTION_DEGRADATIONS = (
    "fractional-shear-v3", "anisotropic-resample-v3", "paper-gradient-v3", "soft-threshold-v3",
)
PUBLIC_DEGRADATIONS = (
    "fractional-shift-v3", "lanczos-cycle-v3", "gamma-band-v3", "column-fade-v3",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def hash_file(path: Path) -> str:
    return hash_bytes(path.read_bytes())


def _case_text(index: int) -> tuple[str, str, str]:
    slot = index % 16
    if slot < 5:
        return NUMERIC_TEXTS[(index * 7 + 3) % len(NUMERIC_TEXTS)], "numeric_text", "numeric"
    if slot < 10:
        text, role = COMPACT_WORDS[(index * 11 + 5) % len(COMPACT_WORDS)]
        return text, role, "compact_word"
    if slot < 13:
        text, role = SPACED_WORDS[(index * 5 + 1) % len(SPACED_WORDS)]
        return text, role, "spaced_word"
    if slot < 15:
        return AMBIGUITY_TEXTS[(index * 3 + 1) % len(AMBIGUITY_TEXTS)], "annotation", "ambiguity"
    return PARTIAL_SPACING_TEXTS[(index * 5 + 2) % len(PARTIAL_SPACING_TEXTS)], "annotation", "partial_spacing"


def _draw_controlled_text(
    text: str, font: ImageFont.FreeTypeFont, *, foreground: int, background: tuple[int, int, int],
    gap_pixels: int, left_pad: int, right_pad: int, top_pad: int, bottom_pad: int,
) -> Image.Image:
    tokens = text.split(" ")
    probe = Image.new("RGB", (1, 1), background)
    draw = ImageDraw.Draw(probe)
    bounds = [draw.textbbox((0, 0), token, font=font) for token in tokens]
    widths = [max(1, item[2] - item[0]) for item in bounds]
    heights = [max(1, item[3] - item[1]) for item in bounds]
    width = left_pad + right_pad + sum(widths) + gap_pixels * max(0, len(tokens) - 1)
    height = top_pad + bottom_pad + max(heights)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    x = float(left_pad)
    for token, token_bounds, token_width in zip(tokens, bounds, widths, strict=True):
        draw.text((x - token_bounds[0], top_pad - token_bounds[1]), token, font=font,
                  fill=(foreground, foreground, foreground))
        x += token_width + gap_pixels
    return image


def _degrade(image: Image.Image, family: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    width, height = image.size
    background = image.getpixel((0, 0))
    if family == "fractional-shear-v3":
        return image.transform(image.size, Image.Transform.AFFINE, (1.0, 0.018, -0.22, 0.0, 1.0, 0.0),
                               resample=Image.Resampling.BICUBIC, fillcolor=background)
    if family == "anisotropic-resample-v3":
        reduced = image.resize((max(8, width - 2), max(8, height - 1)), Image.Resampling.BOX)
        return reduced.resize((width, height), Image.Resampling.BICUBIC)
    if family == "paper-gradient-v3":
        array = np.asarray(image, dtype=np.int16)
        ramp = np.linspace(-2, 3, width, dtype=np.int16)[None, :, None]
        return Image.fromarray(np.clip(array + ramp, 0, 255).astype(np.uint8), "RGB")
    if family == "soft-threshold-v3":
        array = np.asarray(ImageEnhance.Contrast(image).enhance(0.94), dtype=np.int16)
        noise = rng.integers(-1, 2, size=array.shape[:2], dtype=np.int16)[:, :, None]
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")
    if family == "fractional-shift-v3":
        return image.transform(image.size, Image.Transform.AFFINE, (1.0, 0.0, -0.35, 0.0, 1.0, 0.30),
                               resample=Image.Resampling.BICUBIC, fillcolor=background)
    if family == "lanczos-cycle-v3":
        reduced = image.resize((max(8, width - 4), max(8, height - 2)), Image.Resampling.BOX)
        return reduced.resize((width, height), Image.Resampling.LANCZOS)
    if family == "gamma-band-v3":
        array = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(np.rint(255.0 * np.power(array, 1.025)).clip(0, 255).astype(np.uint8), "RGB")
    if family == "column-fade-v3":
        array = np.asarray(image, dtype=np.uint8).copy()
        column = (seed * 17) % max(1, width)
        array[:, max(0, column - 1):min(width, column + 1)] = np.maximum(
            array[:, max(0, column - 1):min(width, column + 1)], 182
        )
        return Image.fromarray(array, "RGB")
    raise ValueError(f"Unknown conservative spacing degradation: {family}")


def render_case(partition: str, index: int) -> tuple[bytes, dict[str, Any]]:
    if partition not in COUNTS or not 0 <= index < COUNTS[partition]:
        raise ValueError("Conservative spacing fixture identity is out of range")
    offset = 0 if partition == "selection" else 300_000
    seed = SEED + offset + index
    rng = random.Random(seed)
    text, role, family = _case_text(index + (0 if partition == "selection" else 79))
    font_relative = FONT_PATHS[(index * 5 + offset) % len(FONT_PATHS)]
    font_path = REPO_ROOT / font_relative
    font_size = 18 + ((index * 11 + offset) % 15)
    font = ImageFont.truetype(str(font_path), font_size)
    gap_ratio = 0.52 + 0.06 * ((index * 7 + offset) % 5)
    gap_pixels = 0 if " " not in text else max(7, int(round(font_size * gap_ratio)))
    left_pad = 4 + index % 5
    right_pad = 5 + (index * 3) % 7
    top_pad = 3 + (index * 5) % 4
    bottom_pad = 4 + (index * 7) % 5
    background = (250 - index % 3, 249 - index % 2, 247 + index % 4)
    foreground = 16 + (index * 9) % 32
    image = _draw_controlled_text(
        text, font, foreground=foreground, background=background, gap_pixels=gap_pixels,
        left_pad=left_pad, right_pad=right_pad, top_pad=top_pad, bottom_pad=bottom_pad,
    )
    if index % 12 in {4, 9}:
        image = image.rotate(-0.4 if index % 2 else 0.4, resample=Image.Resampling.BICUBIC,
                             expand=True, fillcolor=background)
    degradations = SELECTION_DEGRADATIONS if partition == "selection" else PUBLIC_DEGRADATIONS
    degradation = degradations[(index * 3 + 1) % len(degradations)]
    image = _degrade(image, degradation, seed)
    if rng.random() < 0.18:
        image = ImageEnhance.Sharpness(image.filter(ImageFilter.GaussianBlur(0.15))).enhance(1.08)
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    source = stream.getvalue()
    case_id = f"{partition}-conservative-spacing-{index:04d}"
    record = {
        "case_id": case_id,
        "source_path": f"fixtures/{case_id}.png",
        "source_sha256": hash_bytes(source),
        "truth_text": text,
        "truth_role": role,
        "text_family": family,
        "renderer_family": f"noto-controlled-gap-spacing-v3-{partition}",
        "degradation_family": degradation,
        "font_path": font_relative.as_posix(),
        "font_sha256": hash_file(font_path),
        "font_size": font_size,
        "inter_token_gap_pixels": gap_pixels,
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
        "seed": SEED if partition == "selection" else SEED + 300_000,
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
        raise RuntimeError(f"Conservative spacing split output already exists: {output_root}")
    summary: dict[str, Any] = {"schema": "graphreader.ocr-conservative-spacing-freeze.v1", "partitions": {}}
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
    parser.add_argument("--output-root", type=Path,
                        default=Path("ml/ocr/official_recognition_spacing_v3/artifacts/split-freeze"))
    args = parser.parse_args()
    print(json.dumps(write_freeze(REPO_ROOT / args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
