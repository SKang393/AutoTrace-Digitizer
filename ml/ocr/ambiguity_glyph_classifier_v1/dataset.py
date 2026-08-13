# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh deterministic procedural O/o/l/I glyph crops."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from typing import Any
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .protocol import COUNTS_PER_CLASS, GLYPHS, IMAGE_SIZE, SEED


REPO_ROOT = Path(__file__).resolve().parents[3]
FONT_PATHS = (
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)
PARTITION_OFFSET = {"train": 0, "validation": 100_000, "sealed_public": 400_000}


@dataclass(frozen=True)
class GlyphSample:
    sample_id: str
    label: int
    glyph: str
    source_sha256: str
    tensor: np.ndarray


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _render(partition: str, label: int, index: int) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    offset = PARTITION_OFFSET[partition]
    seed = SEED + offset + label * 10_000 + index
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    glyph = GLYPHS[label]
    font_relative = FONT_PATHS[(index * 5 + label * 7 + offset) % len(FONT_PATHS)]
    font_path = REPO_ROOT / font_relative
    font_size = 19 + (index * 11 + label * 3 + offset) % 16
    font = ImageFont.truetype(str(font_path), font_size)
    canvas = Image.new("L", (52, 52), 255)
    draw = ImageDraw.Draw(canvas)
    bounds = draw.textbbox((0, 0), glyph, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = (52 - width) / 2 - bounds[0] + rng.uniform(-2.0, 2.0)
    y = (52 - height) / 2 - bounds[1] + rng.uniform(-2.0, 2.0)
    foreground = 10 + (index * 13 + label * 17) % 50
    draw.text((x, y), glyph, font=font, fill=foreground)
    shear = rng.uniform(-0.045, 0.045)
    shift_x = rng.uniform(-0.45, 0.45)
    canvas = canvas.transform(canvas.size, Image.Transform.AFFINE, (1.0, shear, shift_x, 0.0, 1.0, rng.uniform(-0.35, 0.35)),
                              resample=Image.Resampling.BICUBIC, fillcolor=255)
    if index % 5 == 1:
        canvas = canvas.resize((49, 51), Image.Resampling.BOX).resize((52, 52), Image.Resampling.BICUBIC)
    elif index % 5 == 2:
        canvas = ImageEnhance.Contrast(canvas).enhance(0.88 + 0.12 * rng.random())
    elif index % 5 == 3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(0.15 + 0.25 * rng.random()))
    array = np.asarray(canvas, dtype=np.int16)
    noise = np_rng.integers(-2, 3, size=array.shape, dtype=np.int16)
    canvas = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "L")
    ink = 1.0 - np.asarray(canvas, dtype=np.float32) / 255.0
    coordinates = np.argwhere(ink > 0.08)
    if len(coordinates) == 0:
        raise RuntimeError("Rendered ambiguity glyph contains no ink")
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    crop = canvas.crop((max(0, int(left) - 2), max(0, int(top) - 2), min(52, int(right) + 3), min(52, int(bottom) + 3)))
    scale = min(20 / max(1, crop.width), 20 / max(1, crop.height))
    resized = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.BILINEAR)
    normalized = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    normalized.paste(resized, ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2))
    tensor = (1.0 - np.asarray(normalized, dtype=np.float32) / 255.0)[None, :, :].astype(np.float32)
    stream = BytesIO()
    normalized.save(stream, format="PNG", optimize=False, compress_level=9)
    source = stream.getvalue()
    sample_id = f"{partition}-{label}-{index:04d}"
    record = {
        "sample_id": sample_id, "source_path": f"fixtures/{sample_id}.png",
        "source_sha256": hash_bytes(source), "glyph": glyph, "label": label,
        "font_path": font_relative.as_posix(), "font_sha256": hash_bytes(font_path.read_bytes()),
        "font_size": font_size, "renderer_family": f"noto-glyph-crop-v1-{partition}",
        "degradation_family": f"affine-resample-noise-v1-{index % 5}",
        "private_or_article_image": False, "chandler_image": False,
    }
    return source, tensor, record


def build_partition(partition: str) -> tuple[bytes, bytes, np.ndarray, np.ndarray]:
    if partition not in COUNTS_PER_CLASS:
        raise ValueError(f"Unknown ambiguity glyph partition: {partition}")
    records: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    labels: list[int] = []
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for label, _glyph in enumerate(GLYPHS):
            for index in range(COUNTS_PER_CLASS[partition]):
                source, tensor, record = _render(partition, label, index)
                info = zipfile.ZipInfo(str(record["source_path"]), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source)
                records.append(record)
                values.append(tensor)
                labels.append(label)
    manifest = {
        "schema": "graphreader.ocr-ambiguity-glyph-split.v1", "partition": partition,
        "seed": SEED + PARTITION_OFFSET[partition], "class_order": list(GLYPHS),
        "count_per_class": COUNTS_PER_CLASS[partition], "sample_count": len(records),
        "synthetic_only": True, "private_or_article_images": False,
        "chandler_included": False, "generalization_label_included": False, "samples": records,
    }
    return canonical_json_bytes(manifest), stream.getvalue(), np.stack(values), np.asarray(labels, dtype=np.int64)


def split_fingerprint(partition: str) -> str:
    manifest, archive, values, labels = build_partition(partition)
    digest = sha256()
    digest.update(manifest)
    digest.update(archive)
    digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


def write_freeze(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"Ambiguity glyph split root already exists: {root}")
    summary: dict[str, Any] = {"schema": "graphreader.ocr-ambiguity-glyph-freeze.v1", "partitions": {}}
    for partition in COUNTS_PER_CLASS:
        manifest, archive, values, labels = build_partition(partition)
        partition_root = root / partition
        partition_root.mkdir(parents=True)
        manifest_path = partition_root / "private-manifest.json"
        archive_path = partition_root / "fixtures.zip"
        manifest_path.write_bytes(manifest)
        archive_path.write_bytes(archive)
        summary["partitions"][partition] = {
            "sample_count": len(labels), "count_per_class": COUNTS_PER_CLASS[partition],
            "private_manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
            "private_manifest_sha256": hash_bytes(manifest),
            "fixture_archive_path": archive_path.relative_to(REPO_ROOT).as_posix(),
            "fixture_archive_sha256": hash_bytes(archive), "fixture_archive_bytes": len(archive),
            "tensor_label_stream_sha256": hash_bytes(np.ascontiguousarray(values).tobytes() + np.ascontiguousarray(labels).tobytes()),
            "split_fingerprint": split_fingerprint(partition),
        }
    return summary


__all__ = ["GlyphSample", "build_partition", "canonical_json_bytes", "hash_bytes", "split_fingerprint", "write_freeze"]
