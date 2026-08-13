# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh line-relative procedural O/o/l/I glyph crops."""

from __future__ import annotations

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
PARTITION_OFFSET = {"train": 0, "validation": 120_000, "sealed_public": 450_000}


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
    font_relative = FONT_PATHS[(index * 7 + label * 5 + offset) % len(FONT_PATHS)]
    font_path = REPO_ROOT / font_relative
    font_size = 19 + (index * 13 + label * 3 + offset) % 16
    font = ImageFont.truetype(str(font_path), font_size)
    # The line box is derived from ascender/descender metrics once. Every glyph retains
    # its relative size and baseline within that shared box.
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    advance = max(1, int(round(probe.textlength(glyph, font=font))))
    canvas_width = max(advance + 16, font_size + 18)
    canvas_height = line_height + 16
    canvas = Image.new("L", (canvas_width, canvas_height), 255)
    baseline_y = 8 + ascent
    x = (canvas_width - advance) / 2 + rng.uniform(-1.5, 1.5)
    foreground = 10 + (index * 11 + label * 19) % 48
    ImageDraw.Draw(canvas).text((x, baseline_y), glyph, font=font, anchor="ls", fill=foreground)
    canvas = canvas.transform(canvas.size, Image.Transform.AFFINE,
                              (1.0, rng.uniform(-0.035, 0.035), rng.uniform(-0.35, 0.35),
                               0.0, 1.0, rng.uniform(-0.30, 0.30)),
                              resample=Image.Resampling.BICUBIC, fillcolor=255)
    if index % 4 == 1:
        canvas = ImageEnhance.Contrast(canvas).enhance(0.90 + 0.10 * rng.random())
    elif index % 4 == 2:
        canvas = canvas.filter(ImageFilter.GaussianBlur(0.12 + 0.20 * rng.random()))
    elif index % 4 == 3:
        canvas = canvas.resize((max(8, canvas.width - 2), max(8, canvas.height - 1)), Image.Resampling.BOX).resize(canvas.size, Image.Resampling.BICUBIC)
    array = np.asarray(canvas, dtype=np.int16)
    array = np.clip(array + np_rng.integers(-2, 3, size=array.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    canvas = Image.fromarray(array, "L")
    # Normalize the complete line box to 28 pixels high, preserving glyph scale inside it.
    scale = 28.0 / canvas.height
    resized = canvas.resize((max(1, round(canvas.width * scale)), 28), Image.Resampling.BILINEAR)
    normalized = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    if resized.width > IMAGE_SIZE - 2:
        resized = resized.resize((IMAGE_SIZE - 2, 28), Image.Resampling.BILINEAR)
    normalized.paste(resized, ((IMAGE_SIZE - resized.width) // 2, 2))
    tensor = (1.0 - np.asarray(normalized, dtype=np.float32) / 255.0)[None, :, :].astype(np.float32)
    stream = BytesIO()
    normalized.save(stream, format="PNG", optimize=False, compress_level=9)
    source = stream.getvalue()
    sample_id = f"{partition}-context-{label}-{index:04d}"
    return source, tensor, {
        "sample_id": sample_id, "source_path": f"fixtures/{sample_id}.png", "source_sha256": hash_bytes(source),
        "glyph": glyph, "label": label, "font_path": font_relative.as_posix(),
        "font_sha256": hash_bytes(font_path.read_bytes()), "font_size": font_size,
        "line_ascent": ascent, "line_descent": descent,
        "renderer_family": f"noto-complete-line-box-v2-{partition}",
        "degradation_family": f"line-context-affine-v2-{index % 4}",
        "private_or_article_image": False, "chandler_image": False,
    }


def build_partition(partition: str) -> tuple[bytes, bytes, np.ndarray, np.ndarray]:
    if partition not in COUNTS_PER_CLASS:
        raise ValueError(f"Unknown context glyph partition: {partition}")
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
                records.append(record); values.append(tensor); labels.append(label)
    manifest = {"schema": "graphreader.ocr-ambiguity-context-split.v1", "partition": partition,
                "seed": SEED + PARTITION_OFFSET[partition], "class_order": list(GLYPHS),
                "count_per_class": COUNTS_PER_CLASS[partition], "sample_count": len(records),
                "coordinate_space": "complete-line-box-normalized glyph-centered crop",
                "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
                "generalization_label_included": False, "samples": records}
    return canonical_json_bytes(manifest), stream.getvalue(), np.stack(values), np.asarray(labels, dtype=np.int64)


def split_fingerprint(partition: str) -> str:
    manifest, archive, values, labels = build_partition(partition)
    digest = sha256(); digest.update(manifest); digest.update(archive)
    digest.update(np.ascontiguousarray(values).tobytes()); digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


def write_freeze(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"Context glyph split root exists: {root}")
    summary: dict[str, Any] = {"schema": "graphreader.ocr-ambiguity-context-freeze.v1", "partitions": {}}
    for partition in COUNTS_PER_CLASS:
        manifest, archive, values, labels = build_partition(partition)
        target = root / partition; target.mkdir(parents=True)
        manifest_path = target / "private-manifest.json"; archive_path = target / "fixtures.zip"
        manifest_path.write_bytes(manifest); archive_path.write_bytes(archive)
        summary["partitions"][partition] = {
            "sample_count": len(labels), "count_per_class": COUNTS_PER_CLASS[partition],
            "private_manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(), "private_manifest_sha256": hash_bytes(manifest),
            "fixture_archive_path": archive_path.relative_to(REPO_ROOT).as_posix(), "fixture_archive_sha256": hash_bytes(archive),
            "fixture_archive_bytes": len(archive),
            "tensor_label_stream_sha256": hash_bytes(np.ascontiguousarray(values).tobytes() + np.ascontiguousarray(labels).tobytes()),
            "split_fingerprint": split_fingerprint(partition),
        }
    return summary


__all__ = ["build_partition", "canonical_json_bytes", "hash_bytes", "split_fingerprint", "write_freeze", "REPO_ROOT"]

