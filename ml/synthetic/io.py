# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic serialization helpers for generated datasets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with stable key order, whitespace, and line ending."""

    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> bytes:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def canonical_csv_bytes(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> bytes:
    payload = canonical_csv_bytes(fieldnames, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def write_png(path: Path, image: Image.Image) -> bytes:
    payload = png_bytes(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def relative_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes())
        for path in sorted(paths, key=lambda item: item.as_posix())
    }
