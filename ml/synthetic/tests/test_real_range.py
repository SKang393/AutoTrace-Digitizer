# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ml.synthetic.dataset import generate_dataset
from ml.synthetic.renderer import production_resize_dimensions


def test_production_resize_profile_matches_960_128_contract() -> None:
    profile = production_resize_dimensions(863, 395)

    assert profile["resized_width"] == 960
    assert profile["resized_height"] == 439
    assert profile["aligned_width"] == 1024
    assert profile["aligned_height"] == 512
    assert profile["tensor_scale_x"] == 1024 / 863
    assert profile["tensor_scale_y"] == 512 / 395


def test_real_range_preset_passes_aggregate_distribution_gate(
    tmp_path: Path,
) -> None:
    result = generate_dataset("real_range", 393, tmp_path / "real-range")
    report_path = result.output_directory / "distribution-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = report_path.read_text(encoding="utf-8")

    assert result.case_count == 6
    assert all(report["gates"].values())
    assert "863x395" in report["source_dimensions"]
    assert report["source_pixel_format"] == {
        "bit_depths": [8],
        "modes": ["RGB"],
        "png_color_types": [2],
    }
    assert report["png_encoding"] == {
        "compression_methods": [0],
        "filter_methods": [0],
        "interlace_methods": [0],
    }
    assert report["jpeg_roundtrip_qualities"] == [55, 70, 85]
    assert min(report["text_region_counts"]) <= 38 <= max(
        report["text_region_counts"]
    )
    assert "Chandler" not in serialized
    assert "Generalization" not in serialized

    sizes: set[tuple[int, int]] = set()
    for path in (result.output_directory / "images").glob("*.png"):
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert payload[24:29] == bytes((8, 2, 0, 0, 0))
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            sizes.add(image.size)
    assert (863, 395) in sizes


def test_real_range_preset_is_byte_deterministic(tmp_path: Path) -> None:
    first = generate_dataset("real_range", 393, tmp_path / "first")
    second = generate_dataset("real_range", 393, tmp_path / "second")

    first_manifest = json.loads(
        (first.output_directory / "seed-manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (second.output_directory / "seed-manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest == second_manifest
