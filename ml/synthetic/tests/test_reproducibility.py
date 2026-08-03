# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Byte reproducibility tests for the fixed renderer environment."""

from __future__ import annotations

from pathlib import Path

from ml.synthetic.dataset import generate_dataset


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    }


def test_same_seed_is_byte_identical(smoke_root: Path, tmp_path: Path) -> None:
    second = generate_dataset("smoke", 393, tmp_path / "second")
    assert _files(smoke_root) == _files(second.output_directory)


def test_different_seed_changes_images(smoke_root: Path, tmp_path: Path) -> None:
    second = generate_dataset("smoke", 394, tmp_path / "second")
    first_images = {
        path.name: path.read_bytes() for path in smoke_root.glob("images/*.png")
    }
    second_images = {
        path.name: path.read_bytes() for path in second.output_directory.glob("images/*.png")
    }
    assert set(first_images.values()) != set(second_images.values())
