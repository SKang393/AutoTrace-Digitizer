# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic synthetic single-case design graph generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["generate_dataset"]
__version__ = "0.1.0"


def generate_dataset(
    preset: str,
    seed: int,
    output_directory: Path | None = None,
) -> Any:
    """Load the dataset writer lazily so schema tools stay independently usable."""

    from .dataset import generate_dataset as generate

    return generate(preset, seed, output_directory)
