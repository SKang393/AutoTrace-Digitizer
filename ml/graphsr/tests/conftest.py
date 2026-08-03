# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def chart_fixture() -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    """Return a tiny public-domain synthetic chart with fragile structures."""

    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)

    # One-pixel axes and series strokes intentionally exercise topology loss.
    draw.line((12, 8, 12, 82), fill="black", width=1)
    draw.line((12, 82, 88, 82), fill="black", width=1)
    centers = ((28.0, 66.0), (50.0, 49.0), (74.0, 29.0))
    draw.line(centers, fill="black", width=1)

    # Open markers have a white center surrounded by a one-pixel outline.
    for x, y in centers:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="black", width=1)

    return np.asarray(image, dtype=np.uint8), centers


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail fast if a training or benchmark smoke test tries to use the network."""

    import socket

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("GraphSR tests must remain local and offline")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    yield


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def artifact_root(repository_root: Path) -> Iterator[Path]:
    """Create a unique test run inside the project's explicit ignored cache."""

    cache_root = (repository_root / "ml" / "graphsr" / "cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    run_root = (cache_root / f"pytest-{uuid.uuid4().hex}").resolve()
    assert cache_root in run_root.parents
    run_root.mkdir()
    try:
        yield run_root
    finally:
        assert cache_root in run_root.parents
        shutil.rmtree(run_root, ignore_errors=False)
