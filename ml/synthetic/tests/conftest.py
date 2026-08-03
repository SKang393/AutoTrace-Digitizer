# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Shared fixed-output fixtures for Session 06."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.synthetic.dataset import DatasetResult, generate_dataset


@pytest.fixture(scope="session")
def smoke_result(tmp_path_factory: pytest.TempPathFactory) -> DatasetResult:
    destination = tmp_path_factory.mktemp("synthetic-smoke") / "dataset"
    return generate_dataset("smoke", 393, destination)


@pytest.fixture(scope="session")
def smoke_root(smoke_result: DatasetResult) -> Path:
    return smoke_result.output_directory


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
