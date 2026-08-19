# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.private_acceptance import run_private_acceptance as runner


REPO_ROOT = Path(__file__).resolve().parents[3]
AGGREGATE_PATH = REPO_ROOT / "docs/GOAL-22-PHASE-4-PRIVATE-ACCEPTANCE.json"


def test_private_root_inside_repository_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="PRIVATE_ROOT_INSIDE_REPOSITORY"):
        runner._outside_repository(REPO_ROOT)


def test_ci_detection_requires_an_explicit_local_run(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in runner.CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert runner._ci_enabled() is False
    monkeypatch.setenv("CI", "true")
    assert runner._ci_enabled() is True


def test_tracked_private_result_is_aggregate_only() -> None:
    result = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    serialized = AGGREGATE_PATH.read_text(encoding="utf-8")

    assert result["status"] == "fail"
    assert result["report_scope"] == "aggregate_metrics_only"
    assert result["execution"]["case_level_output"] is False
    assert result["execution"]["prediction_output"] is False
    assert result["execution"]["truth_row_output"] is False
    assert result["execution"]["pixel_output"] is False
    assert "C:\\" not in serialized
    assert "Chandler" not in serialized
    assert "Generalization" not in serialized


def test_real_contradiction_redirects_to_generator_repair_and_blocks_approval() -> None:
    result = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))

    assert result["stop_condition"]["triggered"] is False
    assert result["workflow_redirect"] == {
        "triggered": True,
        "phase": "4R",
        "reason": "real acceptance contradicts passing synthetic evidence",
        "private_training_or_selection_permitted": False,
    }
    assert all(stage["status"] == "fail" for stage in result["stages"])
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
