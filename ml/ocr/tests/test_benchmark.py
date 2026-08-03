# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import pytest

from ml.ocr.benchmark import ACCEPTANCE_EXACT_MATCH, run_benchmark


def test_held_out_benchmark_meets_declared_threshold_and_beats_baseline() -> None:
    report = run_benchmark(seed=20260802)

    assert report.acceptance_passed
    assert report.trained_validation.exact_match >= ACCEPTANCE_EXACT_MATCH
    assert report.trained_test.exact_match >= ACCEPTANCE_EXACT_MATCH
    assert report.trained_test.exact_match > report.baseline_test.exact_match
    assert report.trained_test.character_error_rate < report.baseline_test.character_error_rate


def test_benchmark_is_metric_deterministic() -> None:
    first = run_benchmark(seed=91)
    second = run_benchmark(seed=91)

    assert first.training == second.training
    assert first.baseline_test == second.baseline_test
    assert first.trained_validation == second.trained_validation
    assert first.trained_test == second.trained_test


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_benchmark_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        run_benchmark(threshold=threshold)
