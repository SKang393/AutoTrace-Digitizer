# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import pytest

from ml.ocr.metrics import edit_distance, evaluate_predictions


def test_edit_distance_handles_substitution_insertion_and_deletion() -> None:
    assert edit_distance("100", "1O0") == 1
    assert edit_distance("10", "100") == 1
    assert edit_distance("100", "10") == 1


def test_metrics_report_exact_match_and_cer() -> None:
    metrics = evaluate_predictions((("10", "10"), ("25", "2")))

    assert metrics.sample_count == 2
    assert metrics.exact_count == 1
    assert metrics.exact_match == 0.5
    assert metrics.character_errors == 1
    assert metrics.reference_characters == 4
    assert metrics.character_error_rate == 0.25


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one"):
        evaluate_predictions(())
