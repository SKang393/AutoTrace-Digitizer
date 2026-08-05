# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from ml.ocr.production_gate import evaluate_partition, maximum_absolute_error


def test_production_gate_derives_metrics_from_records() -> None:
    metrics = evaluate_partition(
        [
            ("10", "10", "y_tick", "y_tick"),
            ("-2.5", "-2.0", "annotation", "annotation"),
        ]
    )

    assert metrics.exact_match == 0.5
    assert metrics.character_error_rate == 1 / 6
    assert metrics.role_accuracy == 1.0
    assert maximum_absolute_error([(0.0, 0.00005), (1.0, 1.0)]) == 0.00005
