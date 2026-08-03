# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

import ml.graphsr.benchmark as benchmark_module
from ml.graphsr.benchmark import REQUIRED_CANDIDATES, run_benchmark, select_default
from ml.graphsr.metrics import METRIC_NAMES


def _measured_metrics() -> dict[str, float | int]:
    return {
        "numeric_ocr_exact_match": 0.80,
        "numeric_ocr_proxy_exact_match": 0.80,
        "marker_center_f1": 0.80,
        "shape_fill_classification_f1": 0.80,
        "marker_center_mean_error_pixels": 0.50,
        "axis_thin_line_recall": 0.90,
        "axis_line_localization_error_pixels": 0.25,
        "open_marker_preservation_rate": 0.90,
        "hallucinated_structure_rate": 0.02,
        "runtime_ms_mean": 10.0,
        "peak_memory_bytes": 1_000,
    }


def _thresholds() -> dict[str, float]:
    return {
        "numeric_ocr_exact_match_min": 0.75,
        "marker_center_f1_min": 0.75,
        "shape_fill_classification_f1_min": 0.75,
        "axis_thin_line_recall_min": 0.80,
        "open_marker_preservation_rate_min": 0.80,
        "marker_center_mean_error_pixels_max": 1.0,
        "axis_line_localization_error_pixels_max": 1.0,
        "hallucinated_structure_rate_max": 0.05,
        "runtime_ms_mean_max": 1_000.0,
        "peak_memory_bytes_max": 1_000_000_000.0,
    }


def _measured_rows() -> list[dict[str, object]]:
    return [
        {"model_id": model_id, "status": "measured", "metrics": _measured_metrics()}
        for model_id in REQUIRED_CANDIDATES
    ]


def test_evidence_only_benchmark_compares_all_candidates_without_fabrication(
    repository_root: Path,
    no_network: None,
) -> None:
    manifest = (
        repository_root
        / "models"
        / "manifest"
        / "graphsr"
        / "graphsr-x2-candidate-0.1.0.json"
    )

    report = run_benchmark(manifest)

    assert report["benchmark_schema_version"] == 1
    assert report["status"] == "incomplete"
    assert report["network_access"] == "disabled_by_contract"
    assert "input decode" in report["measurement_contract"]["runtime"]
    assert "candidate child processes" in report["measurement_contract"]["memory"]
    assert tuple(report["metric_names"]) == METRIC_NAMES
    assert tuple(row["model_id"] for row in report["candidates"]) == REQUIRED_CANDIDATES
    assert len(report["candidates"]) == 5
    for row in report["candidates"]:
        assert row["status"] in {"blocked", "unmeasured"}
        assert isinstance(row["reason"], str) and row["reason"].strip()
        assert tuple(row["metrics"]) == METRIC_NAMES
        assert all(value is None for value in row["metrics"].values())
        assert row["metrics"]["numeric_ocr_exact_match"] is None
        assert row["metrics"]["shape_fill_classification_f1"] is None
    assert report["selection"]["status"] == "none"
    assert report["selection"]["model_id"] is None
    assert report["exit_code"] == 0


def test_proxy_only_improvement_cannot_select_a_default() -> None:
    rows = _measured_rows()
    graphsr = rows[REQUIRED_CANDIDATES.index("GraphSR-x2")]["metrics"]
    graphsr["numeric_ocr_proxy_exact_match"] = 1.0

    selection = select_default(rows, _thresholds())

    assert selection["status"] == "none"
    assert selection["model_id"] is None
    checks = selection["gate_checks"]["GraphSR-x2"]
    assert checks["numeric_ocr_strict_improvement"] is False
    assert checks["marker_f1_strict_improvement"] is False


def test_selection_requires_complete_actual_downstream_improvement() -> None:
    rows = _measured_rows()
    graphsr = rows[REQUIRED_CANDIDATES.index("GraphSR-x2")]["metrics"]
    graphsr["numeric_ocr_exact_match"] = 0.85
    graphsr["marker_center_f1"] = 0.85

    selected = select_default(rows, _thresholds())
    graphsr["shape_fill_classification_f1"] = None
    unavailable = select_default(rows, _thresholds())

    assert selected["status"] == "selected"
    assert selected["model_id"] == "GraphSR-x2"
    assert unavailable["status"] == "none"
    assert unavailable["model_id"] is None


def test_fixed_benchmark_retains_partial_structural_evidence(
    tmp_path: Path,
    no_network: None,
) -> None:
    hr = np.full((32, 32, 3), 255, dtype=np.uint8)
    hr[24, 4:28] = 0
    hr[4:25, 4] = 0
    yy, xx = np.ogrid[:32, :32]
    ring = ((xx - 16) ** 2 + (yy - 14) ** 2 >= 9) & (
        (xx - 16) ** 2 + (yy - 14) ** 2 <= 16
    )
    hr[ring] = 0
    lr = np.asarray(
        Image.fromarray(hr).resize((16, 16), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    input_path = tmp_path / "input.png"
    truth_path = tmp_path / "truth.png"
    Image.fromarray(lr).save(input_path)
    Image.fromarray(hr).save(truth_path)
    manifest = {
        "benchmark_manifest_version": 1,
        "dataset": {
            "dataset_id": "partial-evidence-fixture",
            "cases": [
                {
                    "case_id": "case-1",
                    "input_path": "input.png",
                    "ground_truth_path": "truth.png",
                    "annotations": {
                        "ocr_regions": [{"box": [8, 26, 8, 4]}],
                        "marker_centers": [{"center": [16, 14], "radius": 4}],
                        "axis_lines": [[4, 24, 27, 24], [4, 4, 4, 24]],
                        "open_markers": [{"center": [16, 14], "radius": 4}],
                    },
                }
            ],
        },
        "candidates": [
            {"model_id": model_id, "status": "blocked", "reason": "fixture has no runtime"}
            for model_id in REQUIRED_CANDIDATES[:-1]
        ],
        "selection": {"thresholds": {}},
    }
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_benchmark(manifest_path)

    baseline = report["candidates"][-1]
    assert baseline["status"] == "unmeasured"
    assert baseline["metrics"]["numeric_ocr_exact_match"] is None
    assert baseline["metrics"]["shape_fill_classification_f1"] is None
    assert baseline["metrics"]["marker_center_f1"] is None
    assert baseline["metrics"]["marker_center_mean_error_pixels"] is not None
    assert baseline["metrics"]["axis_thin_line_recall"] is not None
    assert baseline["metrics"]["open_marker_preservation_rate"] is not None
    assert baseline["metrics"]["runtime_ms_mean"] is not None
    assert baseline["evidence_sha256"] is not None


def test_command_runtime_uses_declared_memory_sampling_cadence(
    tmp_path: Path,
    monkeypatch,
    no_network: None,
) -> None:
    source = np.full((8, 8, 3), 255, dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(source).save(input_path)
    script = tmp_path / "resize.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys, time\n"
        "from PIL import Image\n"
        "time.sleep(0.03)\n"
        "with Image.open(Path(sys.argv[1])) as image:\n"
        "    image.resize((image.width * 2, image.height * 2)).save(Path(sys.argv[2]))\n",
        encoding="utf-8",
    )
    observed_sleeps: list[float] = []
    original_sleep = time.sleep

    def record_sleep(seconds: float) -> None:
        observed_sleeps.append(seconds)
        original_sleep(seconds)

    monkeypatch.setattr(benchmark_module.time, "sleep", record_sleep)
    runtime = {
        "offline_confirmed": True,
        "artifacts": [
            {
                "path": str(script),
                "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            }
        ],
        "argv": [sys.executable, str(script), "{input}", "{output}"],
        "timeout_seconds": 5,
    }

    output, elapsed, peak = benchmark_module._command_x2(
        runtime,
        tmp_path,
        input_path,
        output_path,
    )

    assert output.shape == (16, 16, 3)
    assert elapsed > 0.0
    assert peak is not None and peak > 0
    assert observed_sleeps
    assert set(observed_sleeps) == {benchmark_module.MEMORY_SAMPLE_INTERVAL_SECONDS}
