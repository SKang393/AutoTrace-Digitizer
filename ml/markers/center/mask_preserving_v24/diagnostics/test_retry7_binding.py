# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics import diagnose_retry
from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    RETRY7_DEV_SPLIT_SHA256,
    RETRY7_GENERATOR_AUDIT_SHA256,
    RETRY7_ONNX_SHA256,
    RETRY7_SAMPLER_SHA256,
    summarize,
    summarize_morphology,
)


def test_retry7_hash_bindings_are_explicit() -> None:
    assert RETRY7_ONNX_SHA256 == "7932b008a9c4372c832215f2f8732c59c59012a25aa4ad2d12cfeaed404bbe3c"
    assert RETRY7_GENERATOR_AUDIT_SHA256 == "9562044526bce45b48254472346ccc1640c6e254915b9e744525154144121748"
    assert RETRY7_DEV_SPLIT_SHA256 == "f453e50e228f54e15bafba83b1a5dda422c435a555e9083dfea18457ed38204d"
    assert RETRY7_SAMPLER_SHA256 == "623ddb69cff4b6c0247d6389bbf803d6fcfe3b3eb9856fc9c83fdf2b469662ee"


def test_retry7_model_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry7.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry7 ONNX hash mismatch"):
        summarize(model, retry7=True)


def test_retry7_morphology_model_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry7.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry7 ONNX hash mismatch"):
        summarize_morphology(model, retry7=True)


def test_retry7_sampler_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "retry7.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    real_sha = diagnose_retry._sha

    def fake_sha(path: Path) -> str:
        if path == model:
            return RETRY7_ONNX_SHA256
        if path.name == "AUDIT.json":
            return RETRY7_GENERATOR_AUDIT_SHA256
        if path.name == "negative_sampler.py":
            return "0" * 64
        return real_sha(path)

    monkeypatch.setattr(diagnose_retry, "_sha", fake_sha)
    with pytest.raises(ValueError, match="retry7 sampler hash mismatch"):
        summarize(model, retry7=True)


def test_retry7_morphology_sampler_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "retry7.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    real_sha = diagnose_retry._sha

    def fake_sha(path: Path) -> str:
        if path == model:
            return RETRY7_ONNX_SHA256
        if path.name == "AUDIT.json":
            return RETRY7_GENERATOR_AUDIT_SHA256
        if path.name == "negative_sampler.py":
            return "0" * 64
        return real_sha(path)

    monkeypatch.setattr(diagnose_retry, "_sha", fake_sha)
    with pytest.raises(ValueError, match="retry7 sampler hash mismatch"):
        summarize_morphology(model, retry7=True)


def test_retry7_report_matches_fixed_metrics_and_is_aggregate_only() -> None:
    report = json.loads(Path(__file__).with_name("V24_RETRY7_DIAGNOSIS.json").read_text(encoding="utf-8"))
    metrics = report["fixed_threshold_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (1991, 46, 13)
    assert metrics["precision"] == pytest.approx(0.9774177712322042)
    assert metrics["recall"] == pytest.approx(0.9935129740518962)
    assert metrics["prohibited_structure_hits"] == 0
    attribution = report["retry7_attribution"]
    assert attribution["accepted_false_positives"] == {
        "connector_anchor": 0,
        "generic": 46,
        "topology_fragment": 0,
        "topology_junction": 0,
    }
    assert attribution["above_threshold"] == {
        "artifact": 80,
        "connector_anchor": 9,
        "generic": 1651,
        "ocr_heavy": 1,
        "topology_fragment": 1,
        "topology_junction": 4,
    }
    assert report["binding"]["model_sha256"] == RETRY7_ONNX_SHA256
    assert report["binding"]["generator_audit_sha256"] == RETRY7_GENERATOR_AUDIT_SHA256
    assert report["binding"]["generator_dev_split_sha256"] == RETRY7_DEV_SPLIT_SHA256
    assert report["binding"]["negative_sampler_sha256"] == RETRY7_SAMPLER_SHA256
    assert report["scope"]["threshold"] == 0.25
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["private_data"] is False
    assert report["scope"]["real_dev_reads"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["scope"]["case_ids_or_pixels_emitted"] is False
    assert "scene_id" not in report and "truth_rows" not in report and "pixels" not in report


def test_retry7_morphology_report_matches_bindings_and_is_aggregate_only() -> None:
    report = json.loads(Path(__file__).with_name("V24_RETRY7_MORPHOLOGY_DIAGNOSIS.json").read_text(encoding="utf-8"))
    assert report["schema"] == "graphreader.marker-center-mask-preserving-v24-retry7-morphology-diagnosis.v1"
    assert report["accepted_generic_false_positive_count"] == 46
    assert set(report["morphology_quantiles"]) == {
        "positives", "negative_below_025", "negative_above_025", "accepted_generic_false_positives"
    }
    assert report["binding"]["model_sha256"] == RETRY7_ONNX_SHA256
    assert report["binding"]["generator_audit_sha256"] == RETRY7_GENERATOR_AUDIT_SHA256
    assert report["binding"]["generator_dev_split_sha256"] == RETRY7_DEV_SPLIT_SHA256
    assert report["binding"]["negative_sampler_sha256"] == RETRY7_SAMPLER_SHA256
    assert report["scope"]["threshold"] == 0.25
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["private_data"] is False
    assert report["scope"]["real_dev_reads"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["scope"]["case_ids_or_pixels_emitted"] is False
    assert report["threshold_change_proposed"] is False
    assert "scene_id" not in report and "truth_rows" not in report and "pixels" not in report
