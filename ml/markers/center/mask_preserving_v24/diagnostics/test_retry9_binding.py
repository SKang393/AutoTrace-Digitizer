# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics import diagnose_retry
from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    RETRY9_CONFIG_SHA256,
    RETRY9_DEV_SPLIT_SHA256,
    RETRY9_GENERATOR_AUDIT_SHA256,
    RETRY9_ONNX_SHA256,
    RETRY9_OPENED_SEAL_SHA256,
    RETRY9_RUNNER_SOURCE_BUNDLE_SHA256,
    RETRY9_SAMPLER_SHA256,
    summarize,
)


def test_retry9_hash_bindings_are_explicit() -> None:
    assert RETRY9_ONNX_SHA256 == "4dece2eeb87229d5d57e0d2d714c1915ebecf8e9475b0d466a03dd970993fdb4"
    assert RETRY9_CONFIG_SHA256 == "9d6f9da5c3f0526cb2719c3425bb2bb64a98cdbf78bfb6b7162ab0adefba239c"
    assert RETRY9_GENERATOR_AUDIT_SHA256 == "1d71d76956e24f0c1a230c9c27e59aecc0d0cd64a04ca9c0d26ef171838ce26b"
    assert RETRY9_DEV_SPLIT_SHA256 == "72dda9b9031f3050d72f5946105576cad89fe938f36f619f84ef4c9cafa8e566"
    assert RETRY9_SAMPLER_SHA256 == "98f970c90943d30a334c951ac3084db5fa62e56eebade252ecd3042e43f22286"
    assert RETRY9_RUNNER_SOURCE_BUNDLE_SHA256 == "b8736824df79aadeacded8fec996c932b92f8c1802fd6aced73907958c6f1cf3"
    assert RETRY9_OPENED_SEAL_SHA256 == "0fc36f3ec59d2ff1c785f926d33aa762a67c336c5f50a97ab5eb195540b6d611"


def test_retry9_model_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry9.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry9 ONNX hash mismatch"):
        summarize(model, retry9=True)


def test_retry9_config_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "retry9.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    real_sha = diagnose_retry._sha

    def fake_sha(path: Path) -> str:
        if path == model:
            return RETRY9_ONNX_SHA256
        if path.name == "AUDIT.json":
            return RETRY9_GENERATOR_AUDIT_SHA256
        if path.name == "p1.json":
            return "0" * 64
        if path.name == "negative_sampler.py":
            return RETRY9_SAMPLER_SHA256
        return real_sha(path)

    monkeypatch.setattr(diagnose_retry, "_sha", fake_sha)
    with pytest.raises(ValueError, match="retry9 config hash mismatch"):
        summarize(model, retry9=True)


def test_retry9_opened_seal_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "retry9.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    real_sha = diagnose_retry._sha

    def fake_sha(path: Path) -> str:
        if path == model:
            return RETRY9_ONNX_SHA256
        if path.name == "AUDIT.json":
            return RETRY9_GENERATOR_AUDIT_SHA256
        if path.name == "p1.json":
            return RETRY9_CONFIG_SHA256
        if path.name == "negative_sampler.py":
            return RETRY9_SAMPLER_SHA256
        if path.name == "opened.json":
            return "0" * 64
        return real_sha(path)

    monkeypatch.setattr(diagnose_retry, "_sha", fake_sha)
    with pytest.raises(ValueError, match="retry9 opened seal hash mismatch"):
        summarize(model, retry9=True)


def test_retry9_report_matches_fixed_metrics_and_is_aggregate_only() -> None:
    report = json.loads(Path(__file__).with_name("V24_RETRY9_DIAGNOSIS.json").read_text(encoding="utf-8"))
    metrics = report["fixed_threshold_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (1992, 69, 12)
    assert metrics["precision"] == pytest.approx(0.9665211062590975)
    assert metrics["recall"] == pytest.approx(0.9940119760479041)
    assert metrics["prohibited_structure_hits"] == 0
    assert metrics["accepted_false_positive_attribution"] == {
        "generic": 40,
        "generic_connector_band": 28,
        "topology_junction": 1,
    }
    assert report["proposals"]["negative_strata"]["generic_connector_band"]["capacity"] == 50241
    assert report["retry9_attribution"]["generic_connector_band"] == {
        "above_threshold": 965,
        "accepted_false_positives": 28,
    }
    assert report["binding"]["model_sha256"] == RETRY9_ONNX_SHA256
    assert report["binding"]["configuration_sha256"] == RETRY9_CONFIG_SHA256
    assert report["binding"]["generator_audit_sha256"] == RETRY9_GENERATOR_AUDIT_SHA256
    assert report["binding"]["generator_dev_split_sha256"] == RETRY9_DEV_SPLIT_SHA256
    assert report["binding"]["negative_sampler_sha256"] == RETRY9_SAMPLER_SHA256
    assert report["binding"]["runner_source_bundle_sha256"] == RETRY9_RUNNER_SOURCE_BUNDLE_SHA256
    assert report["binding"]["opened_seal_sha256"] == RETRY9_OPENED_SEAL_SHA256
    assert report["scope"]["threshold"] == 0.25
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["private_data"] is False
    assert report["scope"]["real_dev_reads"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["scope"]["sealed_runs"] == 0
    assert report["scope"]["case_ids_or_pixels_emitted"] is False
    assert "scene_id" not in report and "truth_rows" not in report and "pixels" not in report
