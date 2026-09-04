# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics import diagnose_retry
from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    RETRY8_CONFIG_SHA256,
    RETRY8_DEV_SPLIT_SHA256,
    RETRY8_GENERATOR_AUDIT_SHA256,
    RETRY8_ONNX_SHA256,
    RETRY8_OPENED_SEAL_SHA256,
    RETRY8_RUNNER_SOURCE_BUNDLE_SHA256,
    RETRY8_SAMPLER_SHA256,
    summarize,
)


def test_retry8_hash_bindings_are_explicit() -> None:
    assert RETRY8_ONNX_SHA256 == "d6f8e9bc64c34f1bb646b6d150e1ccead45e26684836a413a5b904da7f40b5ab"
    assert RETRY8_CONFIG_SHA256 == "80624cf563b4a547b8c81c5021b785da9cfee8739b4e512ab33c79d1bd7fdb88"
    assert RETRY8_GENERATOR_AUDIT_SHA256 == "1d71d76956e24f0c1a230c9c27e59aecc0d0cd64a04ca9c0d26ef171838ce26b"
    assert RETRY8_DEV_SPLIT_SHA256 == "72dda9b9031f3050d72f5946105576cad89fe938f36f619f84ef4c9cafa8e566"
    assert RETRY8_SAMPLER_SHA256 == "623ddb69cff4b6c0247d6389bbf803d6fcfe3b3eb9856fc9c83fdf2b469662ee"
    assert RETRY8_RUNNER_SOURCE_BUNDLE_SHA256 == "ffd479f41f0fe6525b24e1ac6df1d2e2acd187d58313b526feea3e1c4008dab7"
    assert RETRY8_OPENED_SEAL_SHA256 == "7d462c8c400a5d2b021ee239e5e192d96dfebdf2415c82d5a5e5fadff1a9832a"


def test_retry8_model_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry8.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry8 ONNX hash mismatch"):
        summarize(model, retry8=True)


def _patched_retry8_sha(monkeypatch: pytest.MonkeyPatch, model: Path, *, config: str, sampler: str) -> None:
    real_sha = diagnose_retry._sha

    def fake_sha(path: Path) -> str:
        if path == model:
            return RETRY8_ONNX_SHA256
        if path.name == "AUDIT.json":
            return RETRY8_GENERATOR_AUDIT_SHA256
        if path.name == "p1.json":
            return config
        if path.name == "negative_sampler.py":
            return sampler
        return real_sha(path)

    monkeypatch.setattr(diagnose_retry, "_sha", fake_sha)


def test_retry8_config_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "retry8.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    _patched_retry8_sha(monkeypatch, model, config="0" * 64, sampler=RETRY8_SAMPLER_SHA256)
    with pytest.raises(ValueError, match="retry8 config hash mismatch"):
        summarize(model, retry8=True)


def test_retry8_sampler_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "retry8.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    _patched_retry8_sha(monkeypatch, model, config=RETRY8_CONFIG_SHA256, sampler="0" * 64)
    with pytest.raises(ValueError, match="retry8 sampler hash mismatch"):
        summarize(model, retry8=True)


def test_retry8_opened_seal_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "retry8.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    real_sha = diagnose_retry._sha

    def fake_sha(path: Path) -> str:
        if path == model:
            return RETRY8_ONNX_SHA256
        if path.name == "AUDIT.json":
            return RETRY8_GENERATOR_AUDIT_SHA256
        if path.name == "p1.json":
            return RETRY8_CONFIG_SHA256
        if path.name == "negative_sampler.py":
            return RETRY8_SAMPLER_SHA256
        if path.name == "opened.json":
            return "0" * 64
        return real_sha(path)

    monkeypatch.setattr(diagnose_retry, "_sha", fake_sha)
    with pytest.raises(ValueError, match="retry8 opened seal hash mismatch"):
        summarize(model, retry8=True)


def test_retry8_report_matches_fixed_metrics_and_is_aggregate_only() -> None:
    report = json.loads(Path(__file__).with_name("V24_RETRY8_DIAGNOSIS.json").read_text(encoding="utf-8"))
    metrics = report["fixed_threshold_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (1990, 179, 14)
    assert metrics["precision"] == pytest.approx(0.9174734900875979)
    assert metrics["recall"] == pytest.approx(0.9930139720558883)
    assert metrics["prohibited_structure_hits"] == 0
    assert report["retry8_attribution"]["accepted_false_positives"] == {
        "artifact": 1,
        "connector_anchor": 1,
        "generic": 171,
        "topology_fragment": 4,
        "topology_junction": 2,
    }
    assert report["scope"]["threshold"] == 0.25
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["private_data"] is False
    assert report["scope"]["real_dev_reads"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["scope"]["case_ids_or_pixels_emitted"] is False
    assert report["scope"]["sealed_runs"] == 0
    assert report["binding"]["configuration_sha256"] == RETRY8_CONFIG_SHA256
    assert report["binding"]["generator_audit_sha256"] == RETRY8_GENERATOR_AUDIT_SHA256
    assert report["binding"]["generator_dev_split_sha256"] == RETRY8_DEV_SPLIT_SHA256
    assert report["binding"]["negative_sampler_sha256"] == RETRY8_SAMPLER_SHA256
    assert report["binding"]["runner_source_bundle_sha256"] == RETRY8_RUNNER_SOURCE_BUNDLE_SHA256
    assert report["binding"]["opened_seal_sha256"] == RETRY8_OPENED_SEAL_SHA256
    assert report["binding"]["opened_seal_status"] == "opened"
    assert report["binding"]["seal_budget_status"] == "pending_sealed_read"
    assert "scene_id" not in report and "truth_rows" not in report and "pixels" not in report
