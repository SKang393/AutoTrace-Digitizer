# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    RETRY4_DEV_SPLIT_SHA256,
    RETRY4_GENERATOR_AUDIT_SHA256,
    RETRY4_ONNX_SHA256,
    summarize,
)


def test_retry4_binding_uses_current_generator_identity() -> None:
    assert RETRY4_ONNX_SHA256 == "697fbcfb961e4c2af36a1a3d68cf5be874412b2939b03c42b59aaa82c4b0de96"
    assert RETRY4_GENERATOR_AUDIT_SHA256 == "9562044526bce45b48254472346ccc1640c6e254915b9e744525154144121748"
    assert RETRY4_DEV_SPLIT_SHA256 == "f453e50e228f54e15bafba83b1a5dda422c435a555e9083dfea18457ed38204d"


def test_retry4_model_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry4.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry4 ONNX hash mismatch"):
        summarize(model, retry4=True)


def test_retry1_default_hash_mismatch_remains_fail_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry1.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry1 ONNX hash mismatch"):
        summarize(model)


def test_retry4_report_is_aggregate_only_and_matches_fixed_metrics() -> None:
    report_path = Path(__file__).with_name("V24_RETRY4_DIAGNOSIS.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "graphreader.marker-center-mask-preserving-v24-retry4-diagnosis.v1"
    metrics = report["fixed_threshold_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (1989, 138, 15)
    assert metrics["precision"] == pytest.approx(0.9351198871650211)
    assert metrics["recall"] == pytest.approx(0.9925149700598802)
    assert report["scope"]["real_dev_reads"] == 0
    assert report["scope"]["real_sealed_reads"] == 0
    assert report["scope"]["optimizer_steps"] == 0
    assert report["scope"]["case_ids_or_pixels_emitted"] is False
    assert report["binding"]["generator_dev_split_sha256"] == RETRY4_DEV_SPLIT_SHA256
    assert set(report["topology"]["capacity"]) == {"topology_junction", "topology_fragment"}
