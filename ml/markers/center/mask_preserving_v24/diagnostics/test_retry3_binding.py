# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from pathlib import Path
import json

import pytest

from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    RETRY2_ONNX_SHA256,
    RETRY3_DEV_SPLIT_SHA256,
    RETRY3_GENERATOR_AUDIT_SHA256,
    RETRY3_ONNX_SHA256,
    summarize_morphology,
)


def test_retry3_binding_is_explicit_and_uses_current_generator_identity() -> None:
    assert RETRY3_ONNX_SHA256 == "0d80d1994d7b33241c795c9e6f92c802750555a62c3cd3335777eb969fb5083a"
    assert RETRY3_GENERATOR_AUDIT_SHA256 == "3568fff359e3541be14bf1f02774887c8110ded9f02fd95a8f4ff680e8639d69"
    assert RETRY3_DEV_SPLIT_SHA256 == "050df194849c9e787d786624b26fd268e7f7a1832c271868521d89bf6588e960"


def test_retry3_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry3.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    with pytest.raises(ValueError, match="retry3 ONNX hash mismatch"):
        summarize_morphology(model, retry3=True)


def test_retry2_default_hash_check_remains_fail_closed(tmp_path: Path) -> None:
    model = tmp_path / "retry2.onnx"
    model.write_bytes(b"not-an-onnx-payload")
    assert RETRY2_ONNX_SHA256 != RETRY3_ONNX_SHA256
    with pytest.raises(ValueError, match="retry2 ONNX hash mismatch"):
        summarize_morphology(model)


def test_retry3_report_is_separate_and_aggregate_only() -> None:
    report_path = Path(__file__).with_name("V24_RETRY3_MORPHOLOGY_DIAGNOSIS.json")
    report = report_path.read_text(encoding="utf-8")
    parsed = json.loads(report)
    assert '"schema": "graphreader.marker-center-mask-preserving-v24-retry3-morphology-diagnosis.v1"' in report
    assert '"model_sha256": "0d80d1994d7b33241c795c9e6f92c802750555a62c3cd3335777eb969fb5083a"' in report
    assert '"generator_dev_split_sha256": "050df194849c9e787d786624b26fd268e7f7a1832c271868521d89bf6588e960"' in report
    assert '"real_dev_reads": 0' in report
    assert '"real_sealed_reads": 0' in report
    assert '"optimizer_steps": 0' in report
    assert parsed["scope"]["case_ids_or_pixels_emitted"] is False
