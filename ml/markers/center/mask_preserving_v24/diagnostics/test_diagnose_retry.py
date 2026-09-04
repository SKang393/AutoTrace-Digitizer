# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import json
import sys
from pathlib import Path

import torch
import pytest

from ml.markers.center.mask_preserving_v24.diagnostics import diagnose_retry
from ml.markers.center.mask_preserving_v24.diagnostics.diagnose_retry import (
    MORPHOLOGY_KEYS,
    RETRY9_CONFIG_SHA256,
    RETRY9_DEV_SPLIT_SHA256,
    RETRY9_GENERATOR_AUDIT_SHA256,
    RETRY9_ONNX_SHA256,
    RETRY9_OPENED_SEAL_SHA256,
    RETRY9_RUNNER_SOURCE_BUNDLE_SHA256,
    RETRY9_SAMPLER_SHA256,
    STRATA,
    _patch_morphology,
    _quantiles,
    _retry9_strata,
    _strata,
    main,
    summarize_morphology,
)


def test_quantiles_are_aggregate_and_deterministic():
    assert _quantiles([0.1, 0.2, 0.3])["count"] == 3
    assert _quantiles([]) == {"count": 0}


def test_report_shape_has_required_strata_without_case_fields(tmp_path):
    report = {"proposals": {"negative_strata": {name: {} for name in STRATA}}}
    path = tmp_path / "report.json"; path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(loaded["proposals"]["negative_strata"]) == STRATA
    assert not any(key in loaded for key in ("scene_ids", "truth_rows", "pixels"))

def test_morphology_feature_contract():
    features = _patch_morphology(torch.zeros((3, 33, 33)))
    assert tuple(features) == MORPHOLOGY_KEYS
    assert features["dark_fraction_ge_012"] == 0.0
    assert features["max_ring_support_3_12"] == 0.0

def test_covariance_ratio_is_major_over_minor():
    patch = torch.zeros((3, 33, 33))
    patch[0, 16, 10:23] = 1.0
    features = _patch_morphology(patch)
    assert features["covariance_eigen_ratio"] > 100.0

def test_ring_uses_rounded_euclidean_points():
    patch = torch.zeros((3, 33, 33))
    for i in range(8):
        x = int(round(16 + 5 * __import__('math').cos(i*__import__('math').pi/4)))
        y = int(round(16 + 5 * __import__('math').sin(i*__import__('math').pi/4)))
        patch[0, y, x] = 1.0
    assert _patch_morphology(patch)["max_ring_support_3_12"] == 8.0


def test_retry9_strata_are_geometry_and_patch_only(monkeypatch: pytest.MonkeyPatch):
    patches = torch.zeros((3, 3, 33, 33))
    labels = torch.zeros(3, dtype=torch.bool)
    hard = torch.zeros(3, dtype=torch.bool)
    monkeypatch.setattr(
        diagnose_retry,
        "_features",
        lambda _patches: {name: torch.zeros(3, dtype=torch.bool) for name in ("faint_low", "faint_p05", "ocr_heavy", "artifact")},
    )
    names = _retry9_strata(
        patches,
        hard,
        labels,
        band_indices={1},
        topology_by_index={2: "topology_junction"},
        connector_indices=set(),
    )
    assert names[0] in STRATA
    assert names[1] == "generic_connector_band"
    assert names[2] == "generic"


def test_retry9_morphology_binding_and_scope_are_aggregate_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    model = tmp_path / "retry9.onnx"
    model.write_bytes(b"test model")
    audit = tmp_path / "ml/markers/center/real_range_generator_v1/AUDIT.json"
    config = tmp_path / "ml/markers/center/mask_preserving_v24/training/p1.json"
    sampler = tmp_path / "ml/markers/center/real_range_generator_v1/negative_sampler.py"
    seal = tmp_path / "ml/markers/training-seals/marker-center/marker-center-mask-preserving-v24/P1/opened.json"
    for path in (audit, config, sampler, seal):
        path.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({"splits": {"dev": {"aggregate_sha256": RETRY9_DEV_SPLIT_SHA256}}}), encoding="utf-8")
    config.write_text(json.dumps({"expected_runner_source_bundle_sha256": RETRY9_RUNNER_SOURCE_BUNDLE_SHA256}), encoding="utf-8")
    sampler.write_text("# test fixture\n", encoding="utf-8")
    seal.write_text(json.dumps({
        "status": "opened",
        "budget_status": "pending_sealed_read",
        "binding": {
            "candidate_config_sha256": RETRY9_CONFIG_SHA256,
            "runner_source_bundle_sha256": RETRY9_RUNNER_SOURCE_BUNDLE_SHA256,
        },
    }), encoding="utf-8")
    real_sha = diagnose_retry._sha

    def fake_sha(path: Path) -> str:
        if path == model:
            return RETRY9_ONNX_SHA256
        if path.name == "AUDIT.json":
            return RETRY9_GENERATOR_AUDIT_SHA256
        if path.name == "negative_sampler.py":
            return RETRY9_SAMPLER_SHA256
        if path.name == "p1.json":
            return RETRY9_CONFIG_SHA256
        if path.name == "opened.json":
            return RETRY9_OPENED_SEAL_SHA256
        return real_sha(path)

    class _FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [type("Value", (), {"name": "input"})()]

        def get_outputs(self):
            return [type("Value", (), {"name": "output"})()]

        def run(self, _outputs, _inputs):
            return [__import__("numpy").empty((0, 4), dtype="float32")]

    monkeypatch.setattr(diagnose_retry, "_sha", fake_sha)
    monkeypatch.setattr(diagnose_retry, "ROOT", tmp_path)
    monkeypatch.setattr(diagnose_retry.ort, "InferenceSession", lambda *_args, **_kwargs: _FakeSession())
    monkeypatch.setattr(diagnose_retry, "build_split", lambda _split: [])

    report = summarize_morphology(model, retry9=True)

    assert report["schema"] == "graphreader.marker-center-mask-preserving-v24-retry9-morphology-diagnosis.v1"
    assert report["binding"]["model_sha256"] == RETRY9_ONNX_SHA256
    assert report["binding"]["configuration_sha256"] == RETRY9_CONFIG_SHA256
    assert report["binding"]["generator_audit_sha256"] == RETRY9_GENERATOR_AUDIT_SHA256
    assert report["binding"]["generator_dev_split_sha256"] == RETRY9_DEV_SPLIT_SHA256
    assert report["binding"]["negative_sampler_sha256"] == RETRY9_SAMPLER_SHA256
    assert report["binding"]["runner_source_bundle_sha256"] == RETRY9_RUNNER_SOURCE_BUNDLE_SHA256
    assert report["binding"]["opened_seal_sha256"] == RETRY9_OPENED_SEAL_SHA256
    assert tuple(report["binding"]["negative_sampler_priority"]) == STRATA + ("generic_connector_band",)
    assert report["scope"] == {
        "synthetic_only": True,
        "split": "real-range-generator-v1-dev",
        "scene_count": 0,
        "threshold": 0.25,
        "positive_label_distance_px": 3.0,
        "private_data": False,
        "real_dev_reads": 0,
        "real_sealed_reads": 0,
        "optimizer_steps": 0,
        "case_ids_or_pixels_emitted": False,
        "retry_mode": "retry9",
    }
    assert tuple(report["negative_strata_counts"]) == STRATA + ("generic_connector_band",)


def test_retry9_morphology_cli_dispatches_to_morphology(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    model = tmp_path / "retry9.onnx"
    output = tmp_path / "report.json"
    captured = {}

    def fake_summarize(path: Path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return {"scope": {"synthetic_only": True}, "retry_mode": "retry9"}

    monkeypatch.setattr(diagnose_retry, "summarize_morphology", fake_summarize)
    monkeypatch.setattr(sys, "argv", ["diagnose_retry.py", "--model", str(model), "--output", str(output), "--morphology", "--retry9"])

    assert main() == 0
    assert captured == {"path": model.resolve(), "kwargs": {"retry3": False, "retry7": False, "retry9": True}}
    assert json.loads(output.read_text(encoding="utf-8"))["retry_mode"] == "retry9"


def test_retry8_morphology_remains_unsupported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(sys, "argv", ["diagnose_retry.py", "--model", str(tmp_path / "model.onnx"), "--output", str(tmp_path / "report.json"), "--morphology", "--retry8"])
    with pytest.raises(SystemExit):
        main()
