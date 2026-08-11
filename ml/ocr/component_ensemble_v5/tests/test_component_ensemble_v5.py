# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from ml.ocr.component_ensemble_v5.dataset import build_split, isolate_glyphs, split_fingerprint
from ml.ocr.component_ensemble_v5.model import ComponentEnsembleGlyphNet
from ml.ocr.component_ensemble_v5.protocol import (
    ENCODED_GLYPH_WIDTH,
    GLYPH_HEIGHT,
    REVISION,
    STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO,
    TASK,
)
from ml.ocr.component_ensemble_v5.sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from ml.ocr.component_ensemble_v5.train_p1 import CONFIG_PATH, RUNNER_SOURCE_PATHS, _export


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/component_ensemble_v5"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def test_protocol_freezes_distinct_model_and_new_renderer_families() -> None:
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["task"] == TASK
    assert protocol["revision"] == REVISION
    assert protocol["candidate_ids"] == ["P1", "P2", "P3"]
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["experiment_budget"] == 3
    assert protocol["architecture"] == "multi-renderer-fixed-feature-ensemble-mlp-v1"
    assert protocol["exposed_predecessor_cases_used_for_selection"] is False
    families = {(item["renderer_family"], item["degradation_family"]) for item in protocol["splits"]}
    assert len(families) == 3
    assert all("v5" in renderer for renderer, _ in families)
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_selection_splits_reproduce_without_opening_public_truth() -> None:
    selection = json.loads((ROOT / "SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    training = build_split("train")
    validation = build_split("validation")
    assert len(training) == selection["train_sample_count"] == 7168
    assert len(validation) == selection["validation_sample_count"] == 672
    assert split_fingerprint(training) == selection["train_split_fingerprint"]
    assert split_fingerprint(validation) == selection["validation_split_fingerprint"]
    assert selection["sealed_public_truth_available_to_training"] is False
    assert selection["exposed_predecessor_cases_used_for_selection"] is False


def test_positive_validation_components_stay_below_structural_rejection() -> None:
    ratios = [
        float(glyph[0, 0, 20])
        for sample in build_split("validation")
        if sample.exclusion_kind is None
        for glyph in isolate_glyphs(sample.raster)
    ]
    assert ratios
    assert max(ratios) < STRUCTURAL_REJECT_MINIMUM_HEIGHT_RATIO


def test_fixed_feature_model_exports_dynamic_cpu_onnx(tmp_path: Path) -> None:
    model = ComponentEnsembleGlyphNet(seed=20260820).eval()
    example = torch.zeros((8, 1, GLYPH_HEIGHT, ENCODED_GLYPH_WIDTH), dtype=torch.float32)
    onnx_path = tmp_path / "preflight.onnx"
    _export(model, example, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    result = np.asarray(session.run(None, {"glyphs": np.zeros((3, 1, 24, 26), dtype=np.float32)})[0])
    assert result.shape == (3, 14)
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_source_bundles_and_new_public_seal_are_exactly_bound() -> None:
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
    split_paths = tuple(Path(value) for value in protocol["split_generator_source_paths"])
    assert protocol["split_generator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, split_paths)
    training = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert training["expected_runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    gate = json.loads((ROOT / "gates/sealed-public-v1.json").read_text(encoding="utf-8"))
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS
    )
    assert gate["expected_gate_config_sha256"] == sha256_bytes(canonical_json_bytes(GATE_CONFIG))
    seal_path = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    assert gate["sealed_public_test_seal_sha256"] == sha256_file(seal_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["fixture_archive_sha256"] == "3c54da8a410d5aba45bfa131d7a35c8b6d394270c9699212d2159a110865a240"
    assert sha256_file(REPO_ROOT / seal["fixture_archive_path"]) == seal["fixture_archive_sha256"]
    assert seal["predecessor_public_archive_reused"] is False
    assert seal["truth_hidden_from_training_runner"] is True
    assert not (ROOT / "PUBLIC_GATE_REPORT.json").exists()


def test_canonical_budget_records_selected_p1_and_authorizes_only_public_gate() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item["task"] == TASK and item["revision"] == REVISION)
    result = json.loads((ROOT / "P1_RESULT.json").read_text(encoding="utf-8"))
    assert entry["status"] == "selection_passed_public_preregistered"
    assert entry["experiment_budget"] == 3
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["remaining_unregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(REPO_ROOT / CONFIG_PATH)
    assert entry["candidate_checkpoint_sha256"]["P1"] == result["checkpoint_sha256"]
    assert entry["candidate_onnx_sha256"]["P1"] == result["onnx_sha256"]
    assert entry["p1_training_report_sha256"] == result["report_sha256"]
    assert entry["p1_training_opened_seal_sha256"] == result["training_opened_seal_sha256"]
    assert entry["p1_training_result_seal_sha256"] == result["training_result_seal_sha256"]
    seal_root = REPO_ROOT / "ml/markers/training-seals/ocr-recognition/graph-numeric-component-ensemble-v5/P1"
    assert sha256_file(seal_root / "opened.json") == result["training_opened_seal_sha256"]
    assert sha256_file(seal_root / "result.json") == result["training_result_seal_sha256"]
    assert entry["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert entry["selection_manifest_sha256"] == sha256_file(ROOT / "SELECTION_MANIFEST.json")
    assert entry["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert entry["public_gate_config_sha256"] == sha256_file(ROOT / "gates/sealed-public-v1.json")
    assert entry["public_gate_archive_opened"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_authorized"] is True
    assert entry["public_gate_authorized_onnx_sha256"] == result["onnx_sha256"]
    assert entry["public_gate_authorized_selection_report_sha256"] == result["report_sha256"]
    assert result["selection_gate_passed"] is True
    assert result["sealed_public_archive_opened"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_no_tracked_production_manifest_exists() -> None:
    manifests = list((REPO_ROOT / "models/manifest/ocr").glob("*.json"))
    assert all(REVISION not in path.read_text(encoding="utf-8") for path in manifests)
