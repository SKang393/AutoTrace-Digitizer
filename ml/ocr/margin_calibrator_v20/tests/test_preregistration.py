# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.ocr.margin_calibrator_v20.model import MarginSeparatedProposalCalibrator
from ml.ocr.margin_calibrator_v20.prepare_split import SPLIT_SOURCE_PATHS
from ml.ocr.margin_calibrator_v20.protocol import (
    FEATURE_COUNT,
    NEGATIVE_LOGIT_MARGIN,
    POSITIVE_LOGIT_MARGIN,
    REVISION,
    SPLITS,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
)
from ml.ocr.margin_calibrator_v20.sealed_gate import EVALUATOR_SOURCE_PATHS
from ml.ocr.margin_calibrator_v20.train_p1 import RUNNER_SOURCE_PATHS


ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "ml/ocr/margin_calibrator_v20"


def _read(relative: str) -> dict[str, object]:
    return json.loads((MODULE / relative).read_text(encoding="utf-8"))


def test_protocol_trigger_and_margin_contract_are_frozen() -> None:
    assert (MODULE / "PROTOCOL.json").read_bytes() == canonical_json_bytes(protocol_configuration())
    assert sha256_file(ROOT / TRIGGER_RESULT_PATH) == TRIGGER_RESULT_SHA256
    protocol = protocol_configuration()
    assert protocol["revision"] == REVISION
    assert protocol["currently_preregistered_candidate"] == "P1"
    assert protocol["execution_authorized"] is False
    assert protocol["trigger_evidence"]["case_level_details_used"] is False
    assert protocol["trigger_evidence"]["fixture_bytes_scene_truth_or_case_identity_used"] is False
    assert POSITIVE_LOGIT_MARGIN == -NEGATIVE_LOGIT_MARGIN


def test_fresh_split_and_source_identities_are_checksum_bound() -> None:
    selection = _read("SELECTION_MANIFEST.json")
    assert selection["v19_fixture_bytes_scene_truth_or_case_identity_reused"] is False
    assert selection["sealed_public_truth_available_to_candidate"] is False
    assert selection["split_generator_source_bundle_sha256"] == source_bundle_sha256(ROOT, SPLIT_SOURCE_PATHS)
    registrations = {item.split: item for item in SPLITS}
    assert len({item.seed_offset for item in SPLITS}) == 3
    assert len({item.renderer_family for item in SPLITS}) == 3
    assert len({item.degradation_family for item in SPLITS}) == 3
    for split in ("train", "validation", "sealed_public"):
        item = selection[split]
        assert item["scene_count"] == registrations[split].scene_count
        assert sha256_file(ROOT / item["fixture_archive_path"]) == item["fixture_archive_sha256"]
        assert sha256_file(ROOT / item["private_manifest_path"]) == item["private_manifest_sha256"]


def test_model_is_small_deterministic_and_export_shaped() -> None:
    first = MarginSeparatedProposalCalibrator(seed=20262220)
    second = MarginSeparatedProposalCalibrator(seed=20262220)
    values = torch.from_numpy(np.arange(FEATURE_COUNT * 3, dtype=np.float32).reshape(3, FEATURE_COUNT) / 100.0)
    with torch.inference_mode():
        first_output = first(values)
        second_output = second(values)
    assert first_output.shape == (3, 2)
    assert torch.equal(first_output, second_output)
    assert sum(parameter.numel() for parameter in first.parameters()) < 2_200


def test_candidate_and_public_gate_remain_execution_blocked() -> None:
    config = _read("training/p1.json")
    gate = _read("gates/sealed-public-v1.json")
    seal = _read("SEALED_PUBLIC_TEST_SEAL.json")
    ledger = json.loads(
        (ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in ledger["revisions"] if item.get("revision") == REVISION)
    assert config["expected_runner_source_bundle_sha256"] == source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS)
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(ROOT, EVALUATOR_SOURCE_PATHS)
    assert seal["truth_hidden_from_candidate_runner"] is True
    assert seal["public_gate_evaluations"] == 0
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["candidate_config_sha256"]["P1"] == sha256_file(MODULE / "training/p1.json")
    assert not (MODULE / "P1_RESULT.json").exists()
    assert not (MODULE / "PUBLIC_GATE_RESULT.json").exists()

