# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import torch

from ml.ocr.project_numeric_v1 import candidate3_train as train_module
from ml.ocr.project_numeric_v1 import verify_candidate3 as verifier
from ml.ocr.project_numeric_v1.candidate2_dataset import build_candidate2_split
from ml.ocr.project_numeric_v1.candidate2_protocol import (
    protocol_configuration as candidate2_configuration,
)
from ml.ocr.project_numeric_v1.candidate3_model import SpatialQuerySlotRecognizer
from ml.ocr.project_numeric_v1.candidate3_protocol import (
    ARCHITECTURE,
    CANDIDATE_ID,
    CANONICAL_OUTPUT_PATH,
    ProtocolViolation,
    assert_candidate_execution_allowed,
    protocol_configuration,
    validate_frozen_protocol,
)
from ml.ocr.project_numeric_v1.dataset import prepare_inputs, split_fingerprint
from ml.ocr.project_numeric_v1.protocol import CLASS_COUNT


def test_candidate3_protocol_changes_only_architecture_and_consumes_budget() -> None:
    frozen = validate_frozen_protocol()
    configuration = frozen["configuration"]
    candidate2 = candidate2_configuration()

    assert configuration == protocol_configuration()
    assert configuration["candidate_index"] == configuration["maximum_candidates"] == 3
    assert configuration["one_factor_change"]["factor"] == "recognizer architecture"
    assert configuration["one_factor_change"]["sealed_results_used"] is False
    assert configuration["architecture"] == ARCHITECTURE
    for key in (
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "objective",
        "splits",
        "gates",
    ):
        assert configuration[key] == candidate2[key]


def test_candidate3_attention_model_preserves_runtime_contract() -> None:
    samples = build_candidate2_split("validation", positive_count=8, negative_count=2)
    model = SpatialQuerySlotRecognizer().eval()

    with torch.inference_mode():
        time_logits, role_logits = model(prepare_inputs(samples))

    assert list(time_logits.shape) == [10, 32, CLASS_COUNT]
    assert list(role_logits.shape) == [10, 2]
    assert model.slot_classifier.out_features == CLASS_COUNT
    assert not hasattr(model, "ctc")
    assert not hasattr(model, "glyph_isolator")
    assert not hasattr(model, "bottleneck")


def test_candidate3_reuses_every_candidate2_split_byte() -> None:
    frozen = validate_frozen_protocol()
    for split in ("train", "validation", "sealed_test"):
        assert split_fingerprint(build_candidate2_split(split)) == (
            frozen["split_fingerprints"][split]
        )


def test_candidate3_runner_accepts_only_canonical_unused_output(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolation, match="rejects candidate"):
        assert_candidate_execution_allowed("candidate-2", CANONICAL_OUTPUT_PATH)
    with pytest.raises(ProtocolViolation, match="canonical ignored path"):
        assert_candidate_execution_allowed(CANDIDATE_ID, tmp_path / CANDIDATE_ID)


def test_candidate3_commit_guard_runs_before_dataset_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / CANDIDATE_ID
    monkeypatch.setattr(train_module, "assert_candidate_execution_allowed", lambda *_: None)
    monkeypatch.setattr(
        train_module,
        "verify_committed_preregistration",
        lambda: (_ for _ in ()).throw(ProtocolViolation("not committed")),
    )
    monkeypatch.setattr(
        train_module,
        "build_candidate2_split",
        lambda *_: pytest.fail("data must not build before committed guard"),
    )

    with pytest.raises(ProtocolViolation, match="not committed"):
        train_module.run(output, CANDIDATE_ID)
    assert not output.exists()


def test_candidate3_source_binding_verifies_and_rejects_mutation(tmp_path: Path) -> None:
    assert verifier.verify_source_binding()["binding_valid"] is True
    for relative in verifier.REQUIRED_SOURCE_ROLES.values():
        source = verifier.REPOSITORY_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    model_path = tmp_path / verifier.REQUIRED_SOURCE_ROLES["candidate3_model"]
    payload = bytearray(model_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    model_path.write_bytes(payload)

    with pytest.raises(ProtocolViolation, match="SHA-256 mismatch for candidate3_model"):
        verifier.verify_source_binding(tmp_path)
