# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import inspect
from pathlib import Path
import shutil

import pytest

from ml.ocr.project_numeric_v1 import candidate2_dataset as dataset_module
from ml.ocr.project_numeric_v1 import candidate2_train as train_module
from ml.ocr.project_numeric_v1 import verify_candidate2 as verifier
from ml.ocr.project_numeric_v1.candidate2_dataset import build_candidate2_split
from ml.ocr.project_numeric_v1.candidate2_protocol import (
    CANDIDATE_ID,
    CANONICAL_OUTPUT_PATH,
    ProtocolViolation,
    assert_candidate_execution_allowed,
    protocol_configuration,
    validate_frozen_protocol,
)
from ml.ocr.project_numeric_v1.dataset import split_fingerprint
from ml.ocr.project_numeric_v1.protocol import load_frozen_protocol as load_candidate1


def test_candidate2_protocol_is_frozen_with_one_training_change() -> None:
    frozen = validate_frozen_protocol()
    configuration = frozen["configuration"]

    assert configuration == protocol_configuration()
    assert configuration["candidate_id"] == "candidate-2"
    assert configuration["candidate_index"] == 2
    assert configuration["maximum_candidates"] == 3
    assert configuration["one_factor_change"]["factor"] == (
        "training renderer and degradation family"
    )
    assert configuration["one_factor_change"]["sealed_results_used"] is False
    assert configuration["architecture"] == load_candidate1()["configuration"]["architecture"]


def test_candidate2_splits_are_deterministic_and_validation_is_unchanged() -> None:
    frozen = validate_frozen_protocol()
    candidate1 = load_candidate1()

    for split in ("train", "validation", "sealed_test"):
        measured = split_fingerprint(build_candidate2_split(split))
        assert measured == frozen["split_fingerprints"][split]
    assert frozen["split_fingerprints"]["validation"] == (
        candidate1["split_fingerprints"]["validation"]
    )
    assert frozen["split_fingerprints"]["train"] != candidate1["split_fingerprints"]["train"]
    assert frozen["split_fingerprints"]["sealed_test"] != (
        candidate1["split_fingerprints"]["sealed_test"]
    )


def test_candidate2_renderer_does_not_read_validation_or_prior_sealed_definitions() -> None:
    source = inspect.getsource(dataset_module)

    assert "_VALIDATION_BITMAPS" not in source
    assert "_SEALED_SEGMENTS" not in source
    assert "_render_sealed," not in source
    assert "chandler" not in source.lower()
    assert "pillow" not in source.lower()


def test_candidate2_runner_accepts_only_canonical_unused_output(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolation, match="rejects candidate"):
        assert_candidate_execution_allowed("candidate-3", CANONICAL_OUTPUT_PATH)
    with pytest.raises(ProtocolViolation, match="canonical ignored path"):
        assert_candidate_execution_allowed(CANDIDATE_ID, tmp_path / CANDIDATE_ID)


def test_candidate2_commit_guard_runs_before_dataset_or_output(
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


def test_candidate2_source_binding_verifies_and_rejects_mutation(tmp_path: Path) -> None:
    assert verifier.verify_source_binding()["binding_valid"] is True
    for relative in verifier.REQUIRED_SOURCE_ROLES.values():
        source = verifier.REPOSITORY_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    model_path = tmp_path / verifier.REQUIRED_SOURCE_ROLES["base_model"]
    payload = bytearray(model_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    model_path.write_bytes(payload)

    with pytest.raises(ProtocolViolation, match="SHA-256 mismatch for base_model"):
        verifier.verify_source_binding(tmp_path)
