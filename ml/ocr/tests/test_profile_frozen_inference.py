# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from pathlib import Path

import pytest

from ml.ocr import profile_frozen_inference as profiler


def test_profile_scope_is_train_and_validation_only() -> None:
    profiler.validate_profile_scope("train")
    profiler.validate_profile_scope("validation")
    with pytest.raises(ValueError):
        profiler.validate_profile_scope("sealed")


def test_profiler_source_has_no_candidate_or_archive_access() -> None:
    source = Path(profiler.__file__).read_text(encoding="utf-8")
    assert "load_archive(" not in source
    assert "preflight(" not in source
    assert "acquire_training_candidate(" not in source
    assert "sealed_public" not in source
    assert "--output" not in source
    assert ".write_text(" not in source
    assert ".write_bytes(" not in source


def test_frozen_input_checksums_match_protocol() -> None:
    identity = profiler.validate_frozen_inputs()
    assert identity["detector_sha256"] == identity["detector_expected_sha256"]
    assert identity["recognizer_sha256"] == identity["recognizer_expected_sha256"]
    assert identity["recognizer_inference_yaml_sha256"] == identity[
        "recognizer_inference_yaml_expected_sha256"
    ]
    assert identity["role_parent_checkpoint_sha256"] == identity[
        "role_parent_checkpoint_expected_sha256"
    ]


def test_checksum_mismatch_fails_before_profile_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiler, "_sha256", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="detector checksum mismatch"):
        profiler.validate_frozen_inputs()


def test_profile_safety_contract_is_aggregate_only() -> None:
    assert profiler.PROFILE_SAFETY == {
        "sealed_reads": 0,
        "candidate_acquisitions": 0,
        "private_data": False,
        "model_revision_opened": False,
        "checkpoints_written": 0,
        "profile_output_files_written": 0,
    }
