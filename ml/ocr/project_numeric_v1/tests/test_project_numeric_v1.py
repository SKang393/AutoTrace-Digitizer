# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from ml.ocr.project_numeric_v1 import dataset as dataset_module
from ml.ocr.project_numeric_v1 import train as train_module
from ml.ocr.project_numeric_v1 import verify_preregistration as verifier
from ml.ocr.project_numeric_v1.dataset import (
    build_split,
    encode_slots,
    prepare_inputs,
    split_fingerprint,
)
from ml.ocr.project_numeric_v1.model import GlobalSemanticSlotRecognizer
from ml.ocr.project_numeric_v1.protocol import (
    ALPHABET,
    CANDIDATES,
    CANONICAL_OUTPUT_PATH,
    CLASS_COUNT,
    MAXIMUM_CANDIDATES,
    ProtocolViolation,
    SLOT_TIME_INDICES,
    assert_candidate_execution_allowed,
    protocol_configuration,
    validate_frozen_protocol,
)


def test_frozen_protocol_exactly_matches_code_and_three_candidate_budget() -> None:
    frozen = validate_frozen_protocol()

    assert frozen["configuration"] == protocol_configuration()
    assert MAXIMUM_CANDIDATES == 3
    assert len(CANDIDATES) == 3
    assert CANDIDATES[0].configuration_state == "frozen-and-eligible-after-commit"
    assert all(
        candidate.configuration_state == "reserved-not-registered"
        for candidate in CANDIDATES[1:]
    )


def test_only_candidate_one_and_canonical_ignored_output_are_executable(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolation, match="not frozen for execution"):
        assert_candidate_execution_allowed("candidate-2", CANONICAL_OUTPUT_PATH)
    with pytest.raises(ProtocolViolation, match="canonical ignored path"):
        assert_candidate_execution_allowed("candidate-1", tmp_path / "candidate-1")


def test_architecture_is_whole_crop_and_emits_runtime_contract() -> None:
    samples = build_split("validation", positive_count=8, negative_count=2)
    model = GlobalSemanticSlotRecognizer().eval()

    with torch.inference_mode():
        time_logits, role_logits = model(prepare_inputs(samples))

    assert list(time_logits.shape) == [10, 32, CLASS_COUNT]
    assert list(role_logits.shape) == [10, 2]
    assert model.slot_classifier.out_features == 8 * CLASS_COUNT
    assert not hasattr(model, "ctc")
    assert not hasattr(model, "glyph_isolator")
    for time_index in range(32):
        if time_index not in SLOT_TIME_INDICES:
            assert time_logits[:, time_index, 0].min().item() == 12.0


@pytest.mark.parametrize("split", ["train", "validation", "sealed_test"])
def test_procedural_splits_are_deterministic_and_cover_required_cases(split: str) -> None:
    first = build_split(split, positive_count=16, negative_count=9)  # type: ignore[arg-type]
    second = build_split(split, positive_count=16, negative_count=9)  # type: ignore[arg-type]

    assert split_fingerprint(first) == split_fingerprint(second)
    assert {sample.case for sample in first if sample.target_text} == {
        "integer",
        "decimal",
        "negative",
        "percentage",
        "negative_decimal",
        "decimal_percentage",
        "o_zero_ambiguity",
        "l_one_ambiguity",
    }
    assert len({sample.exclusion_kind for sample in first if not sample.target_text}) == 9
    assert all(sample.raster.shape == (32, 128) for sample in first)


def test_renderer_families_are_disjoint_and_do_not_import_exhausted_generator() -> None:
    frozen = validate_frozen_protocol()
    renderers = [entry["renderer_family"] for entry in frozen["configuration"]["splits"]]
    degradations = [entry["degradation_family"] for entry in frozen["configuration"]["splits"]]
    source = inspect.getsource(dataset_module)

    assert len(set(renderers)) == 3
    assert len(set(degradations)) == 3
    assert "ml.ocr.synthetic" not in source
    assert "_GLYPHS" not in source


def test_model_report_does_not_claim_downstream_marker_creation_evidence() -> None:
    source = inspect.getsource(train_module._evaluate)

    assert '"marker_creation_count": 0' not in source
    assert '"marker_creation_evaluated": False' in source
    assert "requires downstream application integration" in source


def test_metric_adapter_uses_reference_prediction_pairs() -> None:
    samples = build_split("validation", positive_count=8, negative_count=2)
    model = GlobalSemanticSlotRecognizer().eval()

    result = train_module._evaluate(model, samples)

    assert result["positive_count"] == 8
    assert result["negative_count"] == 2
    assert 0.0 <= result["exact_match"] <= 1.0
    assert result["marker_creation_evaluated"] is False


def test_recovery_record_binds_exact_checkpoint_and_prohibits_retraining() -> None:
    record = json.loads(
        (Path(train_module.__file__).with_name("RECOVERY_EVALUATION.json")).read_text(
            encoding="utf-8"
        )
    )

    assert record["candidate_id"] == "candidate-1"
    assert record["checkpoint"]["sha256"] == (
        "6e941b2b3b746e092f01bf04a28faea61d0ba0bf584dc83b0530594ddddd8235"
    )
    assert record["retraining_permitted"] is False
    assert record["weight_changes_permitted"] is False
    assert record["approval"] is False


def test_slot_encoding_supports_all_required_characters_and_padding() -> None:
    for text in ("100", "10.5", "-25", "80%", "-1.5", "0.5%"):
        encoded = encode_slots(text)
        assert len(encoded) == 8
        assert all(0 <= value <= len(ALPHABET) for value in encoded)


def test_training_refuses_before_commit_and_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidate-1"
    monkeypatch.setattr(train_module, "assert_candidate_execution_allowed", lambda *_: None)

    def refuse_commit() -> str:
        raise ProtocolViolation("preregistration is not committed")

    monkeypatch.setattr(train_module, "verify_committed_preregistration", refuse_commit)
    monkeypatch.setattr(
        train_module,
        "build_split",
        lambda *_: pytest.fail("dataset must not build before the commit guard"),
    )

    with pytest.raises(ProtocolViolation, match="not committed"):
        train_module.run(output, "candidate-1")
    assert not output.exists()


def test_committed_guard_rejects_scoped_dirty_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verifier,
        "verify_source_binding",
        lambda *_: {"binding_valid": True},
    )

    def fake_git(arguments: list[str], root: Path) -> subprocess.CompletedProcess[str]:
        if arguments[:2] == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(arguments, 0, "main\n", "")
        if arguments[:2] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(arguments, 0, "a" * 40 + "\n", "")
        if arguments[:2] == ["cat-file", "-e"]:
            return subprocess.CompletedProcess(arguments, 0, "", "")
        return subprocess.CompletedProcess(arguments, 0, " M ml/ocr/project_numeric_v1/model.py\n", "")

    monkeypatch.setattr(verifier, "_run_git", fake_git)

    with pytest.raises(ProtocolViolation, match="unchanged committed"):
        verifier.verify_committed_preregistration()


def test_source_binding_verifies_and_rejects_controlled_mutation(tmp_path: Path) -> None:
    assert verifier.verify_source_binding()["binding_valid"] is True
    for relative in verifier.REQUIRED_SOURCE_ROLES.values():
        source = verifier.REPOSITORY_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    model_path = tmp_path / verifier.REQUIRED_SOURCE_ROLES["model"]
    payload = bytearray(model_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    model_path.write_bytes(payload)

    with pytest.raises(ProtocolViolation, match="SHA-256 mismatch for model"):
        verifier.verify_source_binding(tmp_path)
