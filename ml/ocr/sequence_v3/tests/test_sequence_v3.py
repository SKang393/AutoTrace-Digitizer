# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from hashlib import sha256
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.ocr.sequence_v3.dataset import (  # noqa: E402
    CANONICALIZER_INVERTS_SHARED_GENERATOR_WIDTH_TRANSFORMS,
    CLASS_COUNT,
    FAMILY_IMPLEMENTATIONS_INDEPENDENT,
    GENERATOR_SCOPE,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TIME_STEPS,
    build_corpus,
    decode,
    manifest_sha256,
    prepare,
)
from ml.ocr.sequence_v3.model import CanonicalSlotRecognizer  # noqa: E402
from ml.ocr.sequence_v3.protocol import (  # noqa: E402
    CANDIDATES,
    ProtocolViolation,
    assert_execution_allowed,
    protocol_configuration,
    validate_exact_configuration,
)
from ml.ocr.sequence_v3.verify_source_binding import (  # noqa: E402
    REQUIRED_HISTORICAL_ARTIFACTS,
    verify,
    verify_historical_artifacts,
)


def test_corpus_is_deterministic_but_discloses_shared_generator() -> None:
    first = build_corpus(train_count=28, validation_count=28, test_count=28)
    second = build_corpus(train_count=28, validation_count=28, test_count=28)

    assert manifest_sha256(first) == manifest_sha256(second)
    assert GENERATOR_SCOPE == "shared-procedural-glyph-matrix-and-render-function"
    assert FAMILY_IMPLEMENTATIONS_INDEPENDENT is False
    assert CANONICALIZER_INVERTS_SHARED_GENERATOR_WIDTH_TRANSFORMS is True


def test_canonical_targets_decode_before_training() -> None:
    corpus = build_corpus(train_count=28, validation_count=28, test_count=28)
    for sample in corpus.all_samples():
        inputs, targets = prepare(sample)
        logits = torch.full((1, TIME_STEPS, CLASS_COUNT), -10.0)
        logits[0, torch.arange(TIME_STEPS), targets] = 10.0

        assert inputs.shape == (1, INPUT_HEIGHT, INPUT_WIDTH)
        assert decode(logits) == [sample.target_text]


def test_model_preserves_runtime_tensor_contract() -> None:
    torch.manual_seed(20260804)
    model = CanonicalSlotRecognizer().eval()
    output = model(torch.zeros((3, 1, INPUT_HEIGHT, INPUT_WIDTH)))

    assert output.shape == (3, TIME_STEPS, CLASS_COUNT)
    assert len(decode(output)) == 3


def test_protocol_rejects_every_configuration_override() -> None:
    exact = protocol_configuration()
    validate_exact_configuration(exact)
    for key, value in exact.items():
        changed = dict(exact)
        changed[key] = "changed" if not isinstance(value, int | float) else value + 1
        with pytest.raises(ProtocolViolation, match="does not exactly match"):
            validate_exact_configuration(changed)


def test_exhausted_budget_rejects_registered_and_fourth_candidates() -> None:
    exact = protocol_configuration()
    for candidate in CANDIDATES:
        with pytest.raises(ProtocolViolation, match="budget is exhausted"):
            assert_execution_allowed(candidate.candidate_id, exact)
    with pytest.raises(ProtocolViolation, match="Unregistered"):
        assert_execution_allowed("candidate-d", exact)


def test_tracked_source_binding_matches_executable_sources() -> None:
    binding = verify()

    assert binding["scientific_status"] == "failed_historical_research_only"
    assert binding["sealed_evidence_valid"] is False


def _historical_fixture(repository: Path) -> dict[str, object]:
    records = {}
    for role, relative in REQUIRED_HISTORICAL_ARTIFACTS.items():
        artifact = repository / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        payload = f"artifact:{role}".encode("utf-8")
        artifact.write_bytes(payload)
        records[role] = {
            "path": relative,
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    return {"historical_artifacts": records}


def test_historical_artifact_verifier_rejects_mutation(tmp_path: Path) -> None:
    binding = _historical_fixture(tmp_path)
    verify_historical_artifacts(binding, tmp_path)
    relative = REQUIRED_HISTORICAL_ARTIFACTS["candidate_a_report"]
    artifact = tmp_path / relative
    original = artifact.read_bytes()
    artifact.write_bytes(b"X" * len(original))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_historical_artifacts(binding, tmp_path)


def test_historical_artifact_verifier_rejects_missing_file(tmp_path: Path) -> None:
    binding = _historical_fixture(tmp_path)
    relative = REQUIRED_HISTORICAL_ARTIFACTS["representative_parity_report"]
    (tmp_path / relative).unlink()

    with pytest.raises(ValueError, match="is missing"):
        verify_historical_artifacts(binding, tmp_path)
