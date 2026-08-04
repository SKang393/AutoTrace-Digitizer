# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import pytest

torch = pytest.importorskip("torch")

from ml.ocr.ctc_candidate import (  # noqa: E402
    CLASS_COUNT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TIME_STEPS,
    CompactGraphNumericCtc,
    build_ctc_corpus,
    corpus_manifest_sha256,
    decode_logits,
    encode_target,
    prepare_input,
)


def test_candidate_corpus_is_deterministic_and_family_disjoint() -> None:
    first = build_ctc_corpus(train_count=12, validation_count=12, test_count=12)
    second = build_ctc_corpus(train_count=12, validation_count=12, test_count=12)

    assert corpus_manifest_sha256(first) == corpus_manifest_sha256(second)
    for attribute in ("renderer", "font", "degradation"):
        families = [
            {getattr(sample.family, attribute) for sample in split}
            for split in (first.train, first.validation, first.test)
        ]
        assert families[0].isdisjoint(families[1])
        assert families[0].isdisjoint(families[2])
        assert families[1].isdisjoint(families[2])


def test_candidate_preprocessing_matches_runtime_shape_and_range() -> None:
    corpus = build_ctc_corpus(train_count=2, validation_count=2, test_count=2)

    for sample in corpus.all_samples():
        prepared = prepare_input(sample)
        assert prepared.shape == (1, INPUT_HEIGHT, INPUT_WIDTH)
        assert torch.isfinite(prepared).all()
        assert -1.0 <= float(prepared.min()) <= float(prepared.max()) <= 1.0


def test_candidate_output_and_ctc_decoder_contract() -> None:
    torch.manual_seed(20260803)
    model = CompactGraphNumericCtc().eval()
    output = model(torch.zeros((2, 1, INPUT_HEIGHT, INPUT_WIDTH)))

    assert output.shape == (2, TIME_STEPS, CLASS_COUNT)
    assert len(decode_logits(output)) == 2
    assert encode_target("-10.5%")
