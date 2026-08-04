# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import pytest

torch = pytest.importorskip("torch")

from ml.ocr.sequence_v2.dataset import (  # noqa: E402
    CLASS_COUNT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TIME_STEPS,
    build_corpus,
    decode,
    manifest_sha256,
    prepare,
)
from ml.ocr.sequence_v2.model import SpatialAlignedSequenceModel  # noqa: E402


def test_corpus_is_deterministic_and_families_are_held_out() -> None:
    first = build_corpus(train_count=24, validation_count=24, test_count=24)
    second = build_corpus(train_count=24, validation_count=24, test_count=24)

    assert manifest_sha256(first) == manifest_sha256(second)
    for attribute in ("renderer", "font", "degradation"):
        families = [
            {getattr(sample.family, attribute) for sample in split}
            for split in (first.train, first.validation, first.test)
        ]
        assert families[0].isdisjoint(families[1])
        assert families[0].isdisjoint(families[2])
        assert families[1].isdisjoint(families[2])


def test_dense_alignment_decodes_to_the_reference_before_training() -> None:
    corpus = build_corpus(train_count=24, validation_count=24, test_count=24)
    for sample in corpus.all_samples():
        inputs, aligned = prepare(sample)
        logits = torch.full((1, TIME_STEPS, CLASS_COUNT), -10.0)
        logits[0, torch.arange(TIME_STEPS), aligned] = 10.0

        assert inputs.shape == (1, INPUT_HEIGHT, INPUT_WIDTH)
        assert decode(logits) == [sample.target_text]


def test_model_preserves_runtime_tensor_contract() -> None:
    torch.manual_seed(20260804)
    model = SpatialAlignedSequenceModel().eval()
    output = model(torch.zeros((3, 1, INPUT_HEIGHT, INPUT_WIDTH)))

    assert output.shape == (3, TIME_STEPS, CLASS_COUNT)
    assert len(decode(output)) == 3
