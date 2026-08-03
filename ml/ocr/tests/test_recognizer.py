# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import pytest

from ml.ocr.recognizer import PrototypeRecognizer
from ml.ocr.synthetic import build_corpus


def test_recognizer_requires_training() -> None:
    sample = build_corpus().test[0]

    with pytest.raises(RuntimeError, match="fitted"):
        PrototypeRecognizer().predict(sample.raster)


def test_training_builds_every_numeric_character_prototype() -> None:
    corpus = build_corpus()
    recognizer = PrototypeRecognizer()

    summary = recognizer.fit(corpus.train)

    assert summary.labels_skipped == 0
    assert set(summary.prototype_counts) == set("0123456789.-%")
    assert all(count > 0 for count in summary.prototype_counts.values())


def test_orientation_metadata_restores_rotated_labels() -> None:
    corpus = build_corpus()
    recognizer = PrototypeRecognizer()
    recognizer.fit(corpus.train)
    rotated = [sample for sample in corpus.test if sample.case == "rotated_label"]

    assert [
        recognizer.predict(sample.raster, sample.orientation_degrees)
        for sample in rotated
    ] == [sample.target_text for sample in rotated]
