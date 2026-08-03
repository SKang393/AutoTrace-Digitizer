# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from ml.ocr.synthetic import build_corpus


def test_corpus_is_deterministic_for_a_fixed_seed() -> None:
    first = build_corpus(1234)
    second = build_corpus(1234)

    assert first == second
    assert build_corpus(1235).test != first.test


def test_renderer_font_and_degradation_families_do_not_leak() -> None:
    corpus = build_corpus()
    split_samples = (corpus.train, corpus.validation, corpus.test)

    for attribute in ("renderer", "font", "degradation"):
        families = [
            {getattr(sample.family, attribute) for sample in samples}
            for samples in split_samples
        ]
        assert families[0].isdisjoint(families[1])
        assert families[0].isdisjoint(families[2])
        assert families[1].isdisjoint(families[2])


def test_held_out_set_contains_every_required_numeric_case() -> None:
    corpus = build_corpus()
    cases = {sample.case for sample in corpus.test}

    assert {
        "tiny_digits",
        "faded_digits",
        "o_zero_ambiguity",
        "l_one_ambiguity",
        "decimal",
        "percent",
        "negative",
        "rotated_label",
    } <= cases
    assert {sample.orientation_degrees for sample in corpus.test} >= {0, 90, 270}


def test_generator_keeps_samples_in_memory() -> None:
    corpus = build_corpus()

    for sample in corpus.all_samples():
        assert sample.raster
        assert all(0 <= pixel <= 255 for row in sample.raster for pixel in row)
        assert sample.sample_id.startswith(f"{sample.split}-")
