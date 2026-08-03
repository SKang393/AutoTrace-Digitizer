# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Reproducible held-out benchmark for the tiny numeric recognizer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from time import perf_counter
from typing import Protocol

from .metrics import RecognitionMetrics, evaluate_predictions
from .recognizer import ConstantBaseline, PrototypeRecognizer, TrainingSummary
from .synthetic import SyntheticLabelSample, build_corpus

ACCEPTANCE_EXACT_MATCH = 0.90


class _Recognizer(Protocol):
    def predict(self, raster: tuple[tuple[int, ...], ...], orientation_degrees: int = 0) -> str: ...


@dataclass(frozen=True)
class BenchmarkReport:
    seed: int
    threshold: float
    training: TrainingSummary
    baseline_validation: RecognitionMetrics
    baseline_test: RecognitionMetrics
    trained_validation: RecognitionMetrics
    trained_test: RecognitionMetrics
    acceptance_passed: bool
    elapsed_ms: float


def _score(recognizer: _Recognizer, samples: tuple[SyntheticLabelSample, ...]) -> RecognitionMetrics:
    return evaluate_predictions(
        (
            sample.target_text,
            recognizer.predict(sample.raster, sample.orientation_degrees),
        )
        for sample in samples
    )


def run_benchmark(
    seed: int = 20260802,
    threshold: float = ACCEPTANCE_EXACT_MATCH,
) -> BenchmarkReport:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    started = perf_counter()
    corpus = build_corpus(seed)
    baseline = ConstantBaseline()
    recognizer = PrototypeRecognizer()
    training = recognizer.fit(corpus.train)
    baseline_validation = _score(baseline, corpus.validation)
    baseline_test = _score(baseline, corpus.test)
    trained_validation = _score(recognizer, corpus.validation)
    trained_test = _score(recognizer, corpus.test)
    return BenchmarkReport(
        seed=seed,
        threshold=threshold,
        training=training,
        baseline_validation=baseline_validation,
        baseline_test=baseline_test,
        trained_validation=trained_validation,
        trained_test=trained_test,
        acceptance_passed=(
            trained_validation.exact_match >= threshold
            and trained_test.exact_match >= threshold
        ),
        elapsed_ms=(perf_counter() - started) * 1000,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--threshold", type=float, default=ACCEPTANCE_EXACT_MATCH)
    arguments = parser.parse_args()
    report = run_benchmark(arguments.seed, arguments.threshold)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.acceptance_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
