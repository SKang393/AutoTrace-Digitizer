<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Graph numeric OCR training scaffold

This directory contains a deterministic, standard-library-only starting point
for a tiny graph-label recognizer. It generates labels in memory and does not
write datasets, private graph content, checkpoints, or model weights.

The fixed corpus partitions train, validation, and test samples by distinct
renderer, font, and degradation families. The held-out test set covers tiny and
faded digits, `O`/`0`, `l`/`1`, decimals, percentages, negatives, and labels at
90 and 270 degrees. Numeric-role aliases are rendered as ambiguous source text
but scored against their canonical numeric value.

The recognizer is an intentionally small nearest-centroid character model. It
is a transparent training and evaluation scaffold, not a production OCR model.
The constant-`0` result is reported as the honest untrained baseline. The
initial acceptance threshold is 0.90 exact match on both held-out validation
and test families; character error rate is reported alongside exact match.

Run the fixed benchmark and tests from the repository root:

```powershell
python -m ml.ocr.benchmark --seed 20260802 --threshold 0.90
python -m pytest ml/ocr/tests -q
```

The command computes results at runtime. No benchmark number is predeclared or
copied into source. A future production model must be evaluated independently,
exported through the approved model-manifest process, and license-reviewed
before any weights can enter a Windows distribution.
