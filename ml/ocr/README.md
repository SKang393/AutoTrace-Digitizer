<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Graph numeric OCR training scaffold

This directory contains two deterministic graph-label recognizer paths. The
standard-library nearest-centroid benchmark remains a transparent baseline.
The Goal 19 CTC path uses the repository's existing permissive local training
toolchain to export a real ONNX matching the C# runtime contract. Generated
datasets, reports, checkpoints, and ONNX files remain ignored.

The fixed corpus partitions train, validation, and test samples by distinct
renderer, font, and degradation families. The held-out test set covers tiny and
faded digits, `O`/`0`, `l`/`1`, decimals, percentages, negatives, and labels at
90 and 270 degrees. Numeric-role aliases are rendered as ambiguous source text
but scored against their canonical numeric value.

The nearest-centroid recognizer is a transparent training and evaluation
scaffold, not a production OCR model.
The constant-`0` result is reported as the honest untrained baseline. The
initial acceptance threshold is 0.90 exact match on both held-out validation
and test families; character error rate is reported alongside exact match.

Run the fixed benchmark and tests from the repository root:

```powershell
python -m ml.ocr.benchmark --seed 20260802 --threshold 0.90
python -m pytest ml/ocr/tests -q
```

The preregistered Goal 19 CTC experiment is documented in
`CTC_EXPERIMENT_PLAN.md`. Its two-run budget is exhausted and both runs failed
the held-out quality gate. The command remains available for reproducibility:

```powershell
python -m pip install -r ml/ocr/requirements-test.txt
python -m ml.ocr.train_ctc --output ml/ocr/runs/goal19-repair-epochs96 --seed 20260803 --epochs 96 --learning-rate 0.003
```

This pipeline uses only the 5 by 7 project-owned vector glyph definitions in
`synthetic.py`. It does not read Pillow fonts, Windows fonts, external font
binaries, pretrained weights, downloaded models, private graphs, or external
datasets. Exact dependencies and license evidence are in
`DEPENDENCY_PROVENANCE.csv`.

The command computes results at runtime. The failed Goal 19 evidence is recorded
in `models/manifest/ocr/GRAPH_NUMERIC_CTC_EXPERIMENT_AUDIT.md`. A future
production model must use a new preregistered experiment, be evaluated
independently, pass the approved model-manifest process, and complete license
review before any weight can enter a Windows distribution.
