<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Project graph-numeric recognizer V1 preregistration

## Failure mode and distinct architecture

The earlier graph-numeric experiments are exhausted. V1 used CTC loss over a
learned horizontal sequence. Spatial V2 used dense supervision at each original
horizontal position. Canonical-slot V3 isolated glyphs before slot-wise
classification and no longer has valid sealed evidence.

This experiment is a new defect class. It consumes the complete `32x128` crop
with a 2D convolutional encoder, flattens the whole feature field into one global
spatial bottleneck, and predicts eight semantic character positions plus a
numeric-versus-nonnumeric role. Fixed blank-separated positions are projected to
the runtime `[N,32,14]` tensor. It uses no CTC loss, no dense column alignment,
no deterministic glyph isolation, and no V3 topology inversion.

## Frozen data and split boundary

- Protocol: `graph-numeric-project-v1-20260804`.
- Seed: `20260804`.
- Data: project-owned procedural labels and exclusion shapes only.
- Prohibited: Chandler, private graphs, article data, external datasets,
  system/Pillow fonts, downloaded weights, and pretrained weights.
- Train: 4,096 numeric plus 512 exclusion crops from an explicit polyline
  renderer with stroke/speckle degradation.
- Validation: 512 numeric plus 128 exclusion crops from an independently
  defined bitmap renderer with fade/scanline degradation.
- Sealed test: 512 numeric plus 128 exclusion crops from an independently
  implemented seven-segment renderer with contrast/dropout degradation.
- The three renderer glyph definitions and degradation functions do not share
  a glyph table or renderer implementation.
- Label cases are integers, decimals, negatives, percentages, negative
  decimals, decimal percentages, O/zero-like forms, and l/one-like forms.
- Exclusions include open and filled circles, ticks/axes, dividers, brackets,
  arrows, legends, intersections, and filled squares.
- Exact raster and metadata fingerprints are fixed in `FROZEN_PROTOCOL.json`.

## Candidate and observation budget

The defect class permits at most three candidates. Only Candidate 1 is frozen
and registered now. Candidates 2 and 3 are reserved and cannot execute. Either
requires a validation-only defect report, exactly one declared change, an
updated frozen protocol and source binding, review, and a new commit before
training. Sealed-test results may not select or tune a later candidate.

Candidate 1 is fixed at 40 epochs, AdamW `0.001`, weight decay `0.0001`, batch
size 64, slot blank weight `0.25`, and role loss weight `0.35`. No early
stopping, parameter sweep, threshold sweep, or rerun is permitted.

## Fixed gates

- Validation exact match at least `0.90`.
- Sealed-test exact match at least `0.90`.
- Sealed-test CER at most `0.05`.
- Validation and sealed role accuracy at least `0.90`.
- Model-level marker-exclusion accuracy exactly `1.0`.
- Zero marker creation from recognized text is a separate downstream
  application-integration gate. This training report must not claim or infer
  that result.
- Dynamic CPU ONNX output `[N,32,14]` and role output `[N,2]`.
- Representative PyTorch-to-CPU-ONNX maximum absolute difference at most
  `1e-4`, with identical decoded predictions.

ONNX export is justified only after validation and sealed quality gates pass.
Failure produces no ONNX or manifest. Passing these public gates creates only a
candidate. It does not approve the model without private validation, DirectML,
production resolver discovery, the downstream no-marker-creation gate,
packaging parity, notices, and the release audit.

## Commit-before-training boundary

`verify_preregistration.py --require-committed` requires every bound source,
test, plan, protocol, audit record, and binding file to exist unchanged at
`HEAD` on `main`. The training entrypoint calls this guard before building a
dataset, creating an output directory, or initializing training. The canonical
ignored output is `ml/ocr/project_numeric_v1/runs/candidate-1`.

Preregistration status: frozen source only. No training, sealed evaluation,
checkpoint, ONNX, manifest, approval, or promotion has occurred.
