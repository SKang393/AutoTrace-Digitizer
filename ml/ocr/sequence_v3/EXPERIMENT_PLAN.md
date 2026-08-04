<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Graph-numeric canonical-slot V3 experiment record

## Architecture

V3 uses deterministic glyph isolation and canonical slot packing followed by a
compact slot-wise convolutional classifier. It uses independent slot cross
entropy, not CTC loss or V2 original-position dense alignment. Fixed blank
separators preserve the runtime `[N,32,14]` decoder contract.

## Frozen protocol

- Protocol ID: `graph-numeric-sequence-v3-20260804`.
- Seed: `20260804`.
- Counts: 2,048 train, 512 validation, 512 initially designated test.
- Training: 24 epochs, Adam learning rate `0.002`, batch size 64.
- Maximum candidates: A, B, and C only.
- Public procedural inputs only. Private graphs, Chandler, article data,
  external datasets, fonts, pretrained weights, and downloaded weights were
  prohibited.
- Gates: validation and untouched-test exact match at least `0.90`, untouched
  test CER at most `0.05`, representative CPU ONNX parity at most `1e-4`, and
  output `[N,32,14]` with dynamic batch.

`protocol.py` now rejects configuration changes, reruns, and unregistered or
fourth candidates before output creation. The V3 budget is exhausted.

## Scientific correction

Candidate A evaluated the designated test split and failed. That was the first
and only sealed observation. The same records were then evaluated after B and
C, so B and C results are repeatedly observed, nonsealed holdout research and
cannot satisfy an untouched-test gate. Historical wording that called those
later evaluations sealed was incorrect and is superseded by this record.

The split families are also not implementation-independent. All derive from
the same `_GLYPHS` matrices and `_render` function in `ml/ocr/synthetic.py`.
Candidate C explicitly reverses the shared generator's width transformations.
Different family labels therefore do not establish independent-renderer or
production generalization confidence.

## Historical results

| Measure | Candidate A | Candidate B | Candidate C |
|---|---:|---:|---:|
| Validation exact | 0.22265625 | 0.1640625 | 1.0 |
| Holdout exact | 0.828125, first sealed observation | 0.810546875, reused nonsealed | 0.890625, reused nonsealed |
| Holdout CER | 0.051551814834297736 | 0.0583903208837454 | 0.03156233561283535 |

Candidate A failed the sealed gates. Candidate B changed canonical topology
resampling after a recorded validation defect. Candidate C changed width
normalization after another validation defect, but its later holdout result is
not sealed evidence and cannot be promoted. No fourth candidate, repair,
threshold sweep, or training rerun is permitted.

Historical report parity values used an all-zero export-smoke tensor and were
not representative maxima. `evaluate_existing.py` re-evaluates the exact
Candidate C checkpoint and ONNX on all current validation and repeatedly
observed holdout inputs without training. Its report remains research-only and
does not restore sealed validity.

## Release decision

V3 remains failed historical research. No model manifest may be created and no
automatic OCR stage, bundle, or release may use its generated weights. A future
defect class requires a committed preregistration, independently implemented
frozen rendering boundary, validation-only selection, and exactly one untouched
test evaluation after final selection.
