<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Goal 19 graph-numeric CTC experiment plan

## Failure mode

The repository has a transparent nearest-centroid numeric OCR scaffold, but it
cannot export the `[N,T,C]` logits required by `LocalOnnxTextRecognizer` and is
therefore not a production model candidate.

## Fixed protocol

- Seed: `20260803`.
- Input: normalized grayscale `[N,1,32,128]` crops matching the C# OCR batcher.
- Output: CTC logits `[N,32,14]` for blank plus `0123456789.-%`.
- Data: project-generated procedural labels only. No private graphs, external
  datasets, pretrained weights, or downloaded model weights.
- Split boundary: renderer, font, and degradation families are disjoint across
  train, validation, and test.
- Fixed sizes: 768 train, 192 validation, and 192 test samples.
- Candidate A: compact convolutional CTC recognizer, 24 epochs.
- Repair budget: one targeted rerun only if Candidate A fails. The defect and
  single changed parameter must be recorded before that rerun.
- Test-set budget: one evaluation for Candidate A and, only if needed, one for
  the targeted repair. No threshold or test-set sweep.

## Metrics and local candidate gates

- Validation exact match at least `0.90`.
- Held-out test exact match at least `0.90`.
- Held-out character error rate at most `0.05`.
- CPU ONNX output parity maximum absolute difference at most `1e-4`.
- ONNX output shape exactly `[N,32,14]` for dynamic batch `N`.

These are session-local candidate gates, not maintainer-approved release gates.
Passing them cannot approve the model without private-graph validation,
DirectML evidence, packaging discovery, and the complete release audit.

## Candidate A result and targeted repair declaration

Candidate A failed before any repair tuning: validation exact match was
`0.03125`, held-out exact match was `0.015625`, and held-out character error
rate was `0.8560747663551402`. Validation diagnostics showed every sampled
prediction collapsing to `1`. The epoch loss was still falling at epoch 24
(`2.1881163517634072`), so the defect is classified as an undertrained CTC
objective rather than a threshold defect.

The only permitted repair changes `epochs` from `24` to `96`. Seed, model,
optimizer, learning rate, corpus, split, preprocessing, decoder, and gates are
unchanged. The repair will consume the second and final test-set evaluation.
