<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Graph-numeric sequence V2 experiment plan

## Distinct defect class

V1 optimized an unaligned CTC objective after compressing the complete crop to
a learned sequence. It fit the training family but failed to transfer character
alignment across held-out renderer, vector-glyph, and degradation families.

V2 is a different architecture and objective. It preserves the full horizontal
feature map and uses project-renderer glyph-span alignment as dense supervision.
Each of 32 horizontal output positions is explicitly trained as blank or one of
`0123456789.-%`. The ordinary runtime CTC decoder is retained only as the final
collapse rule for contiguous per-glyph logits. V2 does not use CTC loss.

## Fixed protocol

- Seed: `20260804`.
- Input: normalized grayscale `[N,1,32,128]` matching the C# crop contract.
- Output: batch-major logits `[N,32,14]`.
- Alphabet: blank plus `0123456789.-%`.
- Fixed corpus: 1,024 train, 256 validation, 256 sealed test samples.
- Split boundary: renderer, vector-glyph, and degradation family names are
  mutually disjoint across train, validation, and test.
- Data source: project-owned procedural 5 by 7 glyph vectors only.
- Prohibited inputs: private graphs, supplied examples, external datasets,
  system or Pillow fonts, pretrained weights, and downloaded model weights.
- Candidate A: spatial sequence network, 36 epochs, Adam learning rate `0.002`,
  blank loss weight `0.20`.
- Repair budget: at most one rerun after recording the observed defect and one
  changed factor. No other architecture, learning-rate, threshold, data, or
  test-set sweep is allowed.
- Candidate selection uses validation only. The sealed test split is evaluated
  once for Candidate A and once only if the declared repair is required.

## Fixed gates

- Validation exact match at least `0.90`.
- Sealed-test exact match at least `0.90`.
- Sealed-test character error rate at most `0.05`.
- CPU ONNX versus PyTorch maximum absolute difference at most `1e-4`.
- Exact ONNX shape `[N,32,14]` with dynamic batch.
- Exact model must execute through CPU and DirectML with maximum provider
  difference at most `1e-4`.

Passing creates only a candidate. It cannot be approved or release eligible
without maintainer-approved gates, private evaluation, packaging discovery,
offline checks, notices, and the complete public release audit.

## Candidate A result and repair declaration

Candidate A failed with validation exact match `0.0`, sealed-test exact match
`0.046875`, and sealed-test character error rate `2.0643356643356645`.
Validation-only diagnostics showed every sampled prediction collapsing to the
blank class. The training loss reached `0.14866358553990722`, so this is not an
undertraining defect. It is an input-contrast domain defect: training uses the
held-in clean-print intensity family while validation and test use separately
held-out faded and adverse intensity families.

The single permitted repair adds per-crop mean and variance standardization
inside the exported model. This is the only changed factor. Seed, corpus,
splits, architecture layers, dense alignment objective, optimizer, learning
rate, blank weight, epoch count, decoder, gates, and all other inputs remain
unchanged. The repair consumes the second and final sealed-test evaluation.
