<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Goal 19 graph-numeric sequence V2 experiment audit

Audit date: 2026-08-03

## Decision

Sequence V2 is **failed experimental work**. Neither of its two permitted runs
met the preregistered quality gates. No candidate manifest or model notice was
created, no OCR stage was enabled, and no weight is approved or release
eligible.

The exact repair ONNX passed PyTorch-to-CPU parity and the production .NET CPU
and DirectML provider probe. Those results establish tensor and provider
compatibility only.

## Distinct architecture and fixed corpus

V2 is not a continuation of the exhausted V1 CTC objective. It preserves the
horizontal feature map and trains every one of 32 output positions with dense
project-renderer glyph-span alignment. The ordinary runtime CTC decoder is used
only to collapse contiguous character logits; V2 uses no CTC loss.

- Seed: `20260804`.
- Split sizes: 1,024 train, 256 validation, 256 sealed test.
- Corpus manifest SHA-256:
  `5498a83415c11084061b2da8a736ae2b713864ff1572b8ae6aa836a837285bfa`.
- Renderer, vector-glyph, and degradation family names are mutually disjoint
  across all three splits.
- Source: project-owned 5 by 7 vector glyphs in `ml/ocr/synthetic.py`.
- No private graph, supplied example, article data, system or Pillow font,
  external dataset, pretrained weight, or downloaded model weight was read.

Fixed gates were validation and sealed-test exact match at least `0.90`,
sealed-test character error rate at most `0.05`, PyTorch-to-CPU ONNX parity at
most `1e-4`, output `[N,32,14]`, and CPU-to-DirectML difference at most `1e-4`.

## Two-run result

| Measure | Candidate A | Contrast repair | Gate |
|---|---:|---:|---:|
| Validation exact match | 0.0 | 0.19921875 | >= 0.90 |
| Validation character error rate | 1.0 | 0.8559322033898306 | reported |
| Sealed-test exact match | 0.046875 | 0.10546875 | >= 0.90 |
| Sealed-test character error rate | 2.0643356643356645 | 1.2013986013986013 | <= 0.05 |
| PyTorch versus CPU ONNX maximum difference | 1.430511474609375e-06 | 1.6689300537109375e-06 | <= 1e-4 |
| Training time, ms | 221037.81639994122 | 199790.6655999832 | reported |

Candidate A collapsed to blank on validation despite final training loss
`0.14866358553990722`. The sole preregistered repair added per-crop mean and
variance standardization inside the exported model. Seed, corpus, splits,
layers, objective, optimizer, learning rate, blank weight, epochs, decoder, and
gates were unchanged. The repair improved transfer but remained far below all
quality gates. The two-run budget is exhausted.

## Ignored local artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Candidate A report | 3,286 | `2830c95a9ae8d76d057e90300fb35246eaa308eb54af3cc4b6c9b5df89e8bf9c` |
| Candidate A checkpoint | 142,309 | `326045a2bad6c5b257d9ab6ad6838e3eee527a15eeab5e5792253dcd2049f39e` |
| Candidate A ONNX | 140,402 | `2c0805098da0ce9fc66eb47a5ad98abb83659404df9eeecea5150ace7e4c8bd0` |
| Repair report | 3,346 | `6942b1d11de9759fb4f13edee7b0392d4fb9d4b5666a18ecd97cd2259ee26960` |
| Repair checkpoint | 142,373 | `b9f39602835006a942df80718216661a586e27dc2f8cabf866d07144d52bd9af` |
| Repair ONNX | 140,961 | `e7f31d5065f92be6142cd1c17814364646e35d612d5bf0750ed3e403b3b08e3c` |

All artifacts are under ignored `ml/ocr/sequence_v2/runs/`; none is tracked or
eligible for Git, bundling, or release packaging.

## Exact provider probe

The exact repair ONNX executed through the production .NET ONNX Runtime adapter:

- Available providers: `DmlExecutionProvider`, `CPUExecutionProvider`.
- Input: `[2,1,32,128]`.
- Output: 896 floats, exactly `[2,32,14]`.
- CPU inference: `6.8874` ms.
- DirectML inference: `122.6954` ms.
- CPU-to-DirectML maximum absolute difference:
  `6.67572021484375e-06`.

The probe does not establish exclusive GPU node assignment or OCR quality.

## License, privacy, and Git eligibility

- Original training code, procedural data, and generated weights are Copyright
  2026 Sungwoo Kang and Apache-2.0.
- Training and export use the existing permissive, unbundled PyTorch, NumPy,
  ONNX, and ONNX Runtime tooling recorded by the OCR workstream.
- Training inputs are public procedural data only.
- New source, tests, plan, and this audit are Git-eligible.
- Generated corpora, reports, checkpoints, and ONNX files remain ignored and
  are not Git-eligible.

## Remaining blockers

1. A new separately preregistered architecture or data strategy must address
   held-out vector-glyph and degradation transfer without reusing this exhausted
   repair budget.
2. Any future model must meet maintainer-approved quality gates.
3. A manifest and model notice may be added only for a truthful candidate.
4. Private graph evaluation, packaged model discovery, offline CPU fallback,
   installer and portable parity, and complete release-audit evidence remain
   mandatory.
5. Text detection and general text recognition remain separate unresolved model
   stages.
