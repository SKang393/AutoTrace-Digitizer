<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Project numeric OCR Candidate 3 preregistration

## Decision boundary

Candidate 3 is the final candidate in the fixed three-candidate defect budget.
It is frozen but untrained and may run exactly once from committed `main` into
the ignored canonical `runs/candidate-3` directory. Candidates 1 and 2 are
consumed and rejected. No fourth candidate or rerun is authorized.

Candidate 2 selection evidence is validation-only. Its exact match was
`0.34375`, CER was `0.37801047120418846`, numeric-role accuracy was `0.821875`,
and marker exclusion was `1.0`. Candidate 2 failed validation, so Candidate 2
never decoded or scored the Candidate 2/3 sealed split.

## Exactly one changed factor

The only training change is the recognizer architecture. Candidate 2's single
global spatial bottleneck is replaced by a whole-crop 2D convolutional encoder,
ordered column tokens, one self-attention block, and eight learned semantic
queries. The queries attend to complete image evidence and predict semantic
character positions plus a numeric-role head.

This is not CTC, dense original-column supervision, deterministic glyph
isolation, or the exhausted canonical-slot V3 topology. It preserves the same
fixed `[N,32,14]` plus `[N,2]` runtime contract.

Unchanged factors are Candidate 2's renderer and degradation family, every
procedural raster and split fingerprint, label distribution, exclusion kinds,
optimizer, learning rate, weight decay, batch size, 40 epochs, objective, and
all validation, sealed, role, exclusion, CPU, and ONNX parity gates.

## Sealed evidence

The still-unopened Candidate 2 sealed split remains fixed at fingerprint
`a4e2e8c0623d77a52d88da9b997deebe9bb57245a65e25c918b6817892b39aee`.
Pretraining access remains limited to deterministic fingerprint verification.
Candidate 3 may decode or score this split only after it passes every validation
quality gate. A validation failure consumes the final candidate without opening
sealed metrics or predictions.

## Fixed gates and approval boundary

- Validation and sealed exact match at least `0.90`.
- Sealed CER at most `0.05`.
- Validation and sealed role accuracy at least `0.90`.
- Marker exclusion exactly `1.0`.
- CPU ONNX maximum absolute parity difference at most `1e-4`, identical decoded
  predictions, and dynamic `[N,32,14]` plus `[N,2]` outputs.

Passing creates only a public candidate. Production approval additionally
requires zero downstream marker creation from recognized text, private graph
validation, DirectML execution, production resolver discovery, notice and
checksum review, packaging parity, and the complete fail-closed release audit.

## Provenance and privacy

All inputs remain deterministic project-owned procedural labels and exclusion
shapes. No Chandler image, private graph, article data, external dataset, font,
downloaded weight, or pretrained weight is allowed. Source and any future
project-owned weight are Apache-2.0. Generated data, checkpoints, ONNX files,
and reports remain ignored and are not Git eligible.
