<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Project numeric OCR Candidate 2 preregistration

## Decision boundary

Candidate 2 is frozen but untrained. It may run exactly once, from the committed
`main` source bundle, into the ignored canonical `runs/candidate-2` directory.
Candidate 1 is consumed and remains rejected. Candidate 3 remains unregistered.

The selection record is validation-only. Candidate 1 validation exact match was
`0.3359375`, validation CER was `0.537696335078534`, numeric-role accuracy was
`0.5828125`, and marker exclusion was `1.0`. All eight validation label cases
failed the `0.90` exact-match gate. Candidate 2 selection does not use Candidate
1 sealed-test results.

## Exactly one changed factor

The only training change is the renderer and degradation family. The single
polyline training renderer is replaced by deterministic domain randomization of
the same project-owned procedural labels. It varies raster scale, translation,
pixel-cell quantization, ink thickness, local dropout, scanline attenuation,
and contrast without importing validation or sealed glyph definitions.

Unchanged factors are the whole-crop global-spatial-bottleneck architecture,
semantic-slot and numeric-role objectives, optimizer, learning rate, weight
decay, batch size, 40 epochs, train and validation counts, label distribution,
exclusion kinds, fixed validation split, output contract, and all public gates.

## Sealed evidence

A new Candidate 2 sealed split is frozen before training with an independently
authored outline-stencil renderer and separate degradation seed. Its raster and
metadata fingerprint is bound in `CANDIDATE_2_PROTOCOL.json`. Pretraining access
is limited to deterministic fingerprint sealing. Training code may score or
decode it only after every validation quality gate passes. A validation failure
consumes Candidate 2 without opening sealed metrics or predictions.

## Fixed gates

- Validation and sealed exact match at least `0.90`.
- Sealed CER at most `0.05`.
- Validation and sealed role accuracy at least `0.90`.
- Marker exclusion exactly `1.0`.
- CPU ONNX maximum absolute parity difference at most `1e-4` with identical
  decoded predictions and dynamic `[N,32,14]` plus `[N,2]` outputs.

Passing these gates creates only a public candidate. It does not approve the
model without downstream no-marker-creation evidence, private graph validation,
DirectML execution, production resolver discovery, notice and checksum review,
packaging parity, and the fail-closed release audit.

## Provenance and privacy

All inputs are deterministic project-owned procedural labels and exclusion
shapes. No Chandler image, private graph, article data, external dataset, font,
downloaded weight, or pretrained weight is allowed. Source and any future
project-owned weight are Apache-2.0. Generated data, checkpoints, ONNX files,
and reports remain ignored and are not Git eligible.
