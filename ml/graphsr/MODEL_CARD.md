<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# GraphSR-x2 candidate model card

## Model details

| Field | Value |
|---|---|
| Model ID | `graphsr-x2-candidate` |
| Version | `0.1.0-local-candidate` |
| Task | Chart-preserving super resolution at exact 2x scale |
| Architecture | `graphsr-srvgg-x2-v1`: 24 channels, four residual blocks, bilinear anchor, PixelShuffle x2 correction |
| License | Apache-2.0 for original project source and any qualifying original trained artifact |
| Status | Candidate only; not release-eligible; not selected as default |
| Network access | None |

A deterministic ignored 100-step checkpoint and ONNX export exist from 16
public-synthetic training crops. The two-case held-out benchmark rejected the
candidate. The manifest records the exact artifact, parity, and benchmark
evidence without authorizing release or default selection.

## Intended use

GraphSR-x2 is intended to create a local, derived x2 image that may improve
downstream OCR and graph-structure detection for degraded single-case design
graphs. The immutable original remains the scientific source of record.

It is not intended to recover unknowable source detail, infer scientific
values, replace manual review, or enhance photographs and natural images.

## Inputs and outputs

- Input `image`: float32 RGB NCHW `[N,3,H,W]`, finite values in `[0,1]`.
- Output `enhanced`: float32 RGB NCHW `[N,3,2H,2W]`, clamped to `[0,1]`.
- Input coordinate space: original or explicitly transformed panel pixels.
- Output coordinate space: enhanced pixels.
- Inverse scale: `original_x = enhanced_x / 2` and
  `original_y = enhanced_y / 2`, followed by the recorded earlier inverse
  transforms.

## Training data and privacy

Only deterministic procedural Graph Auto Reader scenes are eligible for
training. They contain no participant records, private research data,
published figures, human annotations, or third-party pretrained weights.
Generated datasets and weights remain local and ignored by Git.

Splits are separated by renderer, degradation recipe, chart template, font,
and marker-style families. A real-data generalization claim requires a later,
permission-cleared article-level evaluation and is not made here.

## Training objective

The declared objective uses Charbonnier reconstruction `1.0`, thin-edge
preservation `0.20`, marker-center consistency `0.15`, and OCR-proxy
consistency `0.10`. Adversarial loss is off by default and capped at `0.01` by
configuration. Model selection is downstream-first: perceptual appearance
alone cannot justify selection.

## Evaluation

The frozen benchmark compares three official Real-ESRGAN identities,
GraphSR-x2, and bicubic x2. It reports:

- actual numeric OCR exact match and a separately labeled image-only proxy;
- detector-derived marker-center F1 when available, local mean center error in
  pixels, and downstream shape/fill F1;
- thin-axis recall and axis localization error;
- open-marker preservation rate;
- hallucinated-structure rate;
- mean runtime and peak memory.

The downstream benchmark ran on two held-out hand-drawn scenes. GraphSR mean
marker-center error was `0.953728` pixels against a `0.25` maximum, thin-axis
recall was `0.845171` against a `0.98` minimum, and open-marker preservation
was `0.985294` against a required `1.0`. Hallucinated-structure rate passed at
`0.002459`. ONNX checker and CPU parity passed at `2.384186e-07` maximum
absolute error against `1e-05`. Official model runtimes, actual OCR, the marker
detector, and the shape/fill classifier were unavailable, so those fields
remain blocked or null and no default was selected.

Runtime and memory observations use the same input-decode-through-output-
materialization boundary for in-process and command candidates. Peak memory is
combined host and child resident working set sampled every 1 ms.

Valid measured and truthful incomplete reports exit `0`; invalid manifest or
output contracts exit `2`.

## Scientific-fidelity risks

Restoration can close open markers, move centers, thicken thin axes, create
false strokes, change tiny numerals, or hide compression evidence. The product
must map all derivative detections back to original pixels, reduce confidence
when original and enhanced evidence disagree, and send material disagreements
to review.

## Release gates

A later release review must establish all of the following:

1. exact ONNX and checkpoint hashes;
2. source revision, dataset-manifest hash, and deterministic seed;
3. permissive license and complete notices;
4. ONNX checker and numerical parity;
5. CPU fallback and packaged provider verification;
6. held-out downstream benchmark of all five candidates;
7. marker-center tolerance and open-marker acceptance;
8. no hallucination or downstream regression;
9. identical model bytes in installer and portable artifacts;
10. reviewed release-storage provenance.
