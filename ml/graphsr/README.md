<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# GraphSR-x2 training toolchain

This directory contains the local, synthetic-only training and evaluation
toolchain for an original chart-preserving 2x restoration candidate. It does
not contain trained weights, downloaded models, private article data, or human
annotations.

## Status

GraphSR-x2 is a local candidate, not a production model and not the default
enhancer. The single declared 100-step public-synthetic experiment completed,
including ONNX CPU parity and two held-out hand-drawn scenes. The candidate
failed the marker-center, thin-line, and open-marker gates and is not
release-eligible.

## Experiment contract

The failure mode under test is restoration that looks cleaner while moving,
closing, thinning, or inventing scientific structures. The responsible
subsystem is the GraphSR degradation, training, export, and benchmark pipeline.

The fixed public test set is deterministic procedural Graph Auto Reader data.
Private studies and published article images are prohibited from training and
are not read by default. The acceptance metrics are:

- actual numeric OCR exact match and a separately labeled image-only proxy;
- detector-derived marker-center F1 when an approved detector is available,
  local mean center error in original pixels, and downstream shape/fill F1;
- thin-axis recall and axis localization error;
- open-marker preservation;
- hallucinated-structure rate;
- mean runtime and peak memory.

The experiment budget is one declared training configuration and one final
held-out benchmark report. Parameter sweeps require a written comparison and
may not repeatedly open the held-out split.

## Data and degradations

Training pairs originate from deterministic procedural scenes under
`ml/synthetic`. Clean high-resolution crops remain immutable. Low-resolution
inputs are produced by one or two recorded degradation stages. Each pair
records its seed, source identity, degradation order and parameters, crop
geometry, and the reversible mapping back to the clean source.

One recorded geometry-preparation phase applies skew, perspective, and jitter
to both the clean target and the future degraded input so the pair stays
aligned. Stage 1 then applies resize, blur, Gaussian noise, and JPEG. Stage 2
applies ringing, paper texture, halftone, fade, dark-ink erosion or dilation,
bleed, and clipping, then exact Lanczos x2 downsampling. All 15 operations are
recorded even when skipped. Each operation derives a local NumPy RNG from the
explicit root seed without touching global RNG state. No degradation may
silently change the target marker-center coordinates.

## Candidate and loss contract

`GraphSRx2` is the original `graphsr-srvgg-x2-v1` architecture. It uses 24
feature channels, four two-convolution residual blocks, a bilinear x2 geometry
anchor, and a learned bounded correction projected through PixelShuffle. The
deterministic initialization seed is `20260803`.

Training weights Charbonnier reconstruction `1.0`, edge consistency `0.20`,
marker-center consistency `0.15`, and OCR-proxy consistency `0.10`. Adversarial
loss is disabled by default. Configuration rejects an adversarial weight above
`0.01`, and the current objective contains no hidden discriminator term.

The ONNX input `image` is float32 RGB NCHW in `[0,1]`; output `enhanced` is an
exact x2 float32 RGB NCHW result in `[0,1]`. The enhanced image is a derivative. It never overwrites
the original, and enhanced pixel coordinates map back by dividing x and y by
two before applying earlier inverse transforms.

## Commands

```powershell
python -m pytest ml/graphsr/tests -q
python -m ml.graphsr.train --help
python -m ml.graphsr.export --help
python -m ml.graphsr.experiment --output ml/graphsr/runs/session07-final-v2 --steps 100
python -m ml.graphsr.benchmark --manifest <local-manifest> --output <local-report.json>
```

The benchmark compares, in frozen order:

1. `RealESRGAN_x2plus`
2. `realesr-general-x4v3` configured for output scale 2
3. `realesr-animevideov3` NCNN scale 2
4. `GraphSR-x2`
5. bicubic x2 baseline

Candidate results are `measured`, `blocked`, or `unmeasured`. Missing metrics
remain null and carry a reason while independently available structural and
runtime observations remain visible. A default is forbidden unless every candidate
is measured, all absolute downstream gates pass, no downstream metric regresses
against bicubic, and the selected candidate improves both actual numeric OCR
exact match and marker-center F1. The image-only OCR proxy is diagnostic and
cannot satisfy the OCR selection gate.

A valid measured or truthful incomplete report exits `0`. An invalid manifest
or output contract exits `2`.

Runtime and memory use one common boundary for every candidate: local input
decode through RGB `uint8` output materialization. Common shape validation and
quality scoring are excluded. Peak memory is the maximum combined resident
working set of the benchmark host and candidate child processes sampled every
1 ms over that boundary.

## Dependencies and release storage

The checksum-pinned local training dependencies and copied notices are listed
in `DEPENDENCY_PROVENANCE.csv` and `THIRD_PARTY_NOTICES.md`. They are permissive
unbundled tooling and are not included in the Windows application by this
session.

Checkpoints, ONNX files, generated pairs, reports, and caches remain ignored.
A reviewed model may be published only through reviewed release storage, then
referenced by a schema-valid manifest containing the exact artifact checksum.
