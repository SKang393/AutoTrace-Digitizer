<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# GraphSR-x2 candidate provenance and release audit

Audit date: 2026-08-03

## Decision

The tracked material defines an original local candidate and its evaluation
contract. An ignored local 100-step public-synthetic checkpoint, ONNX export,
and held-out benchmark exist. No weights, generated pairs, private images, or
benchmark reports are tracked. The candidate failed its scientific fidelity
gate, is not release-eligible, and is not the default enhancer.

The manifest records the exact rejected candidate ONNX SHA-256. This identity
does not authorize release storage.

## Required local audit fields

| Dependency/model | Version | Source | License | Bundled or downloaded | Notice path | Checksum | Review status |
|---|---|---|---|---|---|---|---|
| `graphsr-x2-candidate` | `0.1.0-local-candidate` | Original `ml/graphsr` project source | Apache-2.0 | No model bundled; ignored local rejected candidate only | `GRAPHSR_X2_CANDIDATE_NOTICE.md` | ONNX `4b0237683cd61ecd639015380bad9323a5fe79b295ffebf0c93720f51ef0d667` | Source, notice, exact artifact, CPU parity, and held-out failure reviewed; non-CPU providers, revision binding, and storage review blocked |

## Architecture and tensor contract

The candidate is the original `graphsr-srvgg-x2-v1` exact x2 network. It uses
24 feature channels, four two-convolution residual blocks, a bilinear geometry
anchor, and a PixelShuffle correction head. It consumes finite float32 RGB NCHW
`image` tensors in `[0,1]` and produces finite float32 RGB NCHW `enhanced`
tensors with exactly twice the input height and width. Loss weights are
Charbonnier reconstruction `1.0`, edge `0.20`, marker-center `0.15`, OCR proxy
`0.10`, and adversarial `0.0`. Configuration caps adversarial weight at `0.01`.

An enhanced output is always a derivative. Its coordinates map back through a
0.5 scale and then through every earlier recorded inverse transform. The
immutable original remains the scientific source of record.

## Training data and privacy

- Source: deterministic procedural Graph Auto Reader scenes only.
- Privacy: public synthetic structure; no article or participant data.
- External weights: none.
- Network access: none.
- Generated data eligibility: local and ignored, not Git eligible.
- Source, manifest, notices, and tests: Git eligible.
- Checkpoints, ONNX files, reports, and caches: ignored and not Git eligible.

The degradation contract records explicit seeds, actual execution order and
parameters, HR/LR checksums, crop geometry, forward and inverse coordinate
matrices, clipping loss, and marker centers. A geometry-preparation phase
applies skew, perspective, and jitter to both aligned pair targets. Stage 1 is
resize, blur, Gaussian noise, and JPEG. Stage 2 is ringing, paper, halftone,
fade, dark-ink erosion or dilation, bleed, and clipping, followed by exact
Lanczos x2 downsampling. Every operation is recorded even when skipped and uses
a local SHA-256-derived NumPy RNG without global RNG state.

## Dependency review

`ml/graphsr/DEPENDENCY_PROVENANCE.csv` records eight checksum-pinned, permissive
Python dependencies. Existing license and notice artifacts were reused from
the reviewed repository copies under `ml/markers/center/LICENSES`; the psutil
license was copied exactly from its installed distribution and tied to the
downloaded wheel checksum. No package wheel, native library, Python
environment, or training runtime is bundled in the Windows application by this
session.

PyTorch and NumPy are local training tools. Pillow performs local procedural
degradation. ONNX exports the candidate. ONNX Runtime performs optional local
parity and benchmark inference. pytest and jsonschema are test-only. Their
licenses do not replace the Apache-2.0 license of original Graph Auto Reader
work.

## Benchmark contract and status

The frozen report schema compares, in order:

1. `RealESRGAN_x2plus`;
2. `realesr-general-x4v3` at configured output scale 2;
3. `realesr-animevideov3` NCNN scale 2;
4. `GraphSR-x2`;
5. bicubic x2 baseline.

It records actual numeric OCR exact match, a separately labeled image-only OCR
proxy, marker F1 and center error, downstream shape/fill F1, thin-axis recall
and localization error, open-marker preservation, hallucination rate, runtime,
and peak memory. Results may be measured, blocked, or unmeasured. Missing
metrics are null with a reason. The proxy cannot satisfy the OCR selection gate.
Valid measured or truthful incomplete reports exit `0`; invalid manifests or
output contracts exit `2`.

The downstream benchmark ran on two held-out hand-drawn synthetic scenes.
GraphSR failed the marker-center gate at `0.953728` pixels against `0.25`, the
thin-line gate at `0.845171` against `0.98`, and the open-marker gate at
`0.985294` against `1.0`. CPU inference averaged `297.776` ms with a measured
peak of `305905664` bytes. ONNX parity passed at `2.384186e-07` against
`1e-05`. No default can be selected unless all candidates are measured, absolute
gates pass, GraphSR has no downstream regression against bicubic, and it
improves both actual numeric OCR accuracy and marker-center F1.

Candidate timing covers input decode through RGB output materialization. Common
shape validation and metric scoring are excluded. Peak memory is the maximum
combined resident working set of the benchmark host and candidate child
processes sampled every 1 ms over that same boundary.

## Release blockers

1. The candidate failed marker-center, thin-line, and open-marker thresholds.
2. Official candidate runtimes are not approved or locally configured.
3. Actual numeric OCR, marker detector, and shape/fill classifier adapters are unavailable.
4. The final source revision and source-bundle identity are not bound.
5. DirectML and CUDA execution have not been verified.
6. Reviewed release storage has not been assigned.
7. Installer and portable discovery, checksum, provider fallback, and byte
    parity have not been verified.

## Release-storage rule

Weights remain outside Git. A qualifying ONNX artifact may be published only
through reviewed release storage after license, checksum, benchmark, and
provider review. The installer and portable ZIP must stage the same artifact,
manifest, and notice from one commit and verify the same SHA-256.
