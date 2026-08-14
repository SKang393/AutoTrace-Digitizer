<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Marker-center dense production contract V5

Normalized-training V4 passed its fixed proposal-level synthetic selection and
public gates, but its `[N,3,33,33]` input and `[N,4]` output cannot satisfy the
frozen production marker-center and artifact-mask adapters. Those adapters
require one checksum-resolved dense `[N,3,H,W]` input and activated
`[N,3,H,W]` center, radius, and artifact output.

V5 is a distinct contract-repair defect class. It uses new procedural renderer,
degradation, seed, scene, and byte families. No V4 fixture bytes, public case
detail, Chandler, `Generalization`, private/article data, external dataset, or
downloaded weight is used. Project-trained weights remain Apache-2.0.

P1 increases only the compact dense network capacity, trains all three frozen
heads together, and evaluates the same model twice: first to derive a dense
artifact mask from seed OCR, axis, tick, and divider masks, then to detect
markers with that artifact evidence. Selection requires every visible scene
exact, zero false positives, misses, duplicates, or prohibited hits, CPU ONNX
parity at most `1e-5`, and at least three adjacent passing thresholds.

The 96 training, 24 validation, and 32 truth-hidden public scenes are frozen
before optimization. The public archive may open once only after selection
passes and separate authorization is committed. Up to three candidates are
available. P1 is consumed and failed selection with 21 missed markers, six
prohibited hits, and CPU ONNX parity above the fixed tolerance. P2 and P3 are
not yet preregistered. The public gate remains locked and unopened. Production
approval, model-store
promotion, packaging, private validation, and release eligibility remain false.
