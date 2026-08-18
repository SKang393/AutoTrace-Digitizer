<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V26 scene-topology proposal candidate

V26 is a fresh project-owned OCR proposal defect class. Its design uses only
the aggregate terminal V25 P3 metrics. No V25 case identity, truth row,
prediction, fixture pixel, private image, Chandler image, or `Generalization`
label informed the design.

V25 P3 passed recognition, role, and CPU ONNX parity thresholds, but retained
1,017 of 1,024 selection truths with one prohibited false region and seven
misses at every fixed threshold. V26 therefore replaces the exhausted residual
proposal path. It trains a proposal head from scratch over all production
proposals while freezing the exact V24 role parent. The proposal head combines
generic 31-value evidence, global and axial tight/context crop projections, and
per-scene evidence aggregates. The parent role logits are copied unchanged.

The train, visible-selection, and truth-hidden public registrations contain
384, 128, and 192 new procedural scenes with disjoint seed offsets, renderer
families, and degradation families. They were frozen once from clean commit
`03aa8610f009e7a1f96fa52130b38ca2ea0d1b25`. The archive SHA-256 values are
`769b1f1b...a04280`, `5eccc44f...0fc49`, and `84e00f59...1981`; the split-seal
SHA-256 is `9bdcd0da...67b0`. Every split has one production proposal per truth,
no duplicate fixture bytes, and zero cross-split byte overlap.

P1 is preregistered for six epochs and exactly 2,304 optimizer steps. It must
execute every frozen training proposal, perform exactly one stored-byte visible
selection, preserve every role logit, pass CPU ONNX parity at `1e-5`, and pass
three consecutive thresholds with zero false, missed, duplicate, or prohibited
regions. Recognition exact must be at least `0.90`, CER at most `0.05`, overall
role accuracy at least `0.90`, and every role at least `0.85`.

This checkpoint freezes only the split identity and public evaluator. The
truth-hidden public archive remains unopened with zero evaluations. No
candidate is configured or authorized, and no optimizer step, selection,
checkpoint, ONNX file, manifest, model-store entry, package payload, public
evaluation, marker composition, private validation, approval, or release has
been created. A later committed checkpoint must bind one P1 configuration
before any candidate execution can begin.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
