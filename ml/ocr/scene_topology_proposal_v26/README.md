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
families, and degradation families. Their bytes do not exist yet. The one-time
freeze may run only from committed source and will reject any existing archive,
duplicate fixture bytes, or cross-split byte overlap.

P1 is preregistered for six epochs and exactly 2,304 optimizer steps. It must
execute every frozen training proposal, perform exactly one stored-byte visible
selection, preserve every role logit, pass CPU ONNX parity at `1e-5`, and pass
three consecutive thresholds with zero false, missed, duplicate, or prohibited
regions. Recognition exact must be at least `0.90`, CER at most `0.05`, overall
role accuracy at least `0.90`, and every role at least `0.85`.

This checkpoint authorizes neither split freezing nor training. No fixture,
checkpoint, ONNX file, manifest, model-store entry, package payload, public
evaluation, marker composition, private validation, approval, or release has
been created. A later committed checkpoint must bind the frozen archives and a
single candidate configuration before P1 can execute.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
