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

P1 executed exactly once from configuration SHA-256 `27a94be0...a5db3` and
runner-source-bundle SHA-256 `95fdb53b...f643`. It completed all 2,304 fixed
optimizer steps, preserved the frozen role parent exactly, and retained all
1,024 truth regions. Its single stored-byte selection still left five false
prohibited regions across 10 scenes at the best fixed threshold, so only 118
of 128 scenes were exact and no passing three-threshold window existed. CPU
ONNX parity was `1.1444091796875e-05`, above the fixed `1e-5` maximum.
Recognition exact was `0.97265625`, CER was `0.004640371229698376`, and role
accuracy was `0.9951171875`. Aggregate result, ignored report, checkpoint, and
ONNX SHA-256 values are `1c1a3041...e5338`, `d962cd13...ec32b`,
`29fe9349...6d89`, and `d6cb6910...c241`.

P2 then executed exactly once from only those aggregate P1 metrics. It loaded
the exact P1 checkpoint, froze the role parent and all crop/evidence feature
weights, and trained only the proposal head for three epochs and exactly 1,152
steps. The frozen parameter stream was identical before and after training.
At threshold `0.75`, P2 retained all 1,024 truths with zero misses or duplicates
and reduced the residual error to one prohibited false region, leaving 122 of
128 scenes exact. No required three-threshold zero-error window existed. The
recognition and role metrics remained unchanged, and the exact parent role
output was preserved, but CPU ONNX parity again measured
`1.1444091796875e-05`, above the fixed `1e-5` maximum. Aggregate result, ignored
report, checkpoint, and ONNX SHA-256 values are `cce83aa9...e23fe`,
`eaf87de8...db4df`, `ecd1cebc...a0271`, and `04eea8a6...97ce7`.

P1 and P2 are consumed and cannot rerun. Final P3 is preregistered using only
the aggregate P2 terminal metrics. It loads the exact P2 checkpoint, freezes
the role parent, crop and evidence encoders, and the first two proposal layers,
then trains only `proposal_head.5.weight` and `proposal_head.5.bias`. Its fixed
three-epoch, 1,152-step objective combines positive and negative extrema,
the hardest two percent of training negatives, target margins, and per-scene
separation. No P2 case identity, truth row, prediction, or selection pixel was
used to choose that repair.

P2's parity excess was isolated on generated tensors to extended ONNX Runtime
graph optimization in frozen role channels, not proposal output. P3 therefore
binds candidate inference to deterministic CPU `ORT_ENABLE_BASIC`, which kept
generated-tensor maximum error within the existing `1e-5` gate. The tolerance
is unchanged, P2 is not rerun, and the policy must remain provider-compatible
if this candidate later reaches production consideration.

P3 may execute exactly once after its committed source binding. The
truth-hidden public archive remains unauthorized, unopened, and at zero
evaluations. No manifest, model-store entry, package payload, public
evaluation, marker composition, private validation, approval, or release has
been created.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
