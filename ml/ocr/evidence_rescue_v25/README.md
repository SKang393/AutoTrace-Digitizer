<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V25 evidence-rescue candidate

V25 is a fresh project-owned OCR proposal and role defect class. Its design
uses only aggregate terminal evidence from consumed V23 P3 and V24 P2. No
predecessor case identity, truth record, fixture pixel, private image,
Chandler image, or `Generalization` label informed the design.

V23 P3 retained all 1,024 visible-selection truths but also retained three
false prohibited regions. V24 P2 removed every false region but missed two
truths while preserving the V23 role outputs. V25 P1 therefore keeps the exact
frozen V24 P2 model, preserves its roles, retains every V24 P2 acceptance, and
rescues only a parent-accepted rejection whose already-bound CTC evidence
passes fixed generic confidence, entropy, blank, length, and alphanumeric
conditions.

P1 performs zero optimizer steps. The complete split was frozen once from
source commit `9805c2db397e2b7857093b3292cf115ddb6d559b`. The train,
validation, and truth-hidden public archives contain 256, 128, and 192 fresh
scenes and have SHA-256 values `63f9d63a...bcc1`, `20284a9a...f9d`, and
`bab5eaa6...c40`. Their source-byte overlap is zero. The train split remains
available only for a later separately preregistered candidate if P1 fails.

P1 configuration SHA-256 `4c96df20...bb6` and the public gate configuration
SHA-256 `fc687363...7f18` bind the exact split seal, ignored fixture archives,
detector, recognizer, frozen parent, candidate runner, and truth-hidden
evaluator. P1 executed exactly once for zero optimizer steps. Direct stored-byte
CPU selection retained 1,017/1,024 truths with one false prohibited region,
seven misses, zero duplicates, and 112/128 exact scenes at every fixed
threshold. Recognition exact was `0.96484375`, CER was
`0.010632995514205018`, and role accuracy was `0.9833984375`. CPU ONNX parity
passed at `9.5367431640625e-06`, but the same nonzero error meant the required
exact parent-role preservation was not proven. No threshold window passed.
Report SHA-256 is `c655ad88...48a5`; rejected ONNX SHA-256 is
`ff9d0351...5acc`; tracked result SHA-256 is `700e1b65...dcf1`.

P1 and P2 are consumed and cannot rerun. P2 was separately preregistered from
only the aggregate P1 result. It froze the exact V24 P2 parent and trained only
a small pooled tight/context crop plus 31-value evidence residual. Its fixed
loss simultaneously enforced positive and negative logit margins, teacher
preservation, and per-scene separation across all 16,520 frozen training
proposals. It completed all five epochs and exactly 1,280 optimizer steps.
Configuration SHA-256 is `e6bacf0f...1dba`; runner source bundle is
`a637e8c6...12aa`.

The single direct stored-byte CPU selection retained 1,016/1,024 truths with
one false prohibited region, eight misses, zero duplicates, and 112/128 exact
scenes at its best threshold `0.35`. Recognition exact was `0.9638671875`, CER
was `0.011463698288752284`, and role accuracy was `0.982421875`. Exact parent
roles were preserved and CPU ONNX parity passed at
`9.5367431640625e-06`, but no fixed threshold passed the zero-error gate and no
three-threshold window existed. Report, rejected ONNX, checkpoint, and tracked
result SHA-256 values are `057965da...768`, `f612fe00...552`,
`15beecac...520`, and `266df7f7...5c1`.

Final P3 is preregistered from only the aggregate P1 and P2 results. It freezes
the exact P1 proposal and role composition, preserves every parent role logit,
and replaces P2's fixed 8-by-8 crop pooling with a trainable two-scale spatial
encoder plus proposal residual. Fixed configuration SHA-256 is
`4bc77850...821a`; runner source bundle SHA-256 is `fbddfcce...63bd`. It has a
five-epoch, exactly 1,280-step budget with fixed weight decay, dropout, and
gradient clipping. It is not yet authorized to execute.

The public evaluator source bundle remains frozen at `82467dcf...fb15`, but
the public archive stays unauthorized and unopened with zero evaluations.
Marker composition, private validation, manifest creation, model store,
packaging, approval, and release remain unauthorized.

The candidate is not reachable from ordinary Auto Detect. It cannot become a
manifest, production-model-store entry, package payload, approved model, or
release input without direct selection, one-time public, marker-composition,
provenance, packaging, clean-machine, and private-validation evidence.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
