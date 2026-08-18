<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V28 relational-neighborhood proposal candidate

V28 is a fresh project-owned defect class designed only from aggregate terminal
V27 P3 evidence. All three V27 candidates retained the same three prohibited
false regions and produced no required three-threshold zero-error window. V27 P3
still preserved every role argmax and passed strict CPU ONNX parity. No V27 case
identity, truth row, prediction, tensor content, fixture pixel, private image,
Chandler image, or `Generalization` label informed V28.

The new proposal head is trained from scratch. It encodes each complete scene as
an ordered proposal graph. Every proposal pair has 19 deterministic,
truth-independent geometry values covering relative position, relative size,
overlap, gap, alignment, plot-side context, and identity. Two
permutation-equivariant message blocks combine those pair values with proposal
crops and checksum-bound production evidence. This is a new scene-level defect
class, not another V27 morphology threshold or veto-head adjustment.

The exact consumed V24 role anchor is frozen only to retain the strong role
classification boundary. Its proposal logits are context features, not the V28
decision. Final role argmax must remain exact. Candidate execution is float32 on
`CPUExecutionProvider` with ONNX graph optimization disabled and a fixed `0.5`
output scale. The maximum allowed PyTorch to ONNX Runtime error remains `1e-5`.

Fresh 256-scene training, 128-scene visible-selection, and 192-scene
truth-hidden public families use new seed offsets, renderer identities,
degradation identities, fixture bytes, and case identities. P1 is bounded to
four epochs and exactly 1,024 optimizer steps. At most three candidates may
execute under this defect class.

Selection requires three consecutive fixed thresholds with every scene exact,
zero false regions, misses, duplicates, and prohibited hits, recognition exact
at least `0.90`, CER at most `0.05`, overall role accuracy at least `0.90`,
every role at least `0.85`, direct stored-byte execution, input and output
tensor hashes, CPU execution, and ONNX parity at most `1e-5`.

The split freeze at source commit
`d49a7b469ea787d2c991383608dd93e6565e4439` materialized the exact fresh
256/128/192 scene archives with zero source-byte overlap. The seal SHA-256 is
`c968aeb5ec0a3440a9fa76b3a346d3652238230599043f801b4fa46ed9eef9bf`.
Train, visible-selection, and truth-hidden public archive SHA-256 values are
`224b5c4025d6ffccbd51b6c0a72aee85cc14e9fa5b2367a3aae4b614523045ba`,
`2968e75bcb09b728785713c87a774bb58325fd9c8371a2bd921689520b697d35`,
and `db00a6bffda5cefe3ecd747d89f930946782f05e7c5a5f013abf06d2a07e0946`.
The public archive remains unopened and has zero evaluations.

P1 executed once for exactly 1,024 optimizer steps and is consumed. Every fixed
threshold retained all 1,024 truth regions with zero false regions, misses,
duplicates, or prohibited hits. Recognition exact was `0.97265625`, CER was
`0.004634994206257242`, role accuracy was `0.99609375`, the frozen V24 role
argmax was preserved exactly, and strict CPU ONNX parity passed at
`4.76837158203125e-6`. Selection still failed because only 124 of 128 scenes
were exact and no required three-threshold passing window exists. The four
aggregate scene failures align with the four retained frozen-parent role
errors; no case identity or case-level result was emitted or inspected.

The tracked aggregate result SHA-256 is
`aa680630cdc1d94941d6864ec0ad8de5c0a9ad7763d37b18b193ad43339dc0be`;
the ignored candidate report and rejected ONNX SHA-256 values are
`af7d6ca29d374880f479977c6b6f740193f549593b0a9a993e5e06c8b0b5c618`
and `788fe3ff7737b3a32db26533fa343477ef4f2d1db73a83f634eba6fbf6054867`.

P2 source is preregistered from those aggregate P1 facts only. It freezes the
exact P1 checkpoint, including its perfect visible-selection proposal stream,
and adds a zero-initialized relational role residual over the frozen P1 scene
nodes, production evidence, and P1 role logits. Only the six residual-layer
parameter tensors may train. P2 is bounded to four epochs and 1,024 optimizer
steps.
Selection additionally requires exact P1 PyTorch and ONNX proposal logits,
identical acceptance at every fixed threshold, the exact P1 full output stream
reexecution hash, unchanged frozen parameters, and strict P2 ONNX parity.
The committed runner source bundle is SHA-256
`a1322bf488796d6d5b673a0182061edffbed8817ccc7b28e6a7a3b1110e0dae1`.
The checksum-bound P2 candidate configuration is SHA-256
`0d2ac3c51c839c630330442a7a96ef83177ab8867e6c9f6b123c120b300477a9`.
The canonical ledger grants P2 exactly one execution after this authorization
checkpoint is committed. It may open only the frozen training and
visible-selection archives.

Truth-hidden public evaluation, marker
composition, private validation, manifest creation, model-store promotion,
packaging, production approval, and release remain unauthorized. Synthetic
fixtures are training and public-test inputs only and are never application
graph data.
