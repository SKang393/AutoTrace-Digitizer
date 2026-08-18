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

No V28 fixture archive exists yet. Training, visible selection, truth-hidden
public evaluation, marker composition, private validation, manifest creation,
model-store promotion, packaging, production approval, and release remain
unauthorized. Synthetic fixtures are training and public-test inputs only and
are never application graph data.

