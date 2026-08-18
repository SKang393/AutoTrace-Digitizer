<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V27 morphology-veto proposal candidate

V27 is a fresh project-owned defect class designed only from the aggregate
terminal V26 P3 result. V26 retained all 1,024 visible-selection truths with
zero misses or duplicates, but one prohibited false region remained at two
adjacent high thresholds, no required three-threshold zero-error window existed,
and CPU ONNX parity measured `1.1444091796875e-05` even with
`ORT_ENABLE_BASIC`. No V26 case identity, truth row, prediction, fixture pixel,
private image, Chandler image, or `Generalization` label informed V27.

The candidate freezes the exact consumed V26 P3 checkpoint and adds only a
project-owned residual veto head. Its third input contains 24 deterministic
binary projection and morphology values, twelve each for the tight and context
proposal crops. These values describe ink, active rows and columns, peaks,
spans, transitions, and edge, center, and corner occupancy. They expose generic
line, frame, junction, and enclosure structure without article-specific rules.
The output retains the same proposal and eight-role layout. A fixed `0.5` scale
is applied to every final logit, preserving every parent role argmax while
reducing representation-sensitive absolute drift.

The complete research candidate is evaluated in float32 on
`CPUExecutionProvider` with ONNX graph optimization disabled. The fixed output
scale is a preregistered numerical representation change, not a relaxed
tolerance: the maximum absolute ONNX parity error remains `1e-5`. The candidate
remains developer research and is not reachable from the ordinary application
or production model store.

Fresh 256-scene training, 128-scene visible-selection, and 192-scene
truth-hidden public families are registered with new seed offsets, renderer
identities, and degradation identities. They do not reuse V26 fixture bytes or
case identities. P1 is bounded to four epochs and exactly 1,024 optimizer steps.
At most three candidates may execute under this defect class.

Selection still requires three consecutive fixed thresholds with every scene
exact, zero false regions, misses, duplicates, and prohibited hits, recognition
exact at least `0.90`, CER at most `0.05`, overall role accuracy at least
`0.90`, every role at least `0.85`, direct stored-byte execution, tensor hashes,
CPU execution, and ONNX parity at most `1e-5`.

No fixture archive exists yet. Training, visible selection, truth-hidden public
evaluation, marker composition, private validation, manifest creation,
model-store promotion, packaging, production approval, and release remain
unauthorized. Synthetic fixtures are training and public-test inputs only and
are never application graph data.
