<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V23 role-anchor set candidate

V23 is a fresh project-owned OCR proposal and role defect class. It uses only
the tracked aggregate terminal result of exhausted V22 P3. It does not inspect
or reuse V22 case identities, truth records, fixture pixels, private data,
Chandler, or the `Generalization` label.

The isolated architecture change replaces all-proposal self-attention with
eight learned role queries. Each query pools one permutation-invariant anchor
from the complete proposal set. Every proposal then attends those anchors and
receives the scene mean and maximum summaries before separate proposal and role
heads. The exact rejected V17 detector, reviewed official English recognizer,
31 evidence features, role order, thresholds, and gates remain fixed.

P1 permits five epochs and exactly 1,280 optimizer steps on 256 fresh synthetic
training scenes. The visible selection has 128 scenes and the truth-hidden
public set has 192 scenes. `SPLIT_SEAL.json` now freezes their disjoint renderer,
degradation, seed, source-byte, proposal, and archive identities from source
commit `ba3ff1737fa0bf7b63c8343539e036a8d493db97`. Cross-split source overlap is
zero, every truth has exactly one production proposal, and the seal records zero
optimizer, selection, and public executions at freeze.

The checksum-bound P1 runner trains the role-anchor model from scratch with
class-balanced proposal cross entropy, worst-negative and worst-positive scene
margin penalties, and class-balanced role cross entropy. It must execute the
exact detector and recognizer on stored training and selection fixture bytes,
hash all three model tensor streams, export a dynamic ONNX model, prove CPU
parity within `1e-5`, and consume only one visible selection. Its config SHA-256
is `89b9a801f1f08223c8dc251ede56a52619464fab11d7438c6b231beb4f395763`.
The three-candidate budget, single public execution, three-threshold robustness
window, zero false/missed/duplicate/prohibited regions, recognition, CER,
overall role, and per-role gates remain mandatory.

P1 consumed all five epochs and exactly 1,280 optimizer steps. Direct stored
fixture-byte execution and all detector, recognizer, and candidate tensor
hashes were recorded. CPU ONNX parity passed at
`9.5367431640625e-06`. The selected `0.35` threshold retained all 1,024
truths with zero misses and duplicates, recognition exact `0.96875`, CER
`0.005302402651201326`, role accuracy `0.99609375`, and lowest per-role
accuracy `0.9765625`. Three false and prohibited regions remained at thresholds
`0.35`, `0.45`, and `0.55`, so no passing robustness window existed. P1 is
consumed and cannot be rerun.

P2 changes only the worst-negative scene margin from probability `0.10` at
weight `1.5` to probability `0.01` at weight `4.0`. It preserves the P1
architecture, from-scratch policy, fixtures, positive margin, balanced role
objective, thresholds, and gates. P2 consumed all five epochs and exactly
1,280 optimizer steps. CPU ONNX parity passed at
`6.198883056640625e-06`, but the selected `0.35` threshold retained only
1,001 of 1,024 truths, left two false and prohibited regions, and reached 51 of
128 exact scenes. Recognition exact was `0.953125`, CER was
`0.019055509527754765`, overall role accuracy was `0.87890625`, and Annotation
accuracy was `0.6640625`. P2 failed selection and cannot be rerun.

Final candidate P3 loads the exact consumed P1 checkpoint, freezes every
encoder, role-anchor, scene-context, update, and role-head parameter, and
fine-tunes only the four proposal-head tensors for two epochs and exactly 512
optimizer steps. Its training-only relative objective forbids positive-logit
decreases from the P1 teacher, targets lower negative logits, and separates
each scene's worst negative from its weakest positive. The isolated change is
based only on aggregate P1 and P2 metrics and preserves P1 role logits by
construction and direct comparison. Only committed P3 training and its one
visible selection are authorized. The truth-hidden public archive remains
unopened. Public execution, marker
composition, manifest creation, model-store promotion, private validation,
production approval, and release eligibility remain unauthorized.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
