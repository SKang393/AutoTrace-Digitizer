<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V30 unanimous structure-veto candidate

V30 is a fresh project-owned defect class designed only from the tracked
aggregate V29 public result. V29 retained all 1,792 truths with no misses or
duplicates and kept recognition and every role above threshold, but its one
authorized truth-hidden public run passed only 222/224 scenes because it added
two false regions and two prohibited hits at every fixed threshold.

No V29 case identity, truth row, prediction, tensor content, fixture byte,
private image, Chandler image, or `Generalization` label informed V30. The V29
public archive is consumed and closed. Its bytes cannot be used for V30
training, tuning, diagnosis, or evaluation.

V30 replaces averaged proposal logits with a strict unanimous decision across
three independently initialized project-owned experts: relational attention,
permutation-invariant relation summaries, and a local crop structure veto. The
candidate accepts only the minimum positive-vs-negative margin from the three
routes. A single expert can therefore veto a proposal that the other experts
accept. No V29 checkpoint is reused. Roles remain the source-declared
plot-relative deterministic partition.

P1 preregisters a 1,536-step maximum, an asymmetric false-positive objective,
per-route and worst-route losses, route diversity, and hard-negative emphasis.
Fresh 384-scene training, 192-scene visible-selection, and 256-scene
truth-hidden public families use new seed offsets and disjoint renderer and
degradation identities. The three splits must have zero source-byte overlap.

Selection still requires three consecutive fixed thresholds with every scene
exact, zero false regions, misses, duplicates, and prohibited hits,
recognition exact at least `0.90`, CER at most `0.05`, overall role accuracy at
least `0.90`, every role at least `0.85`, direct stored-byte execution, CPU
execution, tensor hashes, and ONNX parity at most `1e-5`.

This checkpoint is source-only. No V30 fixture has been generated, no candidate
is authorized, and no training, selection, public, marker, private, manifest,
model-store, packaging, approval, or release gate is open. Synthetic fixtures
are training and public-test inputs only and can never become application graph
data.
